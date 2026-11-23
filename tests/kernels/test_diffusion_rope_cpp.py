# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch

from sgl_kernel import diffusion_rope_cpp as hip


pytestmark = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="requires HIP",
)


def _rope_reference(value, rope):
    pairs = value.float().unflatten(-1, (-1, 2))
    rotated = torch.view_as_complex(pairs) * rope[None, :, None]
    return torch.view_as_real(rotated).flatten(-2).to(value.dtype)


def _norm_rope_reference(value, weight, rope, eps):
    fp32 = value.contiguous().float()
    normalized = (fp32 * torch.rsqrt(fp32.square().mean(-1, keepdim=True) + eps)).to(value.dtype)
    return _rope_reference(weight * normalized, rope)


def _tensor(shape):
    return torch.randn(shape, device="cuda", dtype=torch.bfloat16)


def _rope(tokens):
    theta = torch.randn((tokens, 64), device="cuda")
    return torch.polar(torch.ones_like(theta), theta)


@pytest.fixture(scope="module")
def dispatch():
    pytest.importorskip("sglang")
    from sglang.kernels.ops.diffusion.rope.complex_rope_triton import fused_complex_rope
    from sglang.kernels.ops.diffusion.rope.qknorm_complex_rope_jit import (
        qknorm_complex_rope_cuda,
        qknorm_complex_rope_pack_,
    )
    from sglang.kernels.ops.diffusion.rope.qknorm_complex_rope_kv_triton import qknorm_complex_rope_kv
    from sglang.kernels.ops.diffusion.rope.qknorm_complex_rope_triton import qknorm_complex_rope

    from hcu_adapter import diffusion_rope

    diffusion_rope.install()
    diffusion_rope.install()
    return fused_complex_rope, qknorm_complex_rope_cuda, qknorm_complex_rope, qknorm_complex_rope_kv, qknorm_complex_rope_pack_


@pytest.mark.parametrize("tokens", [18, 4096])
def test_nonmutating_ops(dispatch, tokens):
    torch.manual_seed(42)
    pure, cuda_entry, triton_entry, kv, _ = dispatch
    k, v = (_tensor((1, tokens, 32, 128)) for _ in range(2))
    kp, vp = (_tensor((1, 18, 32, 128)) for _ in range(2))
    rope, weight = _rope(tokens), _tensor((128,))
    saved = k.clone()
    normalized = _norm_rope_reference(k, weight, rope, 1e-6)
    with patch.object(hip, "_norm_rope_reference", side_effect=AssertionError("norm fallback")), \
         patch.object(hip, "_rope_reference", side_effect=AssertionError("rope fallback")):
        assert torch.equal(pure(k, rope), _rope_reference(k, rope))
        for entry in (cuda_entry, triton_entry):
            result = entry(k, weight, rope, 1e-6)
            assert torch.equal(result, normalized)
            assert result.data_ptr() != k.data_ptr()
        got_k, got_v = kv(k, weight, rope, v, kp, vp, 1e-6)
    assert torch.equal(got_k, torch.cat((kp, normalized), dim=1))
    assert torch.equal(got_v, torch.cat((vp, v), dim=1))
    assert torch.equal(k, saved)


@pytest.mark.parametrize("projected_into_outputs", [False, True])
def test_mutating_pack_and_graph_replay(dispatch, projected_into_outputs):
    torch.manual_seed(17)
    *_, pack = dispatch
    batch, tokens, prefix, heads = 2, 17, 5, 32
    storage = _tensor((batch, prefix + tokens, 3, heads, 128))
    q = storage[:, prefix:, 0]
    k_out, v_out = storage[:, :, 1], storage[:, :, 2]
    if projected_into_outputs:
        k_src = v_src = None
    else:
        k_src, v_src = (_tensor(q.shape) for _ in range(2))
    kp, vp = (_tensor((batch, prefix, heads, 128)) for _ in range(2))
    qw, kw, rope = _tensor((128,)), _tensor((128,)), _rope(tokens)

    def reference():
        k = k_out[:, prefix:] if k_src is None else k_src
        v = v_out[:, prefix:] if v_src is None else v_src
        return (
            _norm_rope_reference(q, qw, rope, 1e-6),
            torch.cat((kp, _norm_rope_reference(k, kw, rope, 1e-6)), dim=1),
            torch.cat((vp, v), dim=1),
        )

    def launch():
        return pack(q, k_out, v_out, qw, kw, rope, kp, vp, k_src, v_src, 1e-6)

    with patch.object(hip, "_norm_rope_reference", side_effect=AssertionError("pack fallback")):
        expected = reference()
        assert launch() is None
        assert all(torch.equal(actual, ref) for actual, ref in zip((q, k_out, v_out), expected))
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        for _ in range(2):
            storage.normal_()
            if k_src is not None:
                k_src.normal_()
                v_src.normal_()
            expected = reference()
            graph.replay()
            assert all(torch.equal(actual, ref) for actual, ref in zip((q, k_out, v_out), expected))


def test_unsupported_hip_layout_returns_reference(dispatch):
    pure, _, triton_entry, _, _ = dispatch
    value, weight, rope = _tensor((1, 3, 1, 128)), _tensor((128,)), _rope(3)
    assert torch.equal(pure(value, rope), _rope_reference(value, rope))
    value, weight = value.half(), weight.half()
    assert torch.equal(triton_entry(value, weight, rope, 1e-6),
                       _norm_rope_reference(value, weight, rope, 1e-6))
