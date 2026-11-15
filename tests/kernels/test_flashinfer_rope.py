# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import re
from unittest.mock import patch

import sgl_kernel
from hcu_adapter._base import Missing


def _api():
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace

    return apply_rope_with_cos_sin_cache_inplace


def test_flashinfer_rope_is_implemented():
    assert not isinstance(_api(), Missing)


def test_flashinfer_rope_schema():
    schema = torch.library.infer_schema(_api(), mutates_args=("query", "key"))
    assert re.search(r"Tensor\(\w+!\) query", schema)
    assert re.search(r"Tensor\(\w+!\) key", schema)
    assert schema.endswith("-> ()")


def test_flashinfer_rope_uses_sgl_kernel():
    assert _api() is sgl_kernel.apply_rope_with_cos_sin_cache_inplace
    assert _api().__module__ == "sgl_kernel.rope_triton"


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU")


def _reference(x, positions, cache, head_size, neox):
    n = x.shape[0]
    y = x.detach().cpu().double().reshape(n, -1, head_size)
    table = cache.detach().cpu().double()[positions.cpu().long()]
    cos, sin = (part[:, None, :] for part in table.chunk(2, -1))
    half = cos.shape[-1]
    prefix = y[..., :2 * half]
    a, b = prefix.chunk(2, -1) if neox else (prefix[..., ::2], prefix[..., 1::2])
    out = y.clone()
    if neox:
        out[..., :half] = a * cos - b * sin
        out[..., half:2 * half] = b * cos + a * sin
    else:
        out[..., :2 * half:2] = a * cos - b * sin
        out[..., 1:2 * half:2] = b * cos + a * sin
    return out.reshape_as(x)


def _inputs(dtype, partial=False, tokens=13):
    d, hq, hk = 128, 4, 2
    angles = torch.randn(29, 32 if partial else 64, device="cuda")
    cache = torch.cat((angles.cos(), angles.sin()), -1)
    pos = torch.randperm(29, device="cuda")[:tokens]
    pad = 7 if partial else 0
    q = torch.randn(tokens, hq * d + pad, device="cuda", dtype=dtype)[:, :hq * d]
    k = torch.randn(tokens, hk * d + pad, device="cuda", dtype=dtype)[:, :hk * d]
    return pos, q, k, d, cache


def _check(got, expected):
    tol = (0.008, 0.002) if got.dtype == torch.bfloat16 else (
        (0.001, 0.001) if got.dtype == torch.float16 else (1e-5, 1e-6)
    )
    torch.testing.assert_close(got.cpu().double(), expected, rtol=tol[0], atol=tol[1])


@gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("neox", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_flashinfer_rope_numerics(dtype, neox, partial):
    torch.manual_seed(31)
    pos, q, k, d, cache = _inputs(dtype, partial)
    q0, k0, table = q.clone(), k.clone(), cache.clone()
    expected = [_reference(x, pos, cache, d, neox) for x in (q, k)]
    addresses = q.data_ptr(), k.data_ptr()
    assert _api()(pos, q, k, d, cache, is_neox=neox) is None
    assert addresses == (q.data_ptr(), k.data_ptr())
    for got, ref in zip((q, k), expected):
        _check(got, ref)
    assert torch.equal(cache, table)
    if partial:
        for got, original in ((q, q0), (k, k0)):
            assert torch.equal(got.view(13, -1, d)[..., 64:], original.view(13, -1, d)[..., 64:])


@gpu
@pytest.mark.parametrize("neox", [False, True])
def test_flashinfer_rope_changed_input_graph(neox):
    torch.manual_seed(32)
    pos, q, k, d, cache = _inputs(torch.bfloat16, partial=True)
    fn = _api()
    for _ in range(3):
        fn(pos, q, k, d, cache, neox)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(pos, q, k, d, cache, neox)
    for seed in [41, 42, 43]:
        torch.manual_seed(seed)
        q.normal_(); k.normal_()
        pos.copy_(torch.randperm(29, device="cuda")[:13])
        angle = torch.randn(29, 32, device="cuda")
        cache.copy_(torch.cat((angle.cos(), angle.sin()), -1))
        expected = [_reference(x, pos, cache, d, neox) for x in (q, k)]
        graph.replay()
        for got, ref in zip((q, k), expected):
            _check(got, ref)


@gpu
def test_flashinfer_rope_empty():
    pos, q, k, d, cache = _inputs(torch.bfloat16, tokens=0)
    assert _api()(pos, q, k, d, cache, False) is None


@gpu
@pytest.mark.parametrize("neox", [False, True])
@pytest.mark.parametrize("tokens", [1, 17, 129])
def test_inplace_autotune_and_strides(neox, tokens):
    from sgl_kernel.rope_triton import _rotary_embedding_kernel as kernel

    kernel.cache.clear()
    d, rotary = 96, 66
    q_storage = torch.randn(tokens, 3 * d + 11, device="cuda", dtype=torch.bfloat16)
    k_storage = torch.randn(tokens, d + 11, device="cuda", dtype=torch.bfloat16)
    q, k = q_storage[:, :3 * d], k_storage[:, :d]
    pos = torch.randint(0, 19, (tokens * 2,), dtype=torch.int32, device="cuda")[::2]
    cache = torch.randn(19, rotary * 2, device="cuda")[:, ::2]
    padding = [q_storage[:, 3 * d:].clone(), k_storage[:, d:].clone()]
    for round_id in range(2):
        q.normal_(); k.normal_()
        q0, k0 = q.clone(), k.clone()
        expected = [_reference(x, pos, cache, d, neox) for x in (q, k)]
        if round_id == 0:
            _api()(pos, q, k, d, cache, neox)
            assert len(kernel.cache) == 2
        else:
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            with patch.object(kernel, "_bench", side_effect=AssertionError("unexpected retune")):
                _api()(pos, q, k, d, cache, neox)
            torch.cuda.synchronize()
            assert torch.cuda.max_memory_allocated() == allocated
        for got, ref, before in zip((q, k), expected, (q0, k0)):
            _check(got, ref)
            assert torch.equal(got.view(tokens, -1, d)[..., rotary:], before.view(tokens, -1, d)[..., rotary:])
        assert torch.equal(q_storage[:, 3 * d:], padding[0])
        assert torch.equal(k_storage[:, d:], padding[1])


@gpu
def test_real_diffusion_rope_helper():
    from sglang.multimodal_gen.runtime.platforms import current_platform

    if not current_platform.is_cuda():
        pytest.skip("requires diffusion CUDA platform semantics")
    from sglang.multimodal_gen.runtime.layers.rotary_embedding.utils import (
        apply_flashinfer_rope_qk_inplace,
        flashinfer_apply_rope_inplace,
    )

    assert flashinfer_apply_rope_inplace is not None
    pos, q, k, d, cache = _inputs(torch.bfloat16, tokens=13)
    q = torch.randn(1, 13, 40, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    expected = [_reference(x.flatten(0, 1).flatten(1), pos, cache, d, False) for x in (q, k)]
    result = apply_flashinfer_rope_qk_inplace(q, k, cache, positions=pos, is_neox=False)
    for got, ref in zip(result, expected):
        _check(got.flatten(0, 1).flatten(1), ref)
