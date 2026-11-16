# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch
import importlib.util

import pytest
import torch
import torch.nn.functional as F

from sgl_kernel import norm_scale_shift_triton as implementation


def test_without_triton_uses_torch():
    import sgl_kernel.compat

    spec = importlib.util.spec_from_file_location(
        "sgl_kernel._norm_scale_shift_without_triton", implementation.__file__
    )
    module = importlib.util.module_from_spec(spec)
    with patch.object(sgl_kernel.compat, "HAS_TRITON", False):
        spec.loader.exec_module(module)
    x = torch.randn(1, 3, 256)
    scale, shift = torch.randn(256), torch.randn(256)
    got = module.fused_norm_scale_shift(x, None, None, scale, shift, "layer")
    expected = implementation.fused_norm_scale_shift(x, None, None, scale, shift, "layer")
    assert torch.equal(got, expected)


def _reference(x, weight, bias, scale, shift, norm_type, eps):
    value = x.float()
    if norm_type == "layer":
        affine = weight is not None and bias is not None
        value = F.layer_norm(value, (x.shape[-1],), weight.float() if affine else None,
                             bias.float() if affine else None, eps)
    else:
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
        if weight is not None:
            value = value * weight.float()
    value = value.to(x.dtype)
    for parameter, is_scale in [(scale, True), (shift, False)]:
        if parameter.ndim == 4:
            frames = parameter.shape[1]
            value = value.unflatten(1, (frames, x.shape[1] // frames))
            value = value * (1 + parameter) if is_scale else value + parameter
            value = value.flatten(1, 2)
        else:
            if parameter.ndim == 2:
                parameter = parameter[:, None]
            value = value * (1 + parameter) if is_scale else value + parameter
    return value.to(x.dtype)


@pytest.mark.parametrize("norm_type", ["layer", "rms"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layout", ["scalar", "vector", "batch", "token", "frame", "mixed"])
def test_general_cpu_contract(norm_type, dtype, layout):
    torch.manual_seed(71)
    b, s, d = 2, 6, 256
    x = torch.randn(b, s, d, dtype=dtype)
    weight, bias = torch.randn(d), torch.randn(d)
    shapes = {"scalar": (1,), "vector": (d,), "batch": (b, d),
              "token": (b, s, d), "frame": (b, 2, 1, d), "mixed": (b, 2, 1, d)}
    scale = torch.randn(shapes[layout])
    shift = torch.randn((b, s, d) if layout == "mixed" else shapes[layout])
    before = [t.clone() for t in (x, weight, bias, scale, shift)]
    expected = _reference(x, weight, bias, scale, shift, norm_type, 1e-5)
    got = implementation.fused_norm_scale_shift(x, weight, bias, scale, shift, norm_type)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    assert got.shape == x.shape and got.dtype == x.dtype
    assert got.data_ptr() != x.data_ptr()
    for current, saved in zip((x, weight, bias, scale, shift), before):
        assert torch.equal(current, saved)


def test_partial_affine_preserves_upstream_layer_semantics():
    x = torch.randn(1, 3, 256)
    scale, shift = torch.randn(256), torch.randn(256)
    expected = implementation.fused_norm_scale_shift(x, None, None, scale, shift, "layer")
    for weight, bias in [(torch.randn(256), None), (None, torch.randn(256))]:
        got = implementation.fused_norm_scale_shift(x, weight, bias, scale, shift, "layer")
        assert torch.equal(got, expected)


@pytest.mark.parametrize("case", ["hidden", "scale", "norm", "weight"])
def test_invalid_contract_rejected(case):
    x = torch.randn(1, 3, 256)
    weight = bias = None
    scale, shift = torch.randn(256), torch.randn(256)
    norm_type = "layer"
    if case == "hidden": x = x[..., :255]
    if case == "scale": scale = torch.randn(2, 7, 256)
    if case == "norm": norm_type = "unknown"
    if case == "weight": weight = torch.randn(255)
    with pytest.raises(ValueError):
        implementation.fused_norm_scale_shift(x, weight, bias, scale, shift, norm_type)


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU")


@gpu
@pytest.mark.parametrize("hidden", [4096, 5120])
@pytest.mark.parametrize("tokens", [0, 1, 17, 129, 513])
def test_triton_numerics(hidden, tokens):
    torch.manual_seed(72)
    x = torch.randn(1, tokens, hidden, device="cuda", dtype=torch.bfloat16)
    scale, shift = torch.randn(1, 1, hidden, device="cuda"), torch.randn(1, 1, hidden, device="cuda")
    normalized = F.layer_norm(x.float(), (hidden,), None, None, 1e-6).bfloat16()
    expected = (normalized.float() * (1 + scale) + shift).bfloat16()
    with patch.object(implementation, "_apply_parameter", side_effect=AssertionError("fast path missed")):
        got = implementation.fused_norm_scale_shift(x, None, None, scale, shift, "layer", 1e-6)
    torch.testing.assert_close(got, expected, rtol=0.008, atol=0.002)


@gpu
@pytest.mark.parametrize("hidden", [4096, 5120])
def test_changed_input_graph_and_strided_parameters(hidden):
    x = torch.randn(2, 17, hidden, device="cuda", dtype=torch.bfloat16)
    storage = torch.randn(2, 6, hidden, device="cuda")
    scale, shift = storage[:, 1:2], storage[:, 3:4]
    fn = implementation.fused_norm_scale_shift
    for _ in range(3): fn(x, None, None, scale, shift, "layer", 1e-6)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fn(x, None, None, scale, shift, "layer", 1e-6)
    for seed in [81, 82, 83]:
        torch.manual_seed(seed)
        x.normal_(); storage.normal_()
        eager = fn(x, None, None, scale, shift, "layer", 1e-6)
        graph.replay()
        assert torch.equal(captured, eager)


@gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("norm_type", ["layer", "rms"])
def test_gpu_general_fallback(dtype, norm_type):
    x = torch.randn(2, 6, 256, device="cuda", dtype=dtype)
    weight, bias = torch.randn(256, device="cuda"), torch.randn(256, device="cuda")
    scale, shift = torch.randn(2, 2, 1, 256, device="cuda"), torch.randn(2, 6, 256, device="cuda")
    expected = _reference(x, weight, bias, scale, shift, norm_type, 1e-5)
    got = implementation.fused_norm_scale_shift(x, weight, bias, scale, shift, norm_type)
    assert torch.equal(got, expected)
