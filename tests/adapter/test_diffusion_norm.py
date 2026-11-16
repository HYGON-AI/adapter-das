# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch

from hcu_adapter import diffusion_norm
from sgl_kernel.norm_scale_shift_triton import fused_norm_scale_shift


def test_install_is_import_lazy():
    code = '''
import sys, torch
from hcu_adapter import diffusion_norm
diffusion_norm.install()
diffusion_norm.install()
assert not torch.cuda.is_initialized()
assert not any(n == "sglang" or n.startswith("sglang.") for n in sys.modules)
assert torch._C._dispatch_has_kernel_for_dispatch_key("sglang::fused_norm_scale_shift", "CUDA")
assert torch._C._dispatch_has_kernel_for_dispatch_key("sglang::fused_scale_residual_norm_scale_shift", "CUDA")
'''
    result = subprocess.run([sys.executable, "-c", code], env={**os.environ, "HCU_ADAPTER_HOST": "none"},
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU")


@gpu
def test_real_custom_op_alias_schema_fake_and_layer_dispatch():
    from sglang.kernels.ops.diffusion.norm import scale_residual_norm_cutedsl as upstream
    from sglang.multimodal_gen.runtime.layers.layernorm import LayerNormScaleShift

    cached = upstream.fused_norm_scale_shift
    schema = str(cached._opoverload._schema)
    caller = LayerNormScaleShift.forward_cuda
    diffusion_norm.install()
    diffusion_norm.install()
    assert upstream.fused_norm_scale_shift is cached
    assert str(cached._opoverload._schema) == schema
    assert LayerNormScaleShift.forward_cuda is caller
    x = torch.randn(1, 17, 5120, device="cuda", dtype=torch.bfloat16)
    scale, shift = torch.randn(1, 1, 5120, device="cuda"), torch.randn(1, 1, 5120, device="cuda")
    expected = fused_norm_scale_shift(x, None, None, scale, shift, "layer", 1e-6)
    layer = LayerNormScaleShift(5120, eps=1e-6, elementwise_affine=False)
    with patch.object(upstream.cute, "compile", side_effect=AssertionError("CuTeDSL reached")):
        assert torch.equal(cached(x, None, None, scale, shift, "layer", 1e-6), expected)
        assert torch.equal(torch.ops.sglang.fused_norm_scale_shift(x, None, None, scale, shift, "layer", 1e-6), expected)
        assert torch.equal(layer(x, shift, scale), expected)
    cpu_x = torch.randn(1, 3, 5120)
    cpu_mod = torch.randn(1, 1, 5120)
    sentinel = torch.full_like(cpu_x, 7)
    with patch(
        "sglang.kernels.kda_kernels.norm_scale_shift_jit.try_fused_norm_scale_shift",
        return_value=sentinel,
    ):
        assert cached(cpu_x, None, None, cpu_mod, cpu_mod, "layer") is sentinel
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        fake_x = torch.empty(1, 17, 5120, device="cuda", dtype=torch.bfloat16)
        fake_mod = torch.empty(1, 1, 5120, device="cuda")
        fake_y = cached(fake_x, None, None, fake_mod, fake_mod, "layer")
        assert fake_y.shape == fake_x.shape and fake_y.dtype == fake_x.dtype


@gpu
def test_residual_custom_op_preserves_alias_schema_fake_and_consumer():
    from sglang.kernels.ops.diffusion.norm import scale_residual_norm_cutedsl as upstream
    from sglang.multimodal_gen.runtime.layers.layernorm import ScaleResidualLayerNormScaleShift
    from sgl_kernel.residual_norm_scale_shift_triton import fused_scale_residual_norm_scale_shift

    cached = upstream.fused_scale_residual_norm_scale_shift
    schema = str(cached._opoverload._schema)
    caller = ScaleResidualLayerNormScaleShift.forward_cuda
    diffusion_norm.install()
    assert cached is upstream.fused_scale_residual_norm_scale_shift
    assert str(cached._opoverload._schema) == schema
    assert ScaleResidualLayerNormScaleShift.forward_cuda is caller
    residual = torch.randn(1, 17, 5120, device='cuda', dtype=torch.bfloat16)
    x = torch.randn_like(residual)
    gate = torch.randn(1, 1, 5120, device='cuda')
    scale = torch.zeros(1, device='cuda', dtype=torch.bfloat16)
    shift = torch.zeros_like(scale)
    layer = ScaleResidualLayerNormScaleShift(5120, eps=1e-6, elementwise_affine=True).cuda()
    args = (residual, x, gate, layer.norm.weight, layer.norm.bias, scale, shift, 'layer', 1e-6)
    expected = fused_scale_residual_norm_scale_shift(*args)
    with patch.object(upstream.cute, 'compile', side_effect=AssertionError('CuTeDSL reached')):
        for actual in [cached(*args), torch.ops.sglang.fused_scale_residual_norm_scale_shift(*args), layer(residual, x, gate, shift, scale)]:
            for a,b in zip(actual,expected): assert torch.equal(a,b)
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        fake = torch.empty(1,17,5120,device='cuda',dtype=torch.bfloat16)
        mod = torch.empty(1,device='cuda')
        outputs = cached(fake,fake,None,None,None,mod,mod,'layer')
        assert len(outputs)==2
        for output in outputs: assert output.shape==fake.shape and output.dtype==fake.dtype
