# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch
import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import residual_norm_scale_shift_triton as impl


def expand(p, shape):
    b, s, d = shape
    if p.ndim == 4:
        return p.expand(b, p.shape[1], s // p.shape[1], d).reshape(shape)
    if p.ndim == 2:
        p = p[:, None]
    return p.expand(shape)


def reference(residual, x, gate, weight, bias, scale, shift, norm_type, eps):
    product = x.float() if gate is None else x.float() * expand(gate, x.shape).float()
    res = (residual.float() + product).to(x.dtype)
    v = res.float()
    if norm_type == 'layer':
        v = F.layer_norm(v, (x.shape[-1],), weight.float() if weight is not None and bias is not None else None,
                         bias.float() if weight is not None and bias is not None else None, eps)
    else:
        v = v * torch.rsqrt(v.square().mean(-1, keepdim=True) + eps)
        if weight is not None: v = v * weight.float()
    v = v.to(x.dtype)
    sc, sh = expand(scale, x.shape), expand(shift, x.shape)
    y = (v * (1 + sc) + sh).to(x.dtype)
    return y, res


def operands(device, dtype, layout, d=5120):
    b,s=2,6
    x=torch.randn(b,s,d,device=device,dtype=dtype)
    residual=torch.randn_like(x)
    shapes={'scalar':(1,), 'vector':(d,), 'batch':(b,1,d), 'token':(b,s,d), 'frame':(b,2,1,d)}
    shape=shapes[layout]
    gate=None if layout=='scalar' else torch.randn(shape,device=device,dtype=torch.float32)
    scale=torch.randn(shape,device=device,dtype=torch.float32)
    shift=torch.randn((1,),device=device,dtype=dtype) if layout=='frame' else torch.randn(shape,device=device,dtype=torch.float32)
    w=torch.randn(d,device=device); bias=torch.randn_like(w)
    return residual,x,gate,w,bias,scale,shift


@pytest.mark.parametrize('norm_type',['layer','rms'])
@pytest.mark.parametrize('layout',['scalar','vector','batch','token','frame'])
def test_cpu_contract(norm_type,layout):
    args=operands('cpu',torch.bfloat16,layout,d=256)
    got=impl.fused_scale_residual_norm_scale_shift(*args,norm_type,1e-6)
    expected=reference(*args,norm_type,1e-6)
    for a,b in zip(got,expected): torch.testing.assert_close(a,b,rtol=0,atol=0)
    for output in got:
        for source in args:
            if source is not None: assert output.data_ptr()!=source.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires HCU')
@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float16,torch.float32])
@pytest.mark.parametrize('norm_type',['layer','rms'])
@pytest.mark.parametrize('layout',['scalar','vector','batch','token','frame'])
def test_gpu_numerics_graph_and_inputs(dtype,norm_type,layout):
    args=operands('cuda',dtype,layout)
    snapshots=[p.clone() if p is not None else None for p in args]
    with patch.object(impl,'_torch_fallback',side_effect=AssertionError('unexpected fallback')):
        for _ in range(3): got=impl.fused_scale_residual_norm_scale_shift(*args,norm_type,1e-6)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): captured=impl.fused_scale_residual_norm_scale_shift(*args,norm_type,1e-6)
        for p,snapshot in zip(args,snapshots):
            if p is not None: assert torch.equal(p,snapshot)
        for seed in [11,19]:
            torch.manual_seed(seed)
            for p in args:
                if p is not None: p.normal_()
            eager=impl.fused_scale_residual_norm_scale_shift(*args,norm_type,1e-6)
            graph.replay()
            for a,b in zip(captured,eager): assert torch.equal(a,b)
            ref=reference(*args,norm_type,1e-6)
            torch.testing.assert_close(eager[1],ref[1],rtol=0,atol=0)
            torch.testing.assert_close(eager[0],ref[0],rtol=0.008 if dtype==torch.bfloat16 else 0.002,atol=0.02 if dtype==torch.bfloat16 else 0.003)


@pytest.mark.parametrize('hidden',[256,4096,8192])
@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires HCU')
def test_gpu_no_affine_scalar_modulation_and_partial_affine(hidden):
    x=torch.randn(1,13,hidden,device='cuda',dtype=torch.bfloat16)
    residual=torch.randn_like(x)
    scale=torch.randn(1,device='cuda',dtype=torch.bfloat16)
    shift=torch.randn_like(scale)
    for weight,bias in [(None,None),(torch.randn(hidden,device='cuda'),None),(None,torch.randn(hidden,device='cuda'))]:
        args=(residual,x,None,weight,bias,scale,shift,'layer',1e-5)
        got=impl.fused_scale_residual_norm_scale_shift(*args)
        expected=reference(*args)
        torch.testing.assert_close(got[0],expected[0],rtol=0.008,atol=0.02)
        assert torch.equal(got[1],expected[1])


def test_empty_strided_and_invalid_contract():
    x=torch.empty(2,0,256)
    y,res=impl.fused_scale_residual_norm_scale_shift(x,x,None,None,None,torch.ones(1),torch.zeros(1),'layer')
    assert y.shape==res.shape==x.shape
    base=torch.randn(2,6,256)
    x=base[:,::2]
    args=(x,x,None,None,None,torch.ones(1),torch.zeros(1),'rms',1e-6)
    for a,b in zip(impl.fused_scale_residual_norm_scale_shift(*args),reference(*args)):
        assert torch.equal(a,b)
    with pytest.raises(ValueError):
        impl.fused_scale_residual_norm_scale_shift(x,x,None,None,None,torch.ones(5),torch.zeros(1),'layer')
    with pytest.raises(ValueError):
        impl.fused_scale_residual_norm_scale_shift(x[:1],x,None,None,None,torch.ones(1),torch.zeros(1),'layer')
