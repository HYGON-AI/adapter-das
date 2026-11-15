# SPDX-License-Identifier: Apache-2.0

import torch
from .compat import HAS_TRITON
from .norm_scale_shift_triton import _validate, _apply_parameter

if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _parameter(ptr, row, cols, seq_len, batch_stride: tl.constexpr,
                   row_stride: tl.constexpr, col_stride: tl.constexpr,
                   frame_len: tl.constexpr):
        batch = row // seq_len
        token = row % seq_len
        return tl.load(ptr + batch * batch_stride + (token // frame_len) * row_stride + cols * col_stride)

    @triton.jit
    def _scale_residual_norm_scale_shift_kernel(
        residual_out_ptr, out_ptr, residual_ptr, x_ptr, gate_ptr,
        weight_ptr, bias_ptr, scale_ptr, shift_ptr, seq_len, eps,
        D: tl.constexpr, BLOCK_D: tl.constexpr, HAS_GATE: tl.constexpr,
        HAS_WEIGHT: tl.constexpr, HAS_BIAS: tl.constexpr, RMS: tl.constexpr,
        G: tl.constexpr, SC: tl.constexpr, SH: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK_D)
        mask = cols < D
        safe_cols = tl.minimum(cols, D - 1)
        off = row * D + cols
        res = tl.load(residual_ptr + off, mask=mask, other=0).to(tl.float32)
        xv = tl.load(x_ptr + off, mask=mask, other=0).to(tl.float32)
        if HAS_GATE:
            gate = _parameter(gate_ptr, row, safe_cols, seq_len, G[0], G[1], G[2], G[3]).to(tl.float32)
            xv = xv * gate
        residual = (res + xv).to(x_ptr.dtype.element_ty)
        tl.store(residual_out_ptr + off, residual, mask=mask)
        value = residual.to(tl.float32)
        if RMS:
            centered = tl.where(mask, value, 0.)
        else:
            mean = tl.sum(tl.where(mask, value, 0.), 0) / D
            centered = tl.where(mask, value - mean, 0.)
        var = tl.sum(centered * centered, 0) / D
        normed = centered * tl.rsqrt(var + eps)
        if RMS:
            if HAS_WEIGHT:
                normed = normed * tl.load(weight_ptr + cols, mask=mask, other=1).to(tl.float32)
        elif HAS_WEIGHT and HAS_BIAS:
            normed = normed * tl.load(weight_ptr + cols, mask=mask, other=1).to(tl.float32) + tl.load(bias_ptr + cols, mask=mask, other=0).to(tl.float32)
        normed = normed.to(x_ptr.dtype.element_ty)
        sc = _parameter(scale_ptr, row, safe_cols, seq_len, SC[0], SC[1], SC[2], SC[3])
        sh = _parameter(shift_ptr, row, safe_cols, seq_len, SH[0], SH[1], SH[2], SH[3])
        one_plus_scale = (1. + sc.to(tl.float32)).to(scale_ptr.dtype.element_ty)
        product = normed.to(tl.float32) * one_plus_scale.to(tl.float32)
        if out_ptr.dtype.element_ty != tl.float32 and scale_ptr.dtype.element_ty == out_ptr.dtype.element_ty:
            product = product.to(out_ptr.dtype.element_ty).to(tl.float32)
        tl.store(out_ptr + off, product + sh.to(tl.float32), mask=mask)


def _layout(parameter, shape):
    b, s, d = shape
    if parameter is None:
        return (0, 0, 0, 1)
    if parameter.ndim == 1:
        return (0, 0, 0 if parameter.numel() == 1 else parameter.stride(0), 1)
    if parameter.ndim == 2:
        return (0 if parameter.shape[0] == 1 else parameter.stride(0), 0, parameter.stride(1), 1)
    if parameter.ndim == 3:
        return (0 if parameter.shape[0] == 1 else parameter.stride(0),
                0 if parameter.shape[1] == 1 else parameter.stride(1), parameter.stride(2), 1)
    return (parameter.stride(0), parameter.stride(1), parameter.stride(3), s // parameter.shape[1])


def _torch_fallback(residual, x, gate, weight, bias, scale, shift, norm_type, eps):
    import torch.nn.functional as F
    product = x.float()
    if gate is not None:
        if gate.ndim == 4:
            frames = gate.shape[1]
            product = (product.unflatten(1, (frames, x.shape[1] // frames)) * gate.float()).flatten(1, 2)
        else:
            product = product * (gate[:, None] if gate.ndim == 2 else gate).float()
    res = (residual.float() + product).to(x.dtype)
    value = res.float()
    if norm_type == 'layer':
        affine = weight is not None and bias is not None
        value = F.layer_norm(value, (x.shape[-1],), weight.float() if affine else None,
                             bias.float() if affine else None, eps)
    else:
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
        if weight is not None:
            value = value * weight.float()
    value = value.to(x.dtype)
    value = _apply_parameter(value, scale, scale=True)
    return _apply_parameter(value, shift).to(x.dtype), res


def fused_scale_residual_norm_scale_shift(residual, x, gate, weight, bias, scale, shift, norm_type, eps=1e-5):
    _validate(x, weight, bias, scale, shift, norm_type)
    if residual.shape != x.shape or residual.device != x.device or residual.dtype != x.dtype:
        raise ValueError('residual must match x shape, dtype and device')
    if gate is not None:
        _validate(x, None, None, gate, gate, norm_type)
    if not (HAS_TRITON and x.is_cuda and torch.version.hip is not None and x.is_contiguous() and residual.is_contiguous()):
        return _torch_fallback(residual, x, gate, weight, bias, scale, shift, norm_type, eps)
    output, residual_out = torch.empty_like(x), torch.empty_like(x)
    if x.numel():
        hidden = x.shape[-1]
        _scale_residual_norm_scale_shift_kernel[(x.shape[0] * x.shape[1],)](
            residual_out, output, residual, x, gate if gate is not None else x,
            weight if weight is not None else x, bias if bias is not None else x,
            scale, shift, x.shape[1], eps, D=hidden, BLOCK_D=triton.next_power_of_2(hidden),
            HAS_GATE=gate is not None, HAS_WEIGHT=weight is not None, HAS_BIAS=bias is not None,
            RMS=norm_type == 'rms', G=_layout(gate, x.shape), SC=_layout(scale, x.shape), SH=_layout(shift, x.shape),
            num_warps=4, enable_fp_fusion=False,
        )
    return output, residual_out
