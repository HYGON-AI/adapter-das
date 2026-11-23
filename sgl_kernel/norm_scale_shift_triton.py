# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
import torch.nn.functional as F
from .compat import HAS_TRITON

if HAS_TRITON:
    import triton
    import triton.language as tl

if HAS_TRITON:
    @triton.jit
    def _rcp4(x):
        return tl.inline_asm_elementwise(
            "v_rcp_f32 $0, $1;", constraints="=v,v", args=[x],
            dtype=tl.float32, is_pure=True, pack=1,
        )



    @triton.jit
    def div_rn_f32(x, y):
        return tl.div_rn(x, y)



    @triton.jit
    def rsqrt_f32(x):
        return tl.rsqrt(x)



    @triton.jit
    def round_bf16_to_fp32(value):
        bits = value.to(tl.int32, bitcast=True)
        rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & -65536
        return rounded.to(tl.float32, bitcast=True)



    @triton.jit
    def _welford_push(val, mean, m2, cnt, valid, MASKED: tl.constexpr,
                      SEPARATE_M2: tl.constexpr = False):
        # ``valid`` masks lanes whose aten thread does not execute this
        # iteration (their state must stay untouched).
        delta = val - mean
        new_cnt = cnt + 1.0
        recip = _rcp4(new_cnt)
        new_mean = tl.fma(delta, recip, mean)
        t = val - new_mean
        if SEPARATE_M2:
            new_m2 = delta * t + m2
        else:
            new_m2 = tl.fma(delta, t, m2)
        if MASKED:
            new_mean = tl.where(valid, new_mean, mean)
            new_m2 = tl.where(valid, new_m2, m2)
            new_cnt = tl.where(valid, new_cnt, cnt)
        return new_mean, new_m2, new_cnt



    @triton.jit
    def _welford_combine(mean_b, m2_b, cnt_b, mean_a, m2_a, cnt_a):
        # b = self / lower lane, a = other / upper lane; the op order matters.
        count = cnt_a + cnt_b
        pos = count > 0.0
        coef = _rcp4(tl.where(pos, count, 1.0))
        delta = mean_b - mean_a
        n_b = coef * cnt_b
        d2 = delta * delta
        n_a = cnt_a * coef
        s = m2_a + m2_b
        t1 = n_b * mean_b
        mean = tl.fma(mean_a, n_a, t1)
        t2 = cnt_a * d2
        m2 = tl.fma(n_b, t2, s)
        mean = tl.where(pos, mean, 0.0)
        m2 = tl.where(pos, m2, 0.0)
        return mean, m2, count



    @triton.jit
    def _split_halves(x, rows: tl.constexpr, half: tl.constexpr):
        # (rows, 2*half) -> two (rows, half) tensors pairing lane i with i+half,
        # the tree a shfl.down fold with the largest offset first produces.
        x = tl.reshape(x, (rows, 2, half), can_reorder=False)
        x = tl.permute(x, (0, 2, 1))
        return tl.split(x)



    @triton.jit
    def _fold_halves(mean, m2, cnt, rows: tl.constexpr, half: tl.constexpr):
        mb, ma = _split_halves(mean, rows, half)
        sb, sa = _split_halves(m2, rows, half)
        cb, ca = _split_halves(cnt, rows, half)
        return _welford_combine(mb, sb, cb, ma, sa, ca)



    @triton.jit
    def _push_vec4(
        x4,
        mean,
        m2,
        cnt,
        valid,
        rows: tl.constexpr,
        lanes: tl.constexpr,
        MASKED: tl.constexpr,
        BF16_ATEN_M2: tl.constexpr = False,
    ):
        # Push the 4 elements of one aligned_vector<bf16, 4> in exact serial
        # order.  x4 is (rows, lanes, 4) fp32.
        g = tl.reshape(x4, (rows, lanes, 2, 2), can_reorder=False)
        p02, p13 = tl.split(g)  # elements (0, 2) / (1, 3)
        e0, e2 = tl.split(tl.reshape(p02, (rows, lanes, 1, 2), can_reorder=False))
        e1, e3 = tl.split(tl.reshape(p13, (rows, lanes, 1, 2), can_reorder=False))
        e0 = tl.reshape(e0, (rows, lanes), can_reorder=False)
        e1 = tl.reshape(e1, (rows, lanes), can_reorder=False)
        e2 = tl.reshape(e2, (rows, lanes), can_reorder=False)
        e3 = tl.reshape(e3, (rows, lanes), can_reorder=False)
        # On the verified gfx936 BF16 ATen kernel, only the last two m2
        # accumulations in each vec4 round the product before adding it.
        mean, m2, cnt = _welford_push(e0, mean, m2, cnt, valid, MASKED)
        mean, m2, cnt = _welford_push(e1, mean, m2, cnt, valid, MASKED)
        mean, m2, cnt = _welford_push(e2, mean, m2, cnt, valid, MASKED, BF16_ATEN_M2)
        mean, m2, cnt = _welford_push(e3, mean, m2, cnt, valid, MASKED, BF16_ATEN_M2)
        return mean, m2, cnt



    @triton.jit
    def _fold_tree(mean, m2, cnt, rows: tl.constexpr, size: tl.constexpr):
        if size > 256:
            mean, m2, cnt = _fold_halves(mean, m2, cnt, rows, 256)
        if size > 128:
            mean, m2, cnt = _fold_halves(mean, m2, cnt, rows, 128)
        if size > 64:
            mean, m2, cnt = _fold_halves(mean, m2, cnt, rows, 64)
        if size > 32:
            mean, m2, cnt = _fold_halves(mean, m2, cnt, rows, 32)
        if size > 16:
            mean, m2, cnt = _fold_halves(mean, m2, cnt, rows, 16)
        if size > 8:
            mean, m2, cnt = _fold_halves(mean, m2, cnt, rows, 8)
        if size > 4:
            mean, m2, cnt = _fold_halves(mean, m2, cnt, rows, 4)
        if size > 2:
            mean, m2, cnt = _fold_halves(mean, m2, cnt, rows, 2)
        if size > 1:
            mean, m2, cnt = _fold_halves(mean, m2, cnt, rows, 1)
        return mean, m2, cnt



    @triton.jit(do_not_specialize=["seq_len", "n_rows"])
    def _layernorm_modulate_kernel(
        y_ptr, x_ptr, scale_ptr, shift_ptr, seq_len, n_rows, scale_row_stride, eps,
        D: tl.constexpr, ROWS: tl.constexpr,
        HAS_SHIFT: tl.constexpr = True,
        POST_OP: tl.constexpr = "bf16_modulate",
        BF16_ATEN_M2: tl.constexpr = False,
    ):

        tl.static_assert(
            POST_OP == "bf16_modulate" or POST_OP == "fp32_modulate"
            or POST_OP == "fp32_affine" or POST_OP == "bf16_norm_fp32_modulate",
            "unsupported POST_OP",
        )
        THREADS: tl.constexpr = 256
        WAVE: tl.constexpr = 64
        pid = tl.program_id(0).to(tl.int64)
        row_offs = pid * ROWS + tl.arange(0, ROWS)
        row_mask = row_offs < n_rows
        row_base = row_offs * D

        lanes = tl.arange(0, THREADS)
        mean = tl.zeros((ROWS, THREADS), dtype=tl.float32)
        m2 = tl.zeros((ROWS, THREADS), dtype=tl.float32)
        cnt = tl.zeros((ROWS, THREADS), dtype=tl.float32)

        # pass 1: per-"thread" serial Welford in the 256-thread HCU element order.
        # Out-of-range rows compute garbage that is never stored.
        for i in tl.static_range(((D + THREADS * 4 - 1) // (THREADS * 4))):
            cols = i * THREADS * 4 + lanes[:, None] * 4 + tl.arange(0, 4)[None, :]
            if (i + 1) * THREADS * 4 <= D:
                x4 = tl.load(
                    x_ptr + row_base[:, None, None] + cols[None, :, :],
                    mask=row_mask[:, None, None],
                    other=0.0,
                ).to(tl.float32)
                mean, m2, cnt = _push_vec4(
                    x4, mean, m2, cnt, row_mask, ROWS, THREADS, MASKED=False,
                    BF16_ATEN_M2=BF16_ATEN_M2,
                )
            else:
                # partial tail chunk: logical threads whose vector
                # index i*THREADS+t reaches N/4 skip the iteration, leaving their
                # Welford state untouched.
                vec_valid = (i * THREADS + lanes < D // 4)[None, :]
                x4 = tl.load(
                    x_ptr + row_base[:, None, None] + cols[None, :, :],
                    mask=row_mask[:, None, None] & vec_valid[:, :, None],
                    other=0.0,
                ).to(tl.float32)
                mean, m2, cnt = _push_vec4(
                    x4, mean, m2, cnt, vec_valid, ROWS, THREADS, MASKED=True,
                    BF16_ATEN_M2=BF16_ATEN_M2,
                )

        # warp fold trees, then the (0,2)/(1,3)/(0,1) inter-warp combines.
        mean = tl.reshape(mean, (ROWS * (THREADS // WAVE), WAVE), can_reorder=False)
        m2 = tl.reshape(m2, (ROWS * (THREADS // WAVE), WAVE), can_reorder=False)
        cnt = tl.reshape(cnt, (ROWS * (THREADS // WAVE), WAVE), can_reorder=False)
        mean, m2, cnt = _fold_tree(mean, m2, cnt, ROWS * (THREADS // WAVE), WAVE)
        mean = tl.reshape(mean, (ROWS, THREADS // WAVE), can_reorder=False)
        m2 = tl.reshape(m2, (ROWS, THREADS // WAVE), can_reorder=False)
        cnt = tl.reshape(cnt, (ROWS, THREADS // WAVE), can_reorder=False)
        mean, m2, cnt = _fold_tree(mean, m2, cnt, ROWS, THREADS // WAVE)

        denom = tl.zeros((ROWS, 1), dtype=tl.float32) + D
        rstd = rsqrt_f32(div_rn_f32(m2, denom) + eps)  # (ROWS, 1)

        batch = row_offs // seq_len

        # pass 2: normalize, then apply the selected post-op precision contract.
        for i in tl.static_range(((D + THREADS * 4 - 1) // (THREADS * 4))):
            cols = i * THREADS * 4 + tl.arange(0, THREADS * 4)
            mask = row_mask[:, None]
            if (i + 1) * THREADS * 4 > D:
                mask = mask & (cols < D)[None, :]
            x = tl.load(
                x_ptr + row_base[:, None] + cols[None, :],
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            y = rstd * (x - mean)
            if POST_OP == "bf16_modulate" or POST_OP == "bf16_norm_fp32_modulate":
                y = round_bf16_to_fp32(y)
            sc = tl.load(scale_ptr + batch[:, None] * scale_row_stride + cols[None, :], mask=mask, other=0.0).to(tl.float32)
            if POST_OP == "bf16_modulate":
                one_plus = round_bf16_to_fp32(1.0 + sc)
                y = round_bf16_to_fp32(y * one_plus)
                if HAS_SHIFT:
                    sh = tl.load(shift_ptr + batch[:, None] * scale_row_stride + cols[None, :], mask=mask, other=0.0).to(tl.float32)
                    y = y + sh
            else:
                sh = tl.load(shift_ptr + batch[:, None] * scale_row_stride + cols[None, :], mask=mask, other=0.0).to(tl.float32)
                if POST_OP == "fp32_affine":
                    y = tl.fma(y, sc, sh)
                elif POST_OP == "bf16_norm_fp32_modulate":
                    # Match the community modulation kernel's FP32 FMA.
                    y = tl.fma(y, 1.0 + sc, sh)
                else:
                    # Caller disables implicit FMA, as in the upstream kernel.
                    y = y * (1.0 + sc) + sh
            tl.store(y_ptr + row_base[:, None] + cols[None, :], y, mask=mask)


def _validate(x, weight, bias, scale, shift, norm_type):
    if norm_type not in ("layer", "rms"):
        raise ValueError("norm_type must be layer or rms")
    if x.ndim != 3 or x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("x must be a 3D fp16/bf16/fp32 tensor")
    b, s, d = x.shape
    if d <= 0 or d % 256 or d > 8192 or x.stride(-1) != 1:
        raise ValueError("hidden dimension must be contiguous, divisible by 256 and <= 8192")
    for parameter in (weight, bias):
        if parameter is not None and (
            parameter.shape != (d,) or parameter.device != x.device
            or parameter.dtype not in (torch.bfloat16, torch.float16, torch.float32)
            or parameter.stride(-1) != 1
        ):
            raise ValueError("weight/bias must be a contiguous hidden-dimension vector on x's device")
    for parameter in (scale, shift):
        shape = parameter.shape
        valid = (
            (parameter.ndim == 1 and shape[0] in (1, d))
            or (parameter.ndim == 2 and shape[0] in (1, b) and shape[1] == d)
            or (parameter.ndim == 3 and shape[0] in (1, b) and shape[1] in (1, s) and shape[2] == d)
            or (parameter.ndim == 4 and shape[0] == b and shape[1] > 0 and shape[2:] == (1, d) and s % shape[1] == 0)
        )
        if (not valid or parameter.device != x.device or parameter.stride(-1) != 1
                or parameter.dtype not in (torch.bfloat16, torch.float16, torch.float32)):
            raise ValueError("unsupported scale/shift shape, layout, dtype or device")


def _batch_parameter(parameter, batch, hidden):
    if parameter.ndim == 1 and parameter.numel() == hidden:
        return parameter.view(1, 1, hidden).expand(batch, 1, hidden)
    if parameter.ndim == 2:
        return parameter.unsqueeze(1).expand(batch, 1, hidden)
    if parameter.ndim == 3 and parameter.shape[1] == 1:
        return parameter.expand(batch, 1, hidden)
    return None


def _can_use_triton(x, weight, bias, scale, shift, norm_type):
    return (
        HAS_TRITON and x.is_cuda and torch.version.hip is not None and x.dtype == torch.bfloat16
        and norm_type == "layer" and weight is None and bias is None
        and x.shape[-1] in (4096, 5120) and x.is_contiguous()
        and scale is not None and shift is not None
        and scale.dtype == shift.dtype == torch.float32
        and scale.stride() == shift.stride()
        and torch.cuda.get_device_properties(x.device).gcnArchName.startswith("gfx936")
    )


def _apply_parameter(value, parameter, scale=False):
    if parameter.ndim == 4:
        batch, tokens, hidden = value.shape
        frames = parameter.shape[1]
        viewed = value.view(batch, frames, tokens // frames, hidden)
        result = viewed * (1 + parameter) if scale else viewed + parameter
        return result.flatten(1, 2)
    if parameter.ndim == 2:
        parameter = parameter.unsqueeze(1)
    return value * (1 + parameter) if scale else value + parameter


def fused_layernorm_modulate(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: Optional[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    """Launch the BF16 LayerNorm modulation used by SGLang's Triton custom op."""
    if (
        x.ndim != 3 or x.dtype != torch.bfloat16 or not x.is_contiguous()
        or x.numel() == 0 or x.shape[-1] % 4 or x.shape[-1] > 8192
    ):
        raise RuntimeError("LayerNorm modulation expects contiguous BF16 [B, S, D], D divisible by 4 and <= 8192")

    batch, tokens, hidden = x.shape
    for parameter in (scale,) if shift is None else (scale, shift):
        if (
            parameter.shape != (batch, hidden) or parameter.stride(1) != 1
            or parameter.dtype != torch.bfloat16 or parameter.device != x.device
        ):
            raise RuntimeError("modulation parameters must have packed BF16 [B, D] rows on x's device")
    if batch > 1 and shift is not None and shift.stride(0) != scale.stride(0):
        raise RuntimeError("shift must match scale's row stride")

    if (
        HAS_TRITON and x.is_cuda and torch.version.hip is not None
        and x.data_ptr() % 16 == 0
    ):
        output = torch.empty_like(x)
        rows = 2
        with torch.cuda.device(x.device):
            _layernorm_modulate_kernel[(triton.cdiv(batch * tokens, rows),)](
                output, x, scale, shift, tokens, batch * tokens,
                scale.stride(0) if batch > 1 else hidden, eps,
                D=hidden, ROWS=rows, HAS_SHIFT=shift is not None,
                POST_OP="bf16_modulate", BF16_ATEN_M2=True,
                num_warps=4, enable_fp_fusion=False,
            )
        return output

    normalized = F.layer_norm(x, (hidden,), eps=eps)
    modulated = normalized * (1 + scale.unsqueeze(1))
    return modulated if shift is None else modulated + shift.unsqueeze(1)


def fused_norm_scale_shift(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    scale: torch.Tensor,
    shift: torch.Tensor,
    norm_type: str,
    eps: float = 1e-5,
) -> torch.Tensor:
    _validate(x, weight, bias, scale, shift, norm_type)
    batch, tokens, hidden = x.shape
    batch_scale = _batch_parameter(scale, batch, hidden)
    batch_shift = _batch_parameter(shift, batch, hidden)
    if _can_use_triton(x, weight, bias, batch_scale, batch_shift, norm_type):
        output = torch.empty_like(x)
        if x.numel():
            _layernorm_modulate_kernel[(triton.cdiv(batch * tokens, 2),)](
                output, x, batch_scale, batch_shift, tokens, batch * tokens,
                batch_scale.stride(0), eps, D=hidden, ROWS=2,
                POST_OP="bf16_norm_fp32_modulate",
                num_warps=4, enable_fp_fusion=False,
            )
        return output
    value = x.float()
    if norm_type == "layer":
        affine = weight is not None and bias is not None
        normalized = F.layer_norm(
            value, (hidden,), weight.float() if affine else None,
            bias.float() if affine else None, eps,
        )
    else:
        normalized = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
        if weight is not None:
            normalized = normalized * weight.float()
    normalized = normalized.to(x.dtype)
    modulated = _apply_parameter(normalized, scale, scale=True)
    return _apply_parameter(modulated, shift).to(x.dtype)
