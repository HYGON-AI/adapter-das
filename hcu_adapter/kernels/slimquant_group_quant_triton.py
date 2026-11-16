# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Fused implementation of the existing SlimQuant group64 activation quantizer."""
import torch
import triton
import triton.language as tl


@triton.jit
def _group_quant_kernel(X, Q, S, M: tl.constexpr, K: tl.constexpr,
                        SM: tl.constexpr, SK: tl.constexpr,
                        BG: tl.constexpr):
    groups: tl.constexpr = K // 64
    g = tl.program_id(0) * BG + tl.arange(0, BG)
    row = g // groups
    col = (g % groups)[:, None]*64 + tl.arange(0, 64)[None, :]
    x = tl.load(X + row[:, None].to(tl.int64)*SM + col*SK,
                mask=row[:, None] < M, other=0).to(tl.float32)
    maximum = tl.max(tl.abs(x), 1)
    # Torch scalar division uses a rounded FP32 reciprocal multiplication.
    # div_rn(maximum, 127) changes scales and INT8 codes near ties.
    scale = maximum * (1.0 / 127.0)
    scale = tl.maximum(scale, 1.0e-10)
    value = tl.div_rn(x, scale[:, None])
    lower = tl.floor(value)
    fraction = value-lower
    increment = (fraction > 0.5) | ((fraction == 0.5) & ((lower.to(tl.int32) & 1) != 0))
    rounded = lower + increment.to(tl.float32)
    rounded = tl.minimum(tl.maximum(rounded, -127.0), 127.0)
    tl.store(Q + row[:, None].to(tl.int64)*K + col, rounded.to(tl.int8),
             mask=row[:, None] < M)
    tl.store(S + g, scale, mask=row < M)


def quantize(x):
    m, k = x.shape
    assert k > 0 and k % 64 == 0
    q = torch.empty((m, k), device=x.device, dtype=torch.int8)
    scale = torch.empty((m, k//64), device=x.device, dtype=torch.float32)
    if m:
        _group_quant_kernel[(triton.cdiv(m*(k//64), 16),)](
            x, q, scale, m, k, x.stride(0), x.stride(1), 16,
            num_warps=4, enable_fp_fusion=False)
    return q, scale
