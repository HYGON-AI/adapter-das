# SPDX-License-Identifier: Apache-2.0

import torch

from .compat import HAS_TRITON

if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_HEADS": 1, "BLOCK_HS_HALF": 32}, num_warps=2),
            triton.Config({"BLOCK_HEADS": 2, "BLOCK_HS_HALF": 32}, num_warps=2),
            triton.Config({"BLOCK_HEADS": 4, "BLOCK_HS_HALF": 32}, num_warps=4),
            triton.Config({"BLOCK_HEADS": 4, "BLOCK_HS_HALF": 64}, num_warps=4),
            triton.Config({"BLOCK_HEADS": 8, "BLOCK_HS_HALF": 64}, num_warps=8),
        ],
        key=["num_tokens", "num_heads", "head_size", "rotary_dim", "is_neox", "stride_x_row"],
        restore_value=["x_ptr"],
    )
    @triton.jit
    def _rotary_embedding_kernel(
        x_ptr, positions_ptr, cache_ptr,
        num_tokens, num_heads, head_size,
        rotary_dim: tl.constexpr, is_neox: tl.constexpr,
        stride_x_row, stride_positions, stride_cache_row, stride_cache_col,
        BLOCK_HEADS: tl.constexpr, BLOCK_HS_HALF: tl.constexpr,
    ):
        token_idx = tl.program_id(0).to(tl.int64)
        head_offsets = tl.program_id(1) * BLOCK_HEADS + tl.arange(0, BLOCK_HEADS)
        head_mask = head_offsets < num_heads
        position = tl.load(positions_ptr + token_idx * stride_positions).to(tl.int64)
        cache_row = cache_ptr + position * stride_cache_row
        x_rows = x_ptr + token_idx * stride_x_row + head_offsets[:, None] * head_size
        half = rotary_dim // 2

        for block_start in range(0, half, BLOCK_HS_HALF):
            offsets = block_start + tl.arange(0, BLOCK_HS_HALF)
            half_mask = offsets < half
            mask = head_mask[:, None] & half_mask[None, :]
            cos = tl.load(cache_row + offsets * stride_cache_col, half_mask, other=0.0)
            sin = tl.load(cache_row + (offsets + half) * stride_cache_col, half_mask, other=0.0)
            if is_neox:
                first, second = offsets, offsets + half
            else:
                first, second = 2 * offsets, 2 * offsets + 1
            x1 = tl.load(x_rows + first[None, :], mask, other=0.0).to(tl.float32)
            x2 = tl.load(x_rows + second[None, :], mask, other=0.0).to(tl.float32)
            y1 = tl.fma(-x2, sin[None, :], x1 * cos[None, :])
            y2 = tl.fma(x1, sin[None, :], x2 * cos[None, :])
            tl.store(x_rows + first[None, :], y1, mask)
            tl.store(x_rows + second[None, :], y2, mask)


def apply_rope_with_cos_sin_cache_inplace(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
) -> None:
    if positions.ndim != 1 or query.ndim != 2 or key.ndim != 2:
        raise ValueError("RoPE expects 1D positions and 2D query/key")
    if head_size <= 0 or head_size % 2:
        raise ValueError("head_size must be positive and even")
    if query.shape[1] % head_size or key.shape[1] % head_size:
        raise ValueError("head_size must divide query/key widths")
    if query.shape[0] != positions.numel() or key.shape[0] != positions.numel():
        raise ValueError("positions must have one entry per query/key row")
    if cos_sin_cache.ndim != 2:
        raise ValueError("cos_sin_cache must be a 2D [cos | sin] table")
    rotary_dim = cos_sin_cache.shape[1]
    if rotary_dim % 2 or rotary_dim > head_size:
        raise ValueError("cache width must be even and <= head_size")
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("RoPE query/key must be fp16, bf16 or fp32")
    if key.dtype != query.dtype or cos_sin_cache.dtype != torch.float32:
        raise ValueError("query/key must share dtype and cache must be fp32")
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("positions must be int32 or int64")
    if not query.is_cuda or any(
        tensor.device != query.device for tensor in (key, positions, cos_sin_cache)
    ):
        raise ValueError("RoPE tensors must be on the same GPU")
    if query.stride(-1) != 1 or key.stride(-1) != 1:
        raise ValueError("query/key must have a dense trailing dimension")
    if any(t.shape[0] > 1 and t.stride(0) < t.shape[1] for t in (query, key)):
        raise ValueError("query/key rows must not overlap")
    if positions.numel() == 0 or rotary_dim == 0:
        return
    if not HAS_TRITON:
        raise NotImplementedError("cached in-place RoPE requires Triton")

    for tensor in (query, key):
        if tensor.numel() == 0:
            continue
        heads = tensor.shape[1] // head_size
        _rotary_embedding_kernel[
            lambda meta: (positions.numel(), triton.cdiv(heads, meta["BLOCK_HEADS"]))
        ](
            tensor, positions, cos_sin_cache,
            positions.numel(), heads, head_size, rotary_dim, is_neox,
            tensor.stride(0), positions.stride(0),
            cos_sin_cache.stride(0), cos_sin_cache.stride(1),
        )
