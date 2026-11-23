# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""HIP ports of the Qwen-Image 2.1 diffusion RoPE kernels."""

from functools import lru_cache
from pathlib import Path

import torch


_SOURCE_DIR = Path(__file__).resolve().parent / "diffusion_rope_csrc"
_CFLAGS = ["-ffp-contract=off", "-fno-slp-vectorize"]


@lru_cache
def _norm_rope_module():
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "hcu_diffusion_qknorm_complex_rope",
        cuda_files=[str(_SOURCE_DIR / "qknorm_complex_rope.cuh")],
        cuda_wrappers=[
            ("run_norm_rope", "sglang_hygon::run_norm_rope"),
            ("run_norm_rope_kv", "sglang_hygon::run_norm_rope_kv"),
            ("run_pack", "sglang_hygon::run_pack"),
        ],
        extra_cuda_cflags=_CFLAGS,
    )


@lru_cache
def _rope_module():
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "hcu_diffusion_complex_rope",
        cuda_files=[str(_SOURCE_DIR / "complex_rope.cuh")],
        cuda_wrappers=[("run_complex_rope", "sglang_hygon::run_complex_rope")],
        extra_cuda_cflags=_CFLAGS,
    )


def _rope_reference(x, rope):
    pairs = x.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    complex_x = torch.view_as_complex(pairs.contiguous())
    return torch.view_as_real(complex_x * rope[None, :, None]).flatten(-2).to(x.dtype)


def _norm_rope_reference(x, weight, rope, eps):
    x_fp32 = x.contiguous().float()
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    normalized = weight * (x_fp32 * torch.rsqrt(variance + eps)).to(x.dtype)
    return _rope_reference(normalized, rope)


def _can_rope(x, rope):
    return (
        torch.version.hip is not None
        and x.is_cuda
        and x.ndim == 4
        and x.numel() > 0
        and x.shape[-1] % 2 == 0
        and rope.device == x.device
        and rope.dtype == torch.complex64
        and rope.shape == (x.shape[1], x.shape[-1] // 2)
        and rope.is_contiguous()
        and rope.data_ptr() % 16 == 0
    )


def _packed_bf16_rows(x, device):
    return (
        x.device == device
        and x.dtype == torch.bfloat16
        and x.ndim == 4
        and x.shape[-1] == 128
        and x.stride()[-2:] == (128, 1)
        and x.data_ptr() % 8 == 0
    )


def _can_norm_rope(x, weight, rope):
    return (
        _can_rope(x, rope)
        and _packed_bf16_rows(x, x.device)
        and weight.shape == (128,)
        and weight.device == x.device
        and weight.dtype == x.dtype
        and weight.is_contiguous()
    )


def fused_complex_rope(x, rope):
    if (
        _can_rope(x, rope)
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.shape[2] > 1
        and x.numel() % 1024 == 0
        and x.is_contiguous()
    ):
        out = torch.empty_like(x)
        with torch.cuda.device(x.device):
            _rope_module().run_complex_rope(x, rope, out)
        return out
    return _rope_reference(x, rope)


def qknorm_complex_rope(x, weight, rope, eps):
    if _can_norm_rope(x, weight, rope):
        out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        with torch.cuda.device(x.device):
            _norm_rope_module().run_norm_rope(x, out, weight, rope, float(eps))
        return out
    return _norm_rope_reference(x, weight, rope, eps)


def qknorm_complex_rope_kv(k, weight, rope, v, k_prefix, v_prefix, eps):
    if (
        _can_norm_rope(k, weight, rope)
        and all(_packed_bf16_rows(t, k.device) for t in (v, k_prefix, v_prefix))
        and v.shape == k.shape
        and k_prefix.shape == v_prefix.shape
        and k_prefix.shape[0] == k.shape[0]
        and k_prefix.shape[2:] == k.shape[2:]
    ):
        shape = (k.shape[0], k_prefix.shape[1] + k.shape[1], *k.shape[2:])
        k_out, v_out = k.new_empty(shape), v.new_empty(shape)
        with torch.cuda.device(k.device):
            _norm_rope_module().run_norm_rope_kv(
                k, v, k_prefix, v_prefix, k_out, v_out, weight, rope, float(eps)
            )
        return k_out, v_out
    return (
        torch.cat((k_prefix, _norm_rope_reference(k, weight, rope, eps)), dim=1),
        torch.cat((v_prefix, v), dim=1),
    )


def qknorm_complex_rope_pack_(
    q, k_out, v_out, q_weight, k_weight, rope, k_prefix, v_prefix, k_src, v_src, eps
):
    prefix = k_prefix.shape[1]
    k = k_out[:, prefix:] if k_src is None else k_src
    v = v_out[:, prefix:] if v_src is None else v_src
    copy_v = v_src is not None
    if (
        _can_norm_rope(q, q_weight, rope)
        and _can_norm_rope(k, k_weight, rope)
        and all(_packed_bf16_rows(t, q.device) for t in (k, v, k_out, v_out, k_prefix, v_prefix))
    ):
        with torch.cuda.device(q.device):
            _norm_rope_module().run_pack(
                q, k, v, k_out, v_out, q_weight, k_weight, rope,
                k_prefix, v_prefix, float(eps), copy_v,
            )
        return

    q_rotated = _norm_rope_reference(q, q_weight, rope, eps)
    k_rotated = _norm_rope_reference(k, k_weight, rope, eps)
    q.copy_(q_rotated)
    k_out[:, :prefix].copy_(k_prefix)
    v_out[:, :prefix].copy_(v_prefix)
    k_out[:, prefix:].copy_(k_rotated)
    if copy_v:
        v_out[:, prefix:].copy_(v)
