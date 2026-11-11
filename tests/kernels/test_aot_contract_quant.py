# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""AOT contract anchors: sgl_per_token_group_quant_8bit bridge (C-class).

Contract sources (upstream sgl-kernel v0.5.22 tag):
  - python/sgl_kernel/gemm.py:28-64 — the bridge itself: destination-
    passing (output_q, output_s written in place), enable_v2=None
    auto-selects v2 for group_size in {16,32,64,128}, v1 asserts on
    fuse_silu_and_mul / masked_m, legacy aliases _fp8/_int8.
  - csrc/gemm/per_token_group_quant_8bit.cu:36-115 (v1 kernel):
    absmax floored at eps; y_s = absmax / max_8bit; q = clamp(x / y_s,
    min, max) cast to dst — the int8 cast is a C++ int8_t(float), i.e.
    TRUNCATION toward zero, not rounding. Row-major scales are written
    linearly per group; column-major is stride(0) < stride(1).
    scale_ue8m0: y_s = exp2(ceil(log2(max(y_s, 1e-10)))), stored as the
    biased exponent byte (log2(y_s)+127) packed 4-per-uint32
    (little-endian, column-major only).
  - csrc/gemm/per_token_group_quant_8bit_v2.cu (v2): eps is TORCH_CHECKed
    to 1e-10 (LOCAL_ABSMAX_ABS), min/max CHECK_EQ the dst dtype range,
    group_size switch over {16,32,64,128}; scale_inv = amax / dst_max,
    ue8m0 uses the DeepEP fast_log2_ceil bit trick (exp field + mantissa
    nonzero), exponent byte = float bits >> 23; quantized via MULTIPLY
    (x * scale_inv); fuse_silu_and_mul computes silu(primary) cast BACK
    to the input dtype before the product (amax and source both low
    precision), input (T, 2H) -> output (T, H); masked_m walks the DeepEP
    (E, maxT, H) layout processing rows [0, masked_m[e]).

Run (CPU face, any host):  pytest tests/kernels/test_aot_contract_quant.py
(fp8-dtype cases need torch CPU fp8 cast support and skip otherwise.)
"""

import math

import pytest
import torch

import sgl_kernel

has_fp8_cpu = hasattr(torch, "float8_e4m3fn")
requires_fp8 = pytest.mark.skipif(
    not has_fp8_cpu, reason="torch lacks float8_e4m3fn"
)


def _ref_v1(x_flat, group_size, eps, lo, hi):
    g = x_flat.reshape(-1, group_size).float()
    amax = g.abs().amax(dim=1).clamp_min(eps)
    y_s = amax / hi
    q = (g / y_s[:, None]).clamp_(lo, hi)
    return y_s, q


# ------------------------------------------------------------- bridge face --

def test_bridge_signature_matches_upstream():
    import inspect

    params = list(inspect.signature(
        sgl_kernel.sgl_per_token_group_quant_8bit).parameters)
    assert params == [
        "input", "output_q", "output_s", "group_size", "eps",
        "fp8_min", "fp8_max", "scale_ue8m0", "fuse_silu_and_mul",
        "masked_m", "enable_v2",
    ]
    # legacy aliases (upstream gemm.py:67-68)
    assert sgl_kernel.sgl_per_token_group_quant_fp8 is sgl_kernel.sgl_per_token_group_quant_8bit
    assert sgl_kernel.sgl_per_token_group_quant_int8 is sgl_kernel.sgl_per_token_group_quant_8bit


def test_bridge_v1_rejects_fuse_and_masked():
    x = torch.zeros(4, 64)
    q = torch.zeros(4, 64, dtype=torch.int8)
    s = torch.zeros(4, 4)
    with pytest.raises(AssertionError, match="fuse_silu_and_mul"):
        sgl_kernel.sgl_per_token_group_quant_8bit(
            x, q, s, 64, 1e-10, -128.0, 127.0,
            fuse_silu_and_mul=True, enable_v2=False)
    with pytest.raises(AssertionError, match="masked_m"):
        sgl_kernel.sgl_per_token_group_quant_8bit(
            x, q, s, 64, 1e-10, -128.0, 127.0,
            masked_m=torch.tensor([1]), enable_v2=False)


def test_bridge_v2_enforces_eps_and_range():
    x = torch.randn(4, 128)
    q = torch.zeros(4, 128, dtype=torch.int8)
    s = torch.zeros(4, 2)
    with pytest.raises(RuntimeError, match="eps"):
        sgl_kernel.sgl_per_token_group_quant_8bit(
            x, q, s, 64, 1e-6, -128.0, 127.0)  # v2 auto (64 in whitelist)
    with pytest.raises(RuntimeError, match="match the output"):
        sgl_kernel.sgl_per_token_group_quant_8bit(
            x, q, s, 64, 1e-10, -448.0, 448.0)  # int8 out, fp8 range


def test_bridge_v2_bad_group_size():
    x = torch.randn(4, 128)
    q = torch.zeros(4, 128, dtype=torch.int8)
    s = torch.zeros(4, 256 // 96)
    with pytest.raises(RuntimeError, match="group_size"):
        sgl_kernel.sgl_per_token_group_quant_8bit(
            x, q, torch.zeros(4, 2), 96, 1e-10, -128.0, 127.0,
            enable_v2=True)


# ------------------------------------------------------------------ v1 math --

@pytest.mark.parametrize("dtype_lo,dtype_hi,dst", [
    (-128.0, 127.0, torch.int8),
])
def test_quant_v1_int8_math(dtype_lo, dtype_hi, dst):
    torch.manual_seed(0)
    gs = 64
    x = (torch.randn(6, 128) * 5)
    y_s_ref, q_ref = _ref_v1(x.reshape(-1), gs, 1e-10, dtype_lo, dtype_hi)
    q = torch.zeros(6, 128, dtype=dst)
    s = torch.zeros(6, 2)  # row-major (6, groups_per_row)
    sgl_kernel.sgl_per_token_group_quant_8bit(
        x, q, s, gs, 1e-10, dtype_lo, dtype_hi, enable_v2=False)
    torch.testing.assert_close(s, y_s_ref.reshape(6, 2), rtol=1e-6, atol=1e-8)
    # int8 cast is truncation (C++ int8_t(float)), mirrored by torch .to
    torch.testing.assert_close(q.float(), q_ref.to(torch.int8).reshape(6, 128).float(),
                               rtol=0, atol=0)


def test_quant_v1_eps_floor():
    # A zero group must produce scale eps/max, not 0 (v1: local_absmax=eps).
    x = torch.zeros(2, 64)
    q = torch.zeros(2, 64, dtype=torch.int8)
    s = torch.zeros(2, 1)
    sgl_kernel.sgl_per_token_group_quant_8bit(
        x, q, s, 64, 1e-10, -128.0, 127.0, enable_v2=False)
    assert s[0, 0].item() == pytest.approx(1e-10 / 127.0, rel=1e-6)


def test_quant_v1_int8_truncation_semantics():
    # x = 0.5 * scale -> x/y_s = 0.5 -> int8 cast TRUNCATES to 0 (a
    # round()-based quantizer would emit 1 or -1 here).
    gs = 64
    x = torch.zeros(1, gs)
    x[0, 0] = 0.5   # amax 0.5 -> y_s = 0.5/127; this lane = 127 exactly
    x[0, 1] = -0.5
    # craft: value 0.25 -> 0.25 / (0.5/127) = 63.5 -> truncates to 63
    x[0, 2] = 0.25
    q = torch.zeros(1, gs, dtype=torch.int8)
    s = torch.zeros(1, 1)
    sgl_kernel.sgl_per_token_group_quant_8bit(
        x, q, s, gs, 1e-10, -128.0, 127.0, enable_v2=False)
    assert q[0, 0].item() == 127
    assert q[0, 1].item() == -127
    assert q[0, 2].item() == 63  # 63.5 truncated, not rounded to 64


def test_quant_v1_column_major_scales():
    # stride(0) < stride(1) -> column-major: scale[t, g] indexed per group.
    torch.manual_seed(1)
    gs = 64
    x = torch.randn(3, 128)
    y_s_ref, _ = _ref_v1(x.reshape(-1), gs, 1e-10, -128.0, 127.0)
    q = torch.zeros(3, 128, dtype=torch.int8)
    s = torch.zeros(3, 2).t()  # shape (2, 3), stride (1, 2): column-major
    sgl_kernel.sgl_per_token_group_quant_8bit(
        x, q, s, gs, 1e-10, -128.0, 127.0, enable_v2=False)
    torch.testing.assert_close(s, y_s_ref.reshape(3, 2).T, rtol=1e-6, atol=1e-8)


# ------------------------------------------------------------------ v2 math --

def test_quant_v2_int8_math_and_floor():
    torch.manual_seed(2)
    gs = 64
    x = torch.randn(5, 128) * 3
    q = torch.zeros(5, 128, dtype=torch.int8)
    s = torch.zeros(5, 2)
    sgl_kernel.sgl_per_token_group_quant_8bit(
        x, q, s, gs, 1e-10, -128.0, 127.0)  # v2 auto (64 whitelisted)
    g = x.reshape(5, 2, gs).float()
    amax = g.abs().amax(dim=-1).clamp_min(1e-10)
    scale_inv = amax / 127.0
    q_ref = (g * scale_inv[..., None]).clamp_(-128.0, 127.0)
    torch.testing.assert_close(s, scale_inv, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(q.float(), q_ref.reshape(5, 128).to(torch.int8).float(),
                               rtol=0, atol=0)
    # zero group -> scale floor 1e-10/127
    x0 = torch.zeros(1, gs)
    q0 = torch.zeros(1, gs, dtype=torch.int8)
    s0 = torch.zeros(1, 1)
    sgl_kernel.sgl_per_token_group_quant_8bit(x0, q0, s0, gs, 1e-10, -128.0, 127.0)
    assert s0[0, 0].item() == pytest.approx(1e-10 / 127.0, rel=1e-6)


def test_quant_v2_ue8m0_packs_biased_exponents():
    # amax chosen so amax/127 lands strictly between powers of two: the
    # ue8m0 scale must round UP to the next power (ceil-log2) and pack as
    # biased exponent bytes, 4 per uint32, column-major.
    gs = 32
    hidden = 256  # 8 groups per row -> two uint32 packs per row
    amax = 100.0  # ratio = 100/127 ~ 0.787 -> ceil-log2 -> 2^0 = 1
    x = torch.full((2, hidden), amax)
    x[1, 0] = -amax
    q = torch.zeros(2, hidden, dtype=torch.int8)
    # column-major uint32 matrix (npacks=2, rows=2): stride (1, 2)
    s = torch.zeros(2, 2, dtype=torch.int32).t()
    sgl_kernel.sgl_per_token_group_quant_8bit(
        x, q, s, gs, 1e-10, -128.0, 127.0, scale_ue8m0=True)
    expect_scale = 1.0  # 2^ceil(log2(0.787))
    expect_byte = int(math.log2(expect_scale)) + 127  # 127
    packed = expect_byte * 0x01010101
    assert s[0, 0].item() == packed
    assert s[1, 1].item() == packed
    # quantized: amax * 1.0 = 100 -> int8 (no clamp needed)
    assert q[0, 0].item() == 100


def test_quant_v2_ue8m0_nonsquare_column_major():
    # rows(4) != npacks(2): the packed (rows, npacks) matrix must be
    # TRANSPOSED into the column-major (npacks, rows) output_s — a plain
    # reshape permutes the slots and silently lands every pack in the
    # wrong (pack, token) slot (review #80; the square buffer in the test
    # above masked this).
    gs = 32
    hidden = 256  # 8 groups per row -> two uint32 packs
    rows = 4      # rows != npacks: reshape would permute the slots
    # per-row amax picked on different sides of a power of two:
    # amax/127 <= 1 -> scale 1.0 (byte 127); amax/127 > 1 -> scale 2.0 (byte 128)
    amax_small, amax_big = 100.0, 200.0
    x = torch.empty(rows, hidden)
    for r in range(rows):
        x[r] = amax_big if r % 2 == 1 else amax_small
    q = torch.zeros(rows, hidden, dtype=torch.int8)
    # column-major uint32 matrix (npacks=2, rows=4): stride (1, 2)
    s = torch.zeros(rows, 2, dtype=torch.int32).t()
    sgl_kernel.sgl_per_token_group_quant_8bit(
        x, q, s, gs, 1e-10, -128.0, 127.0, scale_ue8m0=True)
    byte_small = int(math.log2(1.0)) + 127   # amax 100 -> scale 1.0
    byte_big = int(math.log2(2.0)) + 127     # amax 200 -> ratio 1.575 -> 2.0
    for r in range(rows):
        expect = byte_small * 0x01010101 if r % 2 == 0 else byte_big * 0x01010101
        for p in range(2):
            # int32 holds the uint32 bit pattern: byte 128 sets the sign
            # bit, so compare as unsigned bit patterns, not signed values.
            got = s[p, r].item() & 0xFFFFFFFF
            assert got == expect, f"pack {p}, row {r}: {got:#x} != {expect:#x}"


def test_quant_v2_fuse_silu_and_mul_low_precision():
    # v2: silu(primary) is cast back to the INPUT dtype before the
    # product; amax and quantized source both come from that product.
    torch.manual_seed(3)
    gs = 64
    dtype = torch.float16
    x = (torch.randn(4, 128) * 3).to(dtype)  # (T, 2H), H=64
    q = torch.zeros(4, 64, dtype=torch.int8)
    s = torch.zeros(4, 1)
    sgl_kernel.sgl_per_token_group_quant_8bit(
        x, q, s, gs, 1e-10, -128.0, 127.0, fuse_silu_and_mul=True)
    primary = torch.nn.functional.silu(x[..., :64].float()).to(dtype)
    val = (primary * x[..., 64:]).float()
    amax = val.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    scale_inv = amax / 127.0
    q_ref = (val * scale_inv).clamp_(-128.0, 127.0)
    torch.testing.assert_close(s, scale_inv, rtol=1e-3, atol=1e-6)
    torch.testing.assert_close(q.float(), q_ref.to(torch.int8).float(),
                               rtol=0, atol=0)


def test_quant_v2_masked_m_deepep_layout():
    # DeepEP layout: input (E, maxT, H); rows beyond masked_m[e] untouched.
    torch.manual_seed(4)
    gs = 64
    E, maxT, H = 3, 6, 64
    x = torch.randn(E, maxT, H) * 3
    masked = torch.tensor([4, 0, 6], dtype=torch.int32)
    q = torch.full((E, maxT, H), 99, dtype=torch.int8)
    s = torch.zeros(E, maxT, 1)
    sgl_kernel.sgl_per_token_group_quant_8bit(
        x, q, s, gs, 1e-10, -128.0, 127.0, masked_m=masked)
    for e, m in enumerate([4, 0, 6]):
        if m == 0:
            assert (q[e] == 99).all()  # untouched
            continue
        val = x[e, :m]
        amax = val.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
        q_ref = (val * (amax / 127.0)).clamp_(-128.0, 127.0).to(torch.int8)
        torch.testing.assert_close(q[e, :m].float(), q_ref.float(), rtol=0, atol=0)
        if m < maxT:
            assert (q[e, m:] == 99).all()  # padding rows untouched


@requires_fp8
def test_quant_v1_fp8_math():
    torch.manual_seed(5)
    gs = 128
    x = torch.randn(3, 256)
    y_s_ref, q_ref = _ref_v1(x.reshape(-1), gs, 1e-10, -448.0, 448.0)
    q = torch.zeros(3, 256, dtype=torch.float8_e4m3fn)
    s = torch.zeros(3, 2)
    sgl_kernel.sgl_per_token_group_quant_8bit(
        x, q, s, gs, 1e-10, -448.0, 448.0, enable_v2=False)
    torch.testing.assert_close(s, y_s_ref.reshape(3, 2), rtol=1e-6, atol=1e-8)
    # fp8 cast is round-to-nearest (both __nv_cvt and torch CPU)
    torch.testing.assert_close(
        q.float(), q_ref.to(torch.float8_e4m3fn).reshape(3, 256).float(),
        rtol=0, atol=0)
    # dequant sanity: fp8 e4m3 relative precision is 2^-3 per step, so the
    # round-trip error is bounded by ~6.25% relative (half ULP, RN cast)
    deq = (q.float().reshape(6, gs) * y_s_ref[:, None]).reshape(3, 256)
    rel_err = (deq - x).abs().max() / x.abs().max()
    assert rel_err < 0.07


@requires_fp8
def test_quant_v2_fp8_auto_selected():
    # group_size 128 is whitelisted -> v2 without enable_v2=True.
    torch.manual_seed(6)
    gs = 128
    x = torch.randn(2, 256) * 4
    q = torch.zeros(2, 256, dtype=torch.float8_e4m3fn)
    s = torch.zeros(2, 2)
    sgl_kernel.sgl_per_token_group_quant_8bit(x, q, s, gs, 1e-10, -448.0, 448.0)
    g = x.reshape(2, 2, gs).float()
    amax = g.abs().amax(dim=-1).clamp_min(1e-10)
    scale_inv = amax / 448.0
    q_ref = (g * scale_inv[..., None]).clamp_(-448.0, 448.0)
    torch.testing.assert_close(s, scale_inv, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(
        q.float(), q_ref.reshape(2, 256).to(torch.float8_e4m3fn).float(),
        rtol=0, atol=0)
