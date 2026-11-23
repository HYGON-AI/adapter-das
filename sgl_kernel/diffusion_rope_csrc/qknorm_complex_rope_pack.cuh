#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>

#include <algorithm>
#include <cstdint>

namespace sglang_hygon {

namespace host = sglang::host;

using BF16 = hip_bfloat16;

struct Rows {
  BF16* ptr;
  int64_t batch_stride;
  int64_t token_stride;
};

struct Params {
  Rows q, k, v, k_out, v_out, kp, vp;
  const BF16* qw;
  const BF16* kw;
  const float* rope;
  int64_t batch, seq, prefix, heads;
  float eps;
  bool copy_v;
};

__device__ __forceinline__ BF16* row_ptr(Rows rows, int64_t b, int64_t t, int h) {
  return rows.ptr + b * rows.batch_stride + t * rows.token_stride + h * 128;
}

__device__ __forceinline__ void norm_rope(
    const BF16* src, BF16* dst, const BF16* weight, const float* rope,
    float eps, int lane, int64_t row) {
  float value[4], normalized[4];
  for (int j = 0; j < 4; ++j) value[j] = float(src[lane * 4 + j]);
  float sum = __fmul_rn(value[0], value[0]);
  for (int j = 1; j < 4; ++j) sum = __fadd_rn(sum, __fmul_rn(value[j], value[j]));
  for (int off = 1; off < 32; off <<= 1) sum = __fadd_rn(sum, __shfl_down(sum, off, 32));
  sum = __shfl(sum, 0, 32);
  const float added = __fadd_rn(__fmul_rn(sum, 1.f / 128), eps);
  const float inv = static_cast<float>(::rsqrt(static_cast<double>(added)));
  for (int j = 0; j < 4; ++j) {
    float n = float(BF16(__fmul_rn(value[j], inv)));
    normalized[j] = float(BF16(__fmul_rn(n, float(weight[lane * 4 + j]))));
  }
  for (int j = 0; j < 2; ++j) {
    const int col = lane * 4 + j * 2;
    const float re = normalized[j * 2], im = normalized[j * 2 + 1];
    const float c = rope[col], s = rope[col + 1];
    const int64_t pair = row * 64 + col / 2;
    float r, i;
    if ((pair % 512) >= 384) {
      r = __fmaf_rn(-im, s, __fmul_rn(re, c));
      i = __fadd_rn(__fmul_rn(re, s), __fmul_rn(im, c));
    } else {
      r = __fmaf_rn(re, c, -__fmul_rn(im, s));
      i = __fmaf_rn(im, c, __fmul_rn(re, s));
    }
    dst[col] = BF16(r);
    dst[col + 1] = BF16(i);
  }
}

__device__ __forceinline__ void copy_row(const BF16* src, BF16* dst, int lane) {
  for (int j = 0; j < 4; ++j) dst[lane * 4 + j] = src[lane * 4 + j];
}

__global__ void pack_kernel(Params p) {
  const int lane = threadIdx.x & 31;
  const int64_t rows = p.batch * p.seq * p.heads;
  const int64_t prefix_rows = p.batch * p.prefix * p.heads;
  const int64_t total = 2 * rows + 2 * prefix_rows + (p.copy_v ? rows : 0);
  for (int64_t row = int64_t(blockIdx.x) * 8 + (threadIdx.x >> 5);
       row < total; row += int64_t(gridDim.x) * 8) {
    int64_t r = row;
    if (r < 2 * rows) {
      const bool is_q = r < rows;
      if (!is_q) r -= rows;
      const int h = r % p.heads;
      const int64_t t = r / p.heads % p.seq, b = r / (p.seq * p.heads);
      norm_rope(
          row_ptr(is_q ? p.q : p.k, b, t, h),
          row_ptr(is_q ? p.q : p.k_out, b, is_q ? t : p.prefix + t, h),
          is_q ? p.qw : p.kw, p.rope + t * 128, p.eps, lane, r);
      continue;
    }
    r -= 2 * rows;
    if (r < 2 * prefix_rows) {
      const bool is_k = r < prefix_rows;
      if (!is_k) r -= prefix_rows;
      const int h = r % p.heads;
      const int64_t t = r / p.heads % p.prefix, b = r / (p.prefix * p.heads);
      copy_row(row_ptr(is_k ? p.kp : p.vp, b, t, h),
               row_ptr(is_k ? p.k_out : p.v_out, b, t, h), lane);
      continue;
    }
    r -= 2 * prefix_rows;
    const int h = r % p.heads;
    const int64_t t = r / p.heads % p.seq, b = r / (p.seq * p.heads);
    copy_row(row_ptr(p.v, b, t, h), row_ptr(p.v_out, b, p.prefix + t, h), lane);
  }
}

inline Rows rows(tvm::ffi::TensorView t) {
  return {static_cast<BF16*>(t.data_ptr()), t.stride(0), t.stride(1)};
}

inline bool can_pack(tvm::ffi::TensorView q) {
  return (q.device().device_type == kDLCUDA || q.device().device_type == kDLROCM) &&
         q.dim() == 4 && q.size(3) == 128 &&
         q.dtype().code == kDLBfloat && q.dtype().bits == 16 && q.dtype().lanes == 1;
}

inline void run_pack(
    tvm::ffi::TensorView q, tvm::ffi::TensorView k, tvm::ffi::TensorView v,
    tvm::ffi::TensorView k_out, tvm::ffi::TensorView v_out,
    tvm::ffi::TensorView qw, tvm::ffi::TensorView kw, tvm::ffi::TensorView rope,
    tvm::ffi::TensorView kp, tvm::ffi::TensorView vp, double eps, bool copy_v) {
  using namespace sglang::host;
  auto B = SymbolicSize{"batch"}, S = SymbolicSize{"seq"}, H = SymbolicSize{"heads"};
  auto P = SymbolicSize{"prefix"}, PS = SymbolicSize{"prefix_plus_seq"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA, kDLROCM>();
  const auto verify_rows = [&](tvm::ffi::TensorView t, auto& tokens) {
    auto Bs = SymbolicSize{"batch_stride"}, Ts = SymbolicSize{"token_stride"};
    TensorMatcher({B, tokens, H, 128})
        .with_strides({Bs, Ts, 128, 1})
        .template with_dtype<sglang::bf16_t>()
        .with_device(device)
        .ensure_alignment(8)
        .verify(t);
  };
  verify_rows(q, S);
  verify_rows(k, S);
  verify_rows(k_out, PS);
  verify_rows(v_out, PS);
  if (copy_v) verify_rows(v, S);
  verify_rows(kp, P);
  verify_rows(vp, P);
  TensorMatcher({128}).with_dtype<sglang::bf16_t>().with_device(device).verify(qw).verify(kw);
  CHECK_HOST(rope.dim() == 2 && rope.size(0) == S.unwrap() && rope.size(1) == 64)
      << "RoPE shape must be [seq, 64]";
  CHECK_HOST(rope.dtype().code == kDLComplex && rope.dtype().bits == 64 && rope.dtype().lanes == 1)
      << "RoPE must be complex64";
  CHECK_HOST(rope.device().device_type == device.unwrap().device_type &&
             rope.device().device_id == device.unwrap().device_id) << "RoPE device mismatch";
  CHECK_HOST(rope.is_contiguous() && reinterpret_cast<uintptr_t>(rope.data_ptr()) % 16 == 0)
      << "RoPE must be contiguous and 16-byte aligned";
  CHECK_HOST(PS.unwrap() == P.unwrap() + S.unwrap()) << "outputs must hold prefix + seq tokens";
  if (B.unwrap() == 0 || S.unwrap() == 0 || H.unwrap() == 0) return;
  Params p{rows(q), rows(k), rows(v), rows(k_out), rows(v_out), rows(kp), rows(vp),
           static_cast<const BF16*>(qw.data_ptr()), static_cast<const BF16*>(kw.data_ptr()),
           static_cast<const float*>(rope.data_ptr()), q.size(0), q.size(1), kp.size(1), q.size(2),
           static_cast<float>(eps), copy_v};
  const int64_t total = p.batch * p.heads * (2 * p.seq + 2 * p.prefix + (copy_v ? p.seq : 0));
  const int blocks = static_cast<int>(std::min<int64_t>((total + 7) / 8, 65535));
  sglang::host::LaunchKernel(blocks, 256, q.device())(pack_kernel, p);
}

}  // namespace sglang_hygon
