#pragma once

#include "qknorm_complex_rope_pack.cuh"

namespace sglang_hygon {

struct NormRopeParams {
  Rows x, out, v, v_out, kp, vp;
  const BF16* weight;
  const float* rope;
  int64_t batch, seq, prefix, heads;
  float eps;
};

template <bool PackKV>
__global__ void norm_rope_kernel(NormRopeParams p) {
  const int lane = threadIdx.x & 31;
  const int64_t rows = p.batch * p.seq * p.heads;
  const int64_t prefix_rows = p.batch * p.prefix * p.heads;
  const int64_t total = rows + (PackKV ? 2 * prefix_rows + rows : 0);
  for (int64_t row = int64_t(blockIdx.x) * 8 + (threadIdx.x >> 5);
       row < total; row += int64_t(gridDim.x) * 8) {
    int64_t r = row;
    if (r < rows) {
      const int h = r % p.heads;
      const int64_t t = r / p.heads % p.seq, b = r / (p.seq * p.heads);
      norm_rope(row_ptr(p.x, b, t, h), row_ptr(p.out, b, p.prefix + t, h),
                p.weight, p.rope + t * 128, p.eps, lane, r);
      continue;
    }
    if constexpr (PackKV) {
      r -= rows;
      if (r < 2 * prefix_rows) {
        const bool is_k = r < prefix_rows;
        if (!is_k) r -= prefix_rows;
        const int h = r % p.heads;
        const int64_t t = r / p.heads % p.prefix, b = r / (p.prefix * p.heads);
        copy_row(row_ptr(is_k ? p.kp : p.vp, b, t, h),
                 row_ptr(is_k ? p.out : p.v_out, b, t, h), lane);
        continue;
      }
      r -= 2 * prefix_rows;
      const int h = r % p.heads;
      const int64_t t = r / p.heads % p.seq, b = r / (p.seq * p.heads);
      copy_row(row_ptr(p.v, b, t, h), row_ptr(p.v_out, b, p.prefix + t, h), lane);
    }
  }
}

inline void run_norm_rope_impl(
    tvm::ffi::TensorView x, tvm::ffi::TensorView out,
    tvm::ffi::TensorView weight, tvm::ffi::TensorView rope, double eps,
    tvm::ffi::TensorView v, tvm::ffi::TensorView v_out,
    tvm::ffi::TensorView kp, tvm::ffi::TensorView vp, bool pack_kv) {
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
  verify_rows(x, S);
  if (pack_kv) {
    verify_rows(out, PS);
    verify_rows(v, S);
    verify_rows(v_out, PS);
    verify_rows(kp, P);
    verify_rows(vp, P);
    CHECK_HOST(PS.unwrap() == P.unwrap() + S.unwrap()) << "outputs must hold prefix + seq tokens";
  } else {
    verify_rows(out, S);
  }
  TensorMatcher({128}).with_dtype<sglang::bf16_t>().with_device(device).verify(weight);
  CHECK_HOST(rope.dim() == 2 && rope.size(0) == S.unwrap() && rope.size(1) == 64)
      << "RoPE shape must be [seq, 64]";
  CHECK_HOST(rope.dtype().code == kDLComplex && rope.dtype().bits == 64 && rope.dtype().lanes == 1)
      << "RoPE must be complex64";
  CHECK_HOST(rope.device().device_type == device.unwrap().device_type &&
             rope.device().device_id == device.unwrap().device_id) << "RoPE device mismatch";
  CHECK_HOST(rope.is_contiguous() && reinterpret_cast<uintptr_t>(rope.data_ptr()) % 16 == 0)
      << "RoPE must be contiguous and 16-byte aligned";
  if (B.unwrap() == 0 || S.unwrap() == 0 || H.unwrap() == 0) return;
  NormRopeParams p{};
  p.x = rows(x);
  p.out = rows(out);
  p.weight = static_cast<const BF16*>(weight.data_ptr());
  p.rope = static_cast<const float*>(rope.data_ptr());
  p.batch = x.size(0);
  p.seq = x.size(1);
  p.heads = x.size(2);
  p.eps = static_cast<float>(eps);
  if (pack_kv) {
    p.v = rows(v);
    p.v_out = rows(v_out);
    p.kp = rows(kp);
    p.vp = rows(vp);
    p.prefix = kp.size(1);
  }
  const int64_t total = p.batch * p.heads * (p.seq + (pack_kv ? 2 * p.prefix + p.seq : 0));
  const int blocks = static_cast<int>(std::min<int64_t>((total + 7) / 8, 65535));
  if (pack_kv) {
    LaunchKernel(blocks, 256, x.device())(norm_rope_kernel<true>, p);
  } else {
    LaunchKernel(blocks, 256, x.device())(norm_rope_kernel<false>, p);
  }
}

inline void run_norm_rope(tvm::ffi::TensorView x, tvm::ffi::TensorView out,
                          tvm::ffi::TensorView weight, tvm::ffi::TensorView rope, double eps) {
  run_norm_rope_impl(x, out, weight, rope, eps, x, out, x, x, false);
}

inline void run_norm_rope_kv(
    tvm::ffi::TensorView k, tvm::ffi::TensorView v,
    tvm::ffi::TensorView kp, tvm::ffi::TensorView vp,
    tvm::ffi::TensorView k_out, tvm::ffi::TensorView v_out,
    tvm::ffi::TensorView weight, tvm::ffi::TensorView rope, double eps) {
  run_norm_rope_impl(k, k_out, weight, rope, eps, v, v_out, kp, vp, true);
}

}  // namespace sglang_hygon
