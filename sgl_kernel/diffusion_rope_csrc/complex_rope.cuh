#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>

#include <algorithm>
#include <cstdint>

namespace sglang_hygon {

namespace host = sglang::host;

template <typename T>
__global__ void complex_rope_kernel(
    const T* x, const float* rope, T* out, int64_t pairs, int64_t seq,
    int64_t heads, int64_t dim) {
  for (int64_t pair = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       pair < pairs; pair += int64_t(gridDim.x) * blockDim.x) {
    const int64_t token = pair / (heads * (dim / 2)) % seq;
    const int64_t col = pair % (dim / 2);
    const float re = float(x[2 * pair]), im = float(x[2 * pair + 1]);
    const float c = rope[token * dim + 2 * col];
    const float s = rope[token * dim + 2 * col + 1];
    float r, i;
    if ((pair % 512) >= 384) {
      r = __fmaf_rn(-im, s, __fmul_rn(re, c));
      i = __fadd_rn(__fmul_rn(re, s), __fmul_rn(im, c));
    } else {
      r = __fmaf_rn(re, c, -__fmul_rn(im, s));
      i = __fmaf_rn(im, c, __fmul_rn(re, s));
    }
    out[2 * pair] = T(r);
    out[2 * pair + 1] = T(i);
  }
}

template <typename T>
inline void launch_complex_rope(
    tvm::ffi::TensorView x, tvm::ffi::TensorView rope, tvm::ffi::TensorView out) {
  const int64_t pairs = x.numel() / 2;
  const int blocks = static_cast<int>(std::min<int64_t>((pairs + 255) / 256, 65535));
  sglang::host::LaunchKernel(blocks, 256, x.device())(
      complex_rope_kernel<T>, static_cast<const T*>(x.data_ptr()),
      static_cast<const float*>(rope.data_ptr()), static_cast<T*>(out.data_ptr()),
      pairs, x.size(1), x.size(2), x.size(3));
}

inline void run_complex_rope(
    tvm::ffi::TensorView x, tvm::ffi::TensorView rope, tvm::ffi::TensorView out) {
  using namespace sglang::host;
  auto B = SymbolicSize{"batch"}, S = SymbolicSize{"seq"};
  auto H = SymbolicSize{"heads"}, D = SymbolicSize{"head_dim"};
  auto device = SymbolicDevice{};
  auto dtype = SymbolicDType{};
  TensorMatcher({B, S, H, D})
      .with_dtype<sglang::bf16_t, sglang::fp16_t, float>(dtype)
      .with_device<kDLCUDA, kDLROCM>(device)
      .verify(x)
      .verify(out);
  CHECK_HOST(x.is_contiguous() && out.is_contiguous()) << "RoPE activations must be contiguous";
  CHECK_HOST(H.unwrap() > 1 && D.unwrap() > 0 && D.unwrap() % 2 == 0 &&
             x.numel() > 0 && x.numel() % 1024 == 0)
      << "HCU RoPE requires head broadcasting and complete 512-pair ATen blocks";
  CHECK_HOST(rope.dim() == 2 && rope.size(0) == S.unwrap() && rope.size(1) == D.unwrap() / 2)
      << "RoPE shape must be [seq, head_dim / 2]";
  CHECK_HOST(rope.dtype().code == kDLComplex && rope.dtype().bits == 64 && rope.dtype().lanes == 1)
      << "RoPE must be complex64";
  CHECK_HOST(rope.device().device_type == device.unwrap().device_type &&
             rope.device().device_id == device.unwrap().device_id && rope.is_contiguous())
      << "RoPE must be contiguous and on the activation device";
  if (dtype.is_type<sglang::bf16_t>()) {
    launch_complex_rope<hip_bfloat16>(x, rope, out);
  } else if (dtype.is_type<sglang::fp16_t>()) {
    launch_complex_rope<__half>(x, rope, out);
  } else {
    launch_complex_rope<float>(x, rope, out);
  }
}

}  // namespace sglang_hygon
