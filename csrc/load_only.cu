#include <ATen/ceil_div.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>

#include "common.cuh"
#include "w4a16.h"

namespace int4gemv {
namespace {

// Computes no GEMV: it issues the vectorized kernel's loads at M = 1 and xors the loaded bits into
// out[row0], so the compiler can't drop them. If this is no faster than the vectorized kernel,
// the unpacking costs nothing and memory is the limit.
template <int kRowsPerWarp>
__global__ void load_only_kernel(uint32_t* __restrict__ out, const half* __restrict__ x,
                                 const uint4* __restrict__ qweight, const half* __restrict__ scales,
                                 const half* __restrict__ zeros, int n, int k, int group_size) {
  const int lane = threadIdx.x % kWarpSize;
  const int warps_per_block = blockDim.x / kWarpSize;
  const int row0 = (blockIdx.x * warps_per_block + threadIdx.x / kWarpSize) * kRowsPerWarp;
  if (row0 >= n) {
    return;
  }
  const int chunks_per_row = k / 32;
  const int num_groups = k / group_size;

  uint32_t acc = 0;
  for (int chunk = lane; chunk < chunks_per_row; chunk += kWarpSize) {
    uint4 packed[kRowsPerWarp];
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
      packed[r] = qweight[static_cast<int64_t>(min(row0 + r, n - 1)) * chunks_per_row + chunk];
    }
    const int g = chunk * 32 / group_size;
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
      const int64_t row = min(row0 + r, n - 1);
      acc ^= packed[r].x ^ packed[r].y ^ packed[r].z ^ packed[r].w;
      acc ^= __half_as_ushort(scales[row * num_groups + g]) ^
             __half_as_ushort(zeros[row * num_groups + g]);
    }
#pragma unroll
    for (int word = 0; word < 4; ++word) {
      const uint4 xv = *reinterpret_cast<const uint4*>(x + chunk * 32 + word * 8);
      acc ^= xv.x ^ xv.y ^ xv.z ^ xv.w;
    }
  }

  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    acc ^= __shfl_xor_sync(0xffffffff, acc, offset);
  }
  if (lane == 0) {
    out[row0] = acc;
  }
}

}  // namespace

at::Tensor load_only(const at::Tensor& x, const at::Tensor& qweight, const at::Tensor& scales,
                     const at::Tensor& zeros, int64_t group_size, int64_t threads_per_block,
                     int64_t rows_per_warp) {
  check_inputs("load_only", x, qweight, scales, zeros, group_size, c10::DeviceType::CUDA);
  check_threads_per_block("load_only", threads_per_block);
  TORCH_CHECK(x.size(0) == 1, "load_only: needs M = 1, got ", x.size(0));
  check_16_byte_loads("load_only", x, qweight, group_size);

  const c10::cuda::CUDAGuard guard(x.device());
  const int k = static_cast<int>(x.size(1));
  const int n = static_cast<int>(qweight.size(0));
  const int threads = static_cast<int>(threads_per_block);
  at::Tensor out = qweight.new_empty({n});
  if (n == 0) {
    return out;  // a launch with an empty grid would fail
  }
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  dispatch_rows_per_warp(rows_per_warp, [&]<int kRowsPerWarp>() {
    const int rows_per_block = threads / kWarpSize * kRowsPerWarp;
    check_launchable("load_only", load_only_kernel<kRowsPerWarp>, threads);
    load_only_kernel<kRowsPerWarp><<<at::ceil_div(n, rows_per_block), threads, 0, stream>>>(
        data_as<uint32_t>(out), data_as<half>(x), data_as<uint4>(qweight), data_as<half>(scales),
        data_as<half>(zeros), n, k, static_cast<int>(group_size));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });

  return out;
}

}  // namespace int4gemv
