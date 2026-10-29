#include <ATen/ceil_div.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>

#include <cstdint>

#include "common.cuh"
#include "w4a16.h"

namespace int4gemv {
namespace {

constexpr int kThreadsPerBlock = 128;

// Computes y[m0 ... m0 + kTileM - 1, row] for one weight row per thread: the thread walks the
// row's K / 8 packed words, unpacks each weight once and adds it times every x row of the tile.
//
// The baseline. Neighbouring threads read addresses K / 2 bytes apart, so a warp's load touches
// 32 sectors, but L1 keeps them for the next 7 words and DRAM still reads each byte once. The
// limit is parallelism: only N / 32 warps run (256 for N = 8192), 8% achieved occupancy on an
// RTX 3090.
template <int kTileM>
__global__ void naive_kernel(half* __restrict__ y, const half* __restrict__ x,
                             const uint32_t* __restrict__ qweight, const half* __restrict__ scales,
                             const half* __restrict__ zeros, int m, int n, int k, int group_size) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) {
    return;
  }
  const int m0 = blockIdx.y * kTileM;
  const int words_per_row = k / 8;
  const int num_groups = k / group_size;
  const uint32_t* q_row = qweight + static_cast<int64_t>(row) * words_per_row;
  const half* s_row = scales + static_cast<int64_t>(row) * num_groups;
  const half* z_row = zeros + static_cast<int64_t>(row) * num_groups;

  const half* x_rows[kTileM];
  tile_x_rows(x_rows, x, m, m0, k);

  float acc[kTileM] = {};
  for (int word = 0; word < words_per_row; ++word) {
    const uint32_t packed = q_row[word];
    const int g = word * 8 / group_size;
    const float s = __half2float(s_row[g]);
    const float z = __half2float(z_row[g]);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float w = dequant(packed, j, s, z);
#pragma unroll
      for (int i = 0; i < kTileM; ++i) {
        acc[i] += __half2float(x_rows[i][word * 8 + j]) * w;
      }
    }
  }

#pragma unroll
  for (int i = 0; i < kTileM; ++i) {
    if (m0 + i < m) {
      y[static_cast<int64_t>(m0 + i) * n + row] = __float2half(acc[i]);
    }
  }
}

}  // namespace

at::Tensor naive(const at::Tensor& x, const at::Tensor& qweight, const at::Tensor& scales,
                 const at::Tensor& zeros, int64_t group_size) {
  check_inputs("naive", x, qweight, scales, zeros, group_size, c10::DeviceType::CUDA);

  const c10::cuda::CUDAGuard guard(x.device());
  const int m = static_cast<int>(x.size(0));
  const int k = static_cast<int>(x.size(1));
  const int n = static_cast<int>(qweight.size(0));
  at::Tensor y = x.new_empty({m, n});
  if (m == 0 || n == 0) {
    return y;  // a launch with an empty grid would fail
  }
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  dispatch_tile_m(m, [&]<int kTileM>() {
    const dim3 grid(at::ceil_div(n, kThreadsPerBlock), at::ceil_div(m, kTileM));
    naive_kernel<kTileM><<<grid, kThreadsPerBlock, 0, stream>>>(
        data_as<half>(y), data_as<half>(x), data_as<uint32_t>(qweight), data_as<half>(scales),
        data_as<half>(zeros), m, n, k, static_cast<int>(group_size));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });

  return y;
}

KernelInfo naive_info(int64_t m) {
  KernelInfo info;
  dispatch_tile_m(
      m, [&]<int kTileM>() { info = kernel_info(naive_kernel<kTileM>, kThreadsPerBlock); });
  return info;
}

}  // namespace int4gemv
