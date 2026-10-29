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

// Computes y[m0 ... m0 + kTileM - 1, row0 ... row0 + kRowsPerWarp - 1] for kRowsPerWarp weight
// rows per warp: lane l loads 16-byte chunks l, l + 32, ... (32 weights each) of every row,
// unpacks them, and a shuffle adds up the partial sums as in coalesced.
//
// Faster than coalesced because it keeps more bytes in flight. Memory takes hundreds of cycles to
// answer, so the bandwidth we get is roughly bytes in flight / latency. Three things raise it:
//   - bigger blocks, since one-warp blocks hit the per-SM limit of 16 blocks at a third of the
//     warp slots on an RTX 3090 (16 of 48);
//   - several rows per warp, so each thread has independent loads going at once and each slice
//     of x it loads serves every row (which matters at M > 1);
//   - 16-byte loads, 4x the bytes per load instruction.
// A 16-byte chunk never straddles two groups because group_size % 32 == 0.
template <int kTileM, int kRowsPerWarp>
__global__ void vectorized_kernel(half* __restrict__ y, const half* __restrict__ x,
                                  const uint4* __restrict__ qweight,
                                  const half* __restrict__ scales, const half* __restrict__ zeros,
                                  int m, int n, int k, int group_size) {
  const int lane = threadIdx.x % kWarpSize;
  const int warps_per_block = blockDim.x / kWarpSize;
  const int row0 = (blockIdx.x * warps_per_block + threadIdx.x / kWarpSize) * kRowsPerWarp;
  // Fine to exit early: whole warps leave together and nothing below syncs the block.
  if (row0 >= n) {
    return;
  }
  const int m0 = blockIdx.y * kTileM;
  const int chunks_per_row = k / 32;
  const int num_groups = k / group_size;

  // Rows past n or m reuse the last valid row and are never written.
  const uint4* q_rows[kRowsPerWarp];
  const half* s_rows[kRowsPerWarp];
  const half* z_rows[kRowsPerWarp];
#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r) {
    const int64_t row = min(row0 + r, n - 1);
    q_rows[r] = qweight + row * chunks_per_row;
    s_rows[r] = scales + row * num_groups;
    z_rows[r] = zeros + row * num_groups;
  }
  const half* x_rows[kTileM];
  tile_x_rows(x_rows, x, m, m0, k);

  float acc[kRowsPerWarp][kTileM] = {};
  for (int chunk = lane; chunk < chunks_per_row; chunk += kWarpSize) {
    // Start every row's load before using any of them.
    uint4 packed[kRowsPerWarp];
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
      packed[r] = q_rows[r][chunk];
    }
    const int g = chunk * 32 / group_size;
    float s[kRowsPerWarp];
    float z[kRowsPerWarp];
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
      s[r] = __half2float(s_rows[r][g]);
      z[r] = __half2float(z_rows[r][g]);
    }

#pragma unroll
    for (int word = 0; word < 4; ++word) {
      float w[kRowsPerWarp][8];
#pragma unroll
      for (int r = 0; r < kRowsPerWarp; ++r) {
        const uint32_t bits = reinterpret_cast<const uint32_t*>(&packed[r])[word];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          w[r][j] = dequant(bits, j, s[r], z[r]);
        }
      }
#pragma unroll
      for (int i = 0; i < kTileM; ++i) {
        const uint4 xv = *reinterpret_cast<const uint4*>(x_rows[i] + chunk * 32 + word * 8);
        const auto* xh = reinterpret_cast<const half2*>(&xv);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 xf = __half22float2(xh[j]);
#pragma unroll
          for (int r = 0; r < kRowsPerWarp; ++r) {
            acc[r][i] += xf.x * w[r][2 * j] + xf.y * w[r][2 * j + 1];
          }
        }
      }
    }
  }

#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r) {
#pragma unroll
    for (int i = 0; i < kTileM; ++i) {
      acc[r][i] = warp_sum(acc[r][i]);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
#pragma unroll
      for (int i = 0; i < kTileM; ++i) {
        if (row0 + r < n && m0 + i < m) {
          y[static_cast<int64_t>(m0 + i) * n + row0 + r] = __float2half(acc[r][i]);
        }
      }
    }
  }
}

}  // namespace

at::Tensor vectorized(const at::Tensor& x, const at::Tensor& qweight, const at::Tensor& scales,
                      const at::Tensor& zeros, int64_t group_size, int64_t threads_per_block,
                      int64_t rows_per_warp) {
  check_inputs("vectorized", x, qweight, scales, zeros, group_size, c10::DeviceType::CUDA);
  check_threads_per_block("vectorized", threads_per_block);
  check_16_byte_loads("vectorized", x, qweight, group_size);

  const c10::cuda::CUDAGuard guard(x.device());
  const int m = static_cast<int>(x.size(0));
  const int k = static_cast<int>(x.size(1));
  const int n = static_cast<int>(qweight.size(0));
  const int threads = static_cast<int>(threads_per_block);
  at::Tensor y = x.new_empty({m, n});
  if (m == 0 || n == 0) {
    return y;  // a launch with an empty grid would fail
  }
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  dispatch_rows_per_warp(rows_per_warp, [&]<int kRowsPerWarp>() {
    dispatch_tile_m(m, [&]<int kTileM>() {
      const int rows_per_block = threads / kWarpSize * kRowsPerWarp;
      const dim3 grid(at::ceil_div(n, rows_per_block), at::ceil_div(m, kTileM));
      check_launchable("vectorized", vectorized_kernel<kTileM, kRowsPerWarp>, threads);
      vectorized_kernel<kTileM, kRowsPerWarp><<<grid, threads, 0, stream>>>(
          data_as<half>(y), data_as<half>(x), data_as<uint4>(qweight), data_as<half>(scales),
          data_as<half>(zeros), m, n, k, static_cast<int>(group_size));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
  });

  return y;
}

KernelInfo vectorized_info(int64_t m, int64_t threads_per_block, int64_t rows_per_warp) {
  check_threads_per_block("vectorized_info", threads_per_block);

  KernelInfo info;
  dispatch_rows_per_warp(rows_per_warp, [&]<int kRowsPerWarp>() {
    dispatch_tile_m(m, [&]<int kTileM>() {
      info =
          kernel_info(vectorized_kernel<kTileM, kRowsPerWarp>, static_cast<int>(threads_per_block));
    });
  });
  return info;
}

}  // namespace int4gemv
