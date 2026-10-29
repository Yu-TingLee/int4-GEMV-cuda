#pragma once

#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <cuda_fp16.h>

#include <cstdint>

#include "w4a16.h"

namespace int4gemv {

constexpr int kWarpSize = 32;

// Adds value across the warp. Every lane ends up with the total.
__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_xor_sync(0xffffffff, value, offset);
  }
  return value;
}

// Weight j (0..7) of a packed word, unpacked: (q - z) * s.
__device__ __forceinline__ float dequant(uint32_t word, int j, float s, float z) {
  return (static_cast<float>((word >> (4 * j)) & 0xF) - z) * s;
}

// Points x_rows at rows m0 ... m0 + kTileM - 1 of x [m, k]. The last tile can run past m: those
// rows reuse row m - 1, and callers never write their results.
template <int kTileM>
__device__ __forceinline__ void tile_x_rows(const half* (&x_rows)[kTileM], const half* x, int m,
                                            int m0, int k) {
#pragma unroll
  for (int i = 0; i < kTileM; ++i) {
    x_rows[i] = x + static_cast<int64_t>(min(m0 + i, m - 1)) * k;
  }
}

// Tensor data as the CUDA type a kernel takes (half for at::Half, uint32_t or uint4 for packed
// int32). check_inputs has already checked the dtypes.
template <typename T>
T* data_as(const at::Tensor& t) {
  return reinterpret_cast<T*>(t.data_ptr());
}

// Runs f with kTileM = the smallest power of two >= m, capped at 64. Each thread keeps kTileM
// sums, so one unpacked weight serves kTileM rows of x.
template <typename F>
void dispatch_tile_m(int64_t m, F&& f) {
  if (m == 1) {
    f.template operator()<1>();
  } else if (m == 2) {
    f.template operator()<2>();
  } else if (m <= 4) {
    f.template operator()<4>();
  } else if (m <= 8) {
    f.template operator()<8>();
  } else if (m <= 16) {
    f.template operator()<16>();
  } else if (m <= 32) {
    f.template operator()<32>();
  } else {
    f.template operator()<64>();
  }
}

template <typename F>
void dispatch_rows_per_warp(int64_t rows_per_warp, F&& f) {
  switch (rows_per_warp) {
    case 1:
      f.template operator()<1>();
      break;
    case 2:
      f.template operator()<2>();
      break;
    case 4:
      f.template operator()<4>();
      break;
    default:
      TORCH_CHECK(false, "rows_per_warp must be 1, 2 or 4, got ", rows_per_warp);
  }
}

inline void check_threads_per_block(const char* name, int64_t threads_per_block) {
  TORCH_CHECK(threads_per_block >= kWarpSize && threads_per_block <= 1024 &&
                  threads_per_block % kWarpSize == 0,
              name, ": threads_per_block must be a multiple of 32 in [32, 1024], got ",
              threads_per_block);
}

// The 16-byte loads need group_size % 32 == 0, so a load never straddles two groups (and K % 32
// == 0, since group_size divides K). With that, every row of x and qweight starts 16-byte aligned
// if the tensor does.
inline void check_16_byte_loads(const char* name, const at::Tensor& x, const at::Tensor& qweight,
                                int64_t group_size) {
  TORCH_CHECK(group_size % 32 == 0, name, ": group_size must be a multiple of 32, got ",
              group_size);
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(qweight.data_ptr()) % 16 == 0,
              name, ": x and qweight must be 16-byte aligned");
}

// An SM has 64K registers, so register-hungry configs can't use big blocks. Say so clearly
// instead of failing with a generic launch error.
template <typename Kernel>
void check_launchable(const char* name, Kernel kernel, int threads_per_block) {
  cudaFuncAttributes attr;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
  TORCH_CHECK(threads_per_block <= attr.maxThreadsPerBlock, name, ": this configuration uses ",
              attr.numRegs, " registers per thread, so blocks of at most ", attr.maxThreadsPerBlock,
              " threads fit, not ", threads_per_block);
}

// Comes from the CUDA runtime, so no profiler permissions are needed.
template <typename Kernel>
KernelInfo kernel_info(Kernel kernel, int threads_per_block) {
  cudaFuncAttributes attr;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
  int blocks_per_sm = 0;
  C10_CUDA_CHECK(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, threads_per_block, 0));
  return {{"registers", attr.numRegs},
          {"local_bytes", static_cast<int64_t>(attr.localSizeBytes)},
          {"threads_per_block", threads_per_block},
          {"max_threads_per_block", attr.maxThreadsPerBlock},
          {"blocks_per_sm", blocks_per_sm},
          {"warps_per_sm", blocks_per_sm * threads_per_block / kWarpSize}};
}

}  // namespace int4gemv
