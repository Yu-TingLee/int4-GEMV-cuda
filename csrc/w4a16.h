#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/DeviceType.h>

#include <cstdint>
#include <map>
#include <string>

// y = x @ dequant(W).T, where each weight is w = (q - z) * s and every group_size weights along K
// share one scale s and zero z. All kernels take
//   x        [M, K]          fp16
//   qweight  [N, K / 8]      int32, eight 4-bit q per word, lowest bits first
//   scales   [N, K / group]  fp16
//   zeros    [N, K / group]  fp16
// and return y [M, N] fp16 (empty if M or N is 0). Wrong dtypes, shapes or devices throw.
namespace int4gemv {

// Registers, spills (local_bytes) and theoretical occupancy of one compiled kernel.
using KernelInfo = std::map<std::string, int64_t>;

// Plain single-threaded CPU version. Only used to cross-check the reference in tests.
at::Tensor sequential(const at::Tensor& x, const at::Tensor& qweight, const at::Tensor& scales,
                      const at::Tensor& zeros, int64_t group_size);

// One thread per output row.
at::Tensor naive(const at::Tensor& x, const at::Tensor& qweight, const at::Tensor& scales,
                 const at::Tensor& zeros, int64_t group_size);
KernelInfo naive_info(int64_t m);

// One warp per output row, so its loads coalesce.
at::Tensor coalesced(const at::Tensor& x, const at::Tensor& qweight, const at::Tensor& scales,
                     const at::Tensor& zeros, int64_t group_size);
KernelInfo coalesced_info(int64_t m);

// Like coalesced, plus several warps per block, rows_per_warp rows per warp (1, 2 or 4), 16-byte
// loads. group_size must be a multiple of 32, and x and qweight 16-byte aligned.
at::Tensor vectorized(const at::Tensor& x, const at::Tensor& qweight, const at::Tensor& scales,
                      const at::Tensor& zeros, int64_t group_size, int64_t threads_per_block,
                      int64_t rows_per_warp);
KernelInfo vectorized_info(int64_t m, int64_t threads_per_block, int64_t rows_per_warp);

// The vectorized kernel's loads without the math, to see what the memory side alone can do. M = 1
// only, same constraints as vectorized.
at::Tensor load_only(const at::Tensor& x, const at::Tensor& qweight, const at::Tensor& scales,
                     const at::Tensor& zeros, int64_t group_size, int64_t threads_per_block,
                     int64_t rows_per_warp);

// Shared input checks; name goes in front of the error message.
void check_inputs(const char* name, const at::Tensor& x, const at::Tensor& qweight,
                  const at::Tensor& scales, const at::Tensor& zeros, int64_t group_size,
                  c10::DeviceType device);

}  // namespace int4gemv
