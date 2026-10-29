#include <c10/util/Exception.h>

#include <limits>

#include "w4a16.h"

namespace int4gemv {

void check_inputs(const char* name, const at::Tensor& x, const at::Tensor& qweight,
                  const at::Tensor& scales, const at::Tensor& zeros, int64_t group_size,
                  c10::DeviceType device) {
  TORCH_CHECK(x.scalar_type() == at::kHalf && qweight.scalar_type() == at::kInt &&
                  scales.scalar_type() == at::kHalf && zeros.scalar_type() == at::kHalf,
              name, ": expected fp16 x, int32 qweight, fp16 scales and zeros, got ",
              x.scalar_type(), ", ", qweight.scalar_type(), ", ", scales.scalar_type(), ", ",
              zeros.scalar_type());
  TORCH_CHECK(x.dim() == 2 && qweight.dim() == 2 && scales.dim() == 2 && zeros.dim() == 2, name,
              ": expected 2-D inputs");

  const int64_t m = x.size(0);
  const int64_t k = x.size(1);
  const int64_t n = qweight.size(0);
  // The kernels index rows and columns with int.
  constexpr int64_t kIntMax = std::numeric_limits<int>::max();
  TORCH_CHECK(m <= kIntMax && n <= kIntMax && k <= kIntMax, name, ": M = ", m, ", N = ", n,
              ", K = ", k, " must each fit in int");
  TORCH_CHECK(group_size > 0 && group_size % 8 == 0 && k % group_size == 0, name, ": group_size ",
              group_size, " must be a positive multiple of 8 dividing K = ", k);
  TORCH_CHECK(qweight.size(1) * 8 == k, name, ": qweight of shape ", qweight.sizes(),
              " does not pack K = ", k);
  TORCH_CHECK(
      scales.size(0) == n && scales.size(1) == k / group_size && zeros.sizes() == scales.sizes(),
      name, ": scales ", scales.sizes(), " and zeros ", zeros.sizes(), " must be [", n, ", ",
      k / group_size, "]");
  TORCH_CHECK(x.is_contiguous() && qweight.is_contiguous() && scales.is_contiguous() &&
                  zeros.is_contiguous(),
              name, ": inputs must be contiguous");

  for (const at::Tensor* t : {&x, &qweight, &scales, &zeros}) {
    TORCH_CHECK(t->device().type() == device && t->device() == x.device(), name,
                ": expected all inputs on one ", device, " device, got ", x.device(), " and ",
                t->device());
  }
}

}  // namespace int4gemv
