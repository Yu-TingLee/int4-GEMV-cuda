#include <c10/util/Half.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "w4a16.h"

namespace int4gemv {

at::Tensor sequential(const at::Tensor& x, const at::Tensor& qweight, const at::Tensor& scales,
                      const at::Tensor& zeros, int64_t group_size) {
  check_inputs("sequential", x, qweight, scales, zeros, group_size, c10::DeviceType::CPU);

  const int64_t m = x.size(0);
  const int64_t k = x.size(1);
  const int64_t n = qweight.size(0);
  const int64_t num_groups = k / group_size;
  const at::Tensor x_float = x.to(at::kFloat);
  const float* xp = x_float.data_ptr<float>();
  const auto* qp = reinterpret_cast<const uint32_t*>(qweight.data_ptr<int32_t>());
  const auto* sp = scales.data_ptr<c10::Half>();
  const auto* zp = zeros.data_ptr<c10::Half>();
  at::Tensor out = x.new_empty({m, n});
  auto* yp = out.data_ptr<c10::Half>();
  std::vector<float> acc(m);
  std::vector<float> w(group_size);

  for (int64_t row = 0; row < n; ++row) {
    std::fill(acc.begin(), acc.end(), 0.0f);
    for (int64_t g = 0; g < num_groups; ++g) {
      const float s = static_cast<float>(sp[row * num_groups + g]);
      const float z = static_cast<float>(zp[row * num_groups + g]);
      const uint32_t* words = qp + row * (k / 8) + g * (group_size / 8);
      for (int64_t i = 0; i < group_size; ++i) {
        const uint32_t q = (words[i / 8] >> (4 * (i % 8))) & 0xF;
        w[i] = (static_cast<float>(q) - z) * s;
      }
      for (int64_t r = 0; r < m; ++r) {
        const float* xg = xp + r * k + g * group_size;
        float sum = 0.0f;
        for (int64_t i = 0; i < group_size; ++i) {
          sum += xg[i] * w[i];
        }
        acc[r] += sum;
      }
    }
    for (int64_t r = 0; r < m; ++r) {
      yp[r * n + row] = static_cast<c10::Half>(acc[r]);
    }
  }

  return out;
}

}  // namespace int4gemv
