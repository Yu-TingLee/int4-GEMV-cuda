#include <torch/extension.h>

#include "w4a16.h"

namespace int4gemv {
namespace {

// From bench.block_sweep, 70B Q Proj. At M = 1, 1 and 2 rows per warp tie at 128 threads
// (52 µs) and the best config is 4% faster. At M = 8, 2 rows per warp take 137 µs against 232 µs
// for 1, since each slice of x loaded serves both rows; 4 rows run out of registers.
constexpr int64_t kDefaultThreadsPerBlock = 128;
constexpr int64_t kDefaultRowsPerWarp = 2;

}  // namespace
}  // namespace int4gemv

using int4gemv::kDefaultRowsPerWarp;
using int4gemv::kDefaultThreadsPerBlock;

PYBIND11_MODULE(int4gemv_ops, m) {
  m.def("sequential", &int4gemv::sequential, py::arg("x"), py::arg("qweight"), py::arg("scales"),
        py::arg("zeros"), py::arg("group_size"));
  m.def("naive", &int4gemv::naive, py::arg("x"), py::arg("qweight"), py::arg("scales"),
        py::arg("zeros"), py::arg("group_size"));
  m.def("coalesced", &int4gemv::coalesced, py::arg("x"), py::arg("qweight"), py::arg("scales"),
        py::arg("zeros"), py::arg("group_size"));
  m.def("vectorized", &int4gemv::vectorized, py::arg("x"), py::arg("qweight"), py::arg("scales"),
        py::arg("zeros"), py::arg("group_size"),
        py::arg("threads_per_block") = kDefaultThreadsPerBlock,
        py::arg("rows_per_warp") = kDefaultRowsPerWarp);
  m.def("load_only", &int4gemv::load_only, py::arg("x"), py::arg("qweight"), py::arg("scales"),
        py::arg("zeros"), py::arg("group_size"),
        py::arg("threads_per_block") = kDefaultThreadsPerBlock,
        py::arg("rows_per_warp") = kDefaultRowsPerWarp);

  m.def("naive_info", &int4gemv::naive_info, py::arg("m"));
  m.def("coalesced_info", &int4gemv::coalesced_info, py::arg("m"));
  m.def("vectorized_info", &int4gemv::vectorized_info, py::arg("m"),
        py::arg("threads_per_block") = kDefaultThreadsPerBlock,
        py::arg("rows_per_warp") = kDefaultRowsPerWarp);
}
