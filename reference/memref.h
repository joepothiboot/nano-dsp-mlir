// MLIR's C interface for ranked memrefs, for host programs that call kernels
// compiled with `llvm.emit_c_interface` (test/Hexagon/harness.cpp,
// benchmarks/harness.cpp). A `_mlir_ciface_f(&out, &a, &b)` call takes and
// returns pointers to these descriptors; index is 64-bit on every target
// here, including 32-bit Hexagon (see test/Hexagon/kernels-f32.mlir).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nanodsp {

// Descriptor of an identity-layout memref of rank N.
template <typename T, int N> struct MemRef {
  T *allocated;
  T *aligned;
  std::int64_t offset;
  std::int64_t sizes[N];
  std::int64_t strides[N];
};

// Row-major descriptor over `data` (which keeps ownership).
template <typename T, int N>
MemRef<T, N> wrap(std::vector<T> &data, const std::vector<std::size_t> &shape) {
  MemRef<T, N> m{data.data(), data.data(), 0, {}, {}};
  std::int64_t stride = 1;
  for (int d = N - 1; d >= 0; --d) {
    m.sizes[d] = static_cast<std::int64_t>(shape[d]);
    m.strides[d] = stride;
    stride *= m.sizes[d];
  }
  return m;
}

} // namespace nanodsp
