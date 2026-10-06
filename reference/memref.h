#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nanodsp {

template <typename T, int N> struct MemRef {
  T *allocated;
  T *aligned;
  std::int64_t offset;
  std::int64_t sizes[N];
  std::int64_t strides[N];
};

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

}
