#include "../../reference/nanodsp_ref.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace nanodsp::ref;

extern "C" {
void nd_add_f32(const float *a, const float *b, float *c, std::int32_t n);
void nd_matmul_f32(const float *a, const float *b, float *c, std::int32_t m,
                   std::int32_t k, std::int32_t n);
void nd_qmatmul_i8(const std::int8_t *a, const std::int8_t *b, std::int8_t *c,
                   std::int32_t *acc, std::int32_t m, std::int32_t k,
                   std::int32_t n, std::int32_t lhs_zp, std::int32_t rhs_zp,
                   std::int32_t multiplier, std::int32_t shift,
                   std::int32_t out_zp);
}

namespace {

int failures = 0;

float fill(std::size_t i, std::size_t j) {
  return float((7 * i + 3 * j) % 13) * 0.37f - 1.9f;
}

template <typename T>
void compare(const char *name, const std::vector<T> &actual,
             const std::vector<T> &expected) {
  std::size_t bad = 0;

  for (std::size_t i = 0; i < expected.size(); ++i)
    bad += std::memcmp(&actual[i], &expected[i], sizeof(T)) != 0;

  std::printf("%s mojo %-16s %zu/%zu values bit-identical\n",
              bad != 0 ? "FAIL" : "PASS", name, expected.size() - bad,
              expected.size());
  failures += bad != 0 ? 1 : 0;
}

void check_add(std::size_t n) {
  Tensor a({n});
  Tensor b({n});

  for (std::size_t i = 0; i < n; ++i) {
    a.data[i] = fill(i, 1);
    b.data[i] = fill(2, i);
  }

  std::vector<float> c(n);
  nd_add_f32(a.data.data(), b.data.data(), c.data(), std::int32_t(n));
  compare("add", c, add(a, b).data);
}

void check_matmul(const char *name, std::size_t m, std::size_t k,
                  std::size_t n) {
  Tensor a({m, k});
  Tensor b({k, n});

  for (std::size_t i = 0; i < m; ++i)
    for (std::size_t j = 0; j < k; ++j)
      a.data[i * k + j] = fill(i, j);

  for (std::size_t i = 0; i < k; ++i)
    for (std::size_t j = 0; j < n; ++j)
      b.data[i * n + j] = fill(j, i);

  std::vector<float> c(m * n);
  nd_matmul_f32(a.data.data(), b.data.data(), c.data(), std::int32_t(m),
                std::int32_t(k), std::int32_t(n));
  compare(name, c, matmul(a, b).data);
}

void check_qmatmul(const char *name, const QTensor &a, const QTensor &b,
                   const QuantParams &q) {
  std::size_t m = a.shape[0];
  std::size_t k = a.shape[1];
  std::size_t n = b.shape[1];
  std::vector<std::int8_t> c(m * n);
  std::vector<std::int32_t> acc(n);
  nd_qmatmul_i8(a.data.data(), b.data.data(), c.data(), acc.data(),
                std::int32_t(m), std::int32_t(k), std::int32_t(n), q.lhs_zp,
                q.rhs_zp, q.multiplier, q.shift, q.out_zp);
  compare(name, c, qmatmul(a, b, q).data);
}

}

int main() {
  check_add(1003);
  check_matmul("matmul", 64, 96, 48);
  check_matmul("matmul edges", 13, 20, 37);

  check_qmatmul("qmatmul golden", QTensor({2, 3}, {-128, 0, 127, 10, -7, 50}),
                QTensor({3, 4}, {1, -3, 127, -128, 4, 2, 0, 9, -1, 5, -55, 30}),
                {3, -2, 1073741824, 3, -5});

  std::vector<std::int8_t> av(32 * 128);
  std::vector<std::int8_t> bv(128 * 64);

  for (std::size_t i = 0; i < av.size(); ++i)
    av[i] = std::int8_t(int((i * 73 + 11) % 256) - 128);

  for (std::size_t i = 0; i < bv.size(); ++i)
    bv[i] = std::int8_t(int((i * 151 + 7) % 256) - 128);

  check_qmatmul("qmatmul", QTensor({32, 128}, av), QTensor({128, 64}, bv),
                {-7, 12, 1276901417, 9, 4});

  return failures == 0 ? 0 : 1;
}
