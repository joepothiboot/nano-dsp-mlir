#include "../reference/memref.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

using nanodsp::MemRef;

extern "C" void _mlir_ciface_matmul_512_sweep(MemRef<float, 2> *,
                                              MemRef<float, 2> *,
                                              MemRef<float, 2> *);

static constexpr std::size_t kN = 512;
static constexpr int kSamples = 10;
static constexpr double kMinSampleSeconds = 0.05;

static std::vector<float> matrix(std::size_t salt) {
  std::vector<float> m(kN * kN);

  for (std::size_t i = 0; i < kN; ++i)
    for (std::size_t j = 0; j < kN; ++j)
      m[i * kN + j] = float((7 * i + 3 * j + 5 * salt) % 13) * 0.37f - 1.9f;

  return m;
}

static double seconds(MemRef<float, 2> &a, MemRef<float, 2> &b, long iters) {
  auto t0 = std::chrono::steady_clock::now();

  for (long i = 0; i < iters; ++i) {
    MemRef<float, 2> o;
    _mlir_ciface_matmul_512_sweep(&o, &a, &b);
    std::free(o.allocated);
  }

  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
      .count();
}

int main() {
#if defined(__APPLE__)
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

  std::vector<float> av = matrix(0), bv = matrix(5);
  MemRef<float, 2> a = nanodsp::wrap<float, 2>(av, {kN, kN});
  MemRef<float, 2> b = nanodsp::wrap<float, 2>(bv, {kN, kN});

  std::vector<float> expected(kN * kN);

  for (std::size_t i = 0; i < kN; ++i)
    for (std::size_t j = 0; j < kN; ++j) {
      float acc = 0.0f;

      for (std::size_t k = 0; k < kN; ++k) {
        float product = av[i * kN + k] * bv[k * kN + j];
        acc += product;
      }

      expected[i * kN + j] = acc;
    }

  MemRef<float, 2> o;
  _mlir_ciface_matmul_512_sweep(&o, &a, &b);
  bool exact = std::memcmp(o.aligned, expected.data(),
                           expected.size() * sizeof(float)) == 0;
  std::free(o.allocated);

  long iters = 1;

  while (seconds(a, b, iters) < kMinSampleSeconds)
    iters *= 2;

  double best = 1e30;

  for (int s = 0; s < kSamples; ++s)
    best = std::min(best, seconds(a, b, iters) / double(iters));

  std::printf("%.4f,%.2f,%s\n", best * 1e3, 2.0 * kN * kN * kN / best / 1e9,
              exact ? "yes" : "NO");

  return exact ? 0 : 1;
}
