#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

using Clock = std::chrono::steady_clock;

static double since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

static constexpr std::size_t kTriadN = std::size_t(32) << 20;
static constexpr int kTriadReps = 10;

__attribute__((noinline)) static void triad(float *__restrict a,
                                            const float *__restrict b,
                                            const float *__restrict c, float s,
                                            std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    a[i] = b[i] + s * c[i];
}

struct Bw {
  double best_gbs, median_gbs;
};

static Bw measure_triad() {
  std::vector<float> a(kTriadN, 0.0f), b(kTriadN, 1.0f), c(kTriadN, 2.0f);
  volatile float s = 3.0f;
  triad(a.data(), b.data(), c.data(), s, kTriadN);
  std::vector<double> gbs;

  for (int r = 0; r < kTriadReps; ++r) {
    auto t0 = Clock::now();
    triad(a.data(), b.data(), c.data(), s, kTriadN);
    gbs.push_back(12.0 * double(kTriadN) / since(t0) / 1e9);
  }

  std::sort(gbs.begin(), gbs.end());

  if (a[kTriadN / 2] != 7.0f)
    std::fprintf(stderr, "triad produced a wrong value\n");

  return {gbs.back(), gbs[gbs.size() / 2]};
}

#if defined(__ARM_NEON)
static constexpr int kAcc = 24;
static constexpr long kIters = 20'000'000;

__attribute__((noinline)) static float fma_loop(float x, float y, long iters) {
  float32x4_t acc[kAcc];

  for (int i = 0; i < kAcc; ++i)
    acc[i] = vdupq_n_f32(float(i) * 1e-3f);

  const float32x4_t vx = vdupq_n_f32(x), vy = vdupq_n_f32(y);

  for (long it = 0; it < iters; ++it)
    for (int i = 0; i < kAcc; ++i)
      acc[i] = vfmaq_f32(acc[i], vx, vy);

  float32x4_t s = acc[0];

  for (int i = 1; i < kAcc; ++i)
    s = vaddq_f32(s, acc[i]);

  return vaddvq_f32(s);
}

__attribute__((noinline)) static float mul_add_loop(float x, float y,
                                                    long iters) {
  float32x4_t acc[kAcc];

  for (int i = 0; i < kAcc; ++i)
    acc[i] = vdupq_n_f32(float(i) * 1e-3f);

  const float32x4_t vx = vdupq_n_f32(x), vy = vdupq_n_f32(y);

  for (long it = 0; it < iters; ++it)
    for (int i = 0; i < kAcc; ++i)
      acc[i] = vaddq_f32(vmulq_f32(acc[i], vx), vy);

  float32x4_t s = acc[0];

  for (int i = 1; i < kAcc; ++i)
    s = vaddq_f32(s, acc[i]);

  return vaddvq_f32(s);
}

static volatile float g_sink;

template <typename F> static double best_gflops(F loop, int reps) {
  volatile float x = 0.999f, y = 1e-3f;
  g_sink = loop(x, y, kIters / 10);
  double best = 0;

  for (int r = 0; r < reps; ++r) {
    auto t0 = Clock::now();
    g_sink = loop(x, y, kIters);
    const double flops = 8.0 * kAcc * double(kIters);
    best = std::max(best, flops / since(t0) / 1e9);
  }

  return best;
}
#endif

int main(int argc, char **argv) {
  const char *json_path = nullptr;

  if (argc == 3 && std::strcmp(argv[1], "--json") == 0)
    json_path = argv[2];
  else if (argc != 1) {
    std::fprintf(stderr, "usage: %s [--json path]\n", argv[0]);

    return 2;
  }
#if defined(__APPLE__)
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

  const Bw bw = measure_triad();
  std::printf("triad bandwidth     %8.2f GB/s best, %.2f median "
              "(%d runs, 3 x %zu MiB arrays)\n",
              bw.best_gbs, bw.median_gbs, kTriadReps,
              kTriadN * sizeof(float) >> 20);
#if defined(__ARM_NEON)
  const double fma = best_gflops(fma_loop, 5);
  const double mul_add = best_gflops(mul_add_loop, 5);
  std::printf(
      "NEON FMA peak       %8.2f GFLOP/s (%d accumulators, best of 5)\n", fma,
      kAcc);
  std::printf("NEON mul+add peak   %8.2f GFLOP/s (unfused, best of 5)\n",
              mul_add);
#endif

  if (!json_path)
    return 0;

  std::FILE *f = std::fopen(json_path, "w");

  if (!f) {
    std::perror(json_path);

    return 2;
  }

  std::fprintf(f,
               "{\n    \"threads\": 1,\n"
               "    \"triad_bw\": {\"value\": %.2f, \"unit\": \"GB/s\", "
               "\"median\": %.2f, \"runs\": %d, \"array_bytes\": %zu, "
               "\"method\": \"a[i] = b[i] + s*c[i], 3 float arrays, "
               "12 B/element counted (no write-allocate), best run\"}",
               bw.best_gbs, bw.median_gbs, kTriadReps, kTriadN * sizeof(float));
#if defined(__ARM_NEON)
  std::fprintf(f,
               ",\n    \"neon_fma_peak\": {\"value\": %.2f, \"unit\": "
               "\"GFLOP/s\", \"accumulators\": %d, \"method\": \"vfmaq_f32 "
               "on independent 4-lane accumulators, 8 flops/instruction, "
               "best of 5\"},\n"
               "    \"neon_mul_add_peak\": {\"value\": %.2f, \"unit\": "
               "\"GFLOP/s\", \"accumulators\": %d, \"method\": \"vmulq_f32 + "
               "vaddq_f32 (unfused, like the bit-exact kernels), 8 flops per "
               "pair, best of 5\"}",
               fma, kAcc, mul_add, kAcc);
#else
  std::fprintf(f, ",\n    \"neon_fma_peak\": null,\n"
                  "    \"neon_mul_add_peak\": null");
#endif
  std::fprintf(f, "\n  }");
  std::fclose(f);

  return 0;
}
