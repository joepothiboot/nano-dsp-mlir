// Stage 5 benchmark harness: MLIR-compiled kernels (benchmarks/kernels.mlir,
// once untiled and once per schedule) against the scalar C++ reference
// (reference/nanodsp_ref.h). Built and run by scripts/bench.sh.
//
// Every implementation is checked against nanodsp::ref before it is timed,
// bit for bit (matmul, conv2d and qmatmul are all bit-exact by construction;
// see docs/05-soundness.md and docs/quantization.md). A mismatch aborts the
// run with a non-zero exit status and no timing.
//
//   harness --check              correctness only (CI)
//   harness --json <path>        check, time, write the "benchmarks" array
//
// Timing: one warmup call, then the call count per sample is doubled until a
// sample takes at least --min-time seconds; the reported time is the best
// (minimum) per-call time over --samples samples. Single thread. Each MLIR
// call returns a freshly allocated result (malloc), freed inside the timed
// loop, just as the reference allocates its result std::vector.
//
// Build with -ffp-contract=off (scripts/bench.sh does), so the reference is
// not contracted into FMAs and stays comparable bit for bit.
#include "../reference/memref.h"
#include "../reference/nanodsp_ref.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

using namespace nanodsp::ref;
using nanodsp::MemRef;
using nanodsp::wrap;

// Kernel name, element type, rank (operands and result share it). Must match
// the functions in benchmarks/kernels.mlir.
#define NANODSP_BENCH_KERNELS(X)                                               \
  X(matmul_64, float, 2)                                                       \
  X(matmul_128, float, 2)                                                      \
  X(matmul_256, float, 2)                                                      \
  X(matmul_512, float, 2)                                                      \
  X(conv2d_56_64_64, float, 4)                                                 \
  X(conv2d_28_128_128, float, 4)                                               \
  X(qmatmul_256, std::int8_t, 2)

// Symbol suffix (scripts/bench.sh appends `_<suffix>` to every function of
// the configuration), "impl" and "config" in the JSON.
#define NANODSP_BENCH_CONFIGS(X)                                               \
  X(untiled, "mlir-untiled", "none")                                           \
  X(scheduled, "mlir-scheduled", "host-neon")

#define DECLARE_ONE(name, T, N, suffix)                                        \
  void _mlir_ciface_##name##_##suffix(MemRef<T, N> *, MemRef<T, N> *,          \
                                      MemRef<T, N> *);
#define DECLARE_UNTILED(name, T, N) DECLARE_ONE(name, T, N, untiled)
#define DECLARE_SCHEDULED(name, T, N) DECLARE_ONE(name, T, N, scheduled)
extern "C" {
NANODSP_BENCH_KERNELS(DECLARE_UNTILED)
NANODSP_BENCH_KERNELS(DECLARE_SCHEDULED)
}

template <typename T, int N>
using Kernel = void (*)(MemRef<T, N> *, MemRef<T, N> *, MemRef<T, N> *);

// One kernel, all compiled configurations of it.
template <typename T, int N> struct KernelSet {
  const char *name;
  std::vector<std::pair<const char *, Kernel<T, N>>> by_suffix;
};

// Adding a configuration takes three edits: NANODSP_BENCH_CONFIGS, a
// DECLARE_* line above and an entry here.
#define DEFINE_SET(name, T, N)                                                 \
  static KernelSet<T, N> name##_set() {                                        \
    return {#name,                                                             \
            {{"untiled", &_mlir_ciface_##name##_untiled},                      \
             {"scheduled", &_mlir_ciface_##name##_scheduled}}};                \
  }
NANODSP_BENCH_KERNELS(DEFINE_SET)

struct ConfigInfo {
  const char *suffix, *impl, *config;
};
#define CONFIG_INFO(suffix, impl, config) {#suffix, impl, config},
static const ConfigInfo kConfigs[] = {NANODSP_BENCH_CONFIGS(CONFIG_INFO)};

static const ConfigInfo &config_of(const char *suffix) {
  for (const auto &c : kConfigs)
    if (std::strcmp(c.suffix, suffix) == 0)
      return c;
  std::fprintf(stderr, "unknown configuration suffix %s\n", suffix);
  std::exit(2);
}

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

// Fractional values whose partial sums round, as in
// test/Integration/Schedule/bit-exact.mlir: any change in summation order
// changes output bits.
static float fill(std::size_t i, std::size_t j, std::size_t k = 0,
                  std::size_t l = 0) {
  return float((7 * i + 3 * j + 5 * k + 11 * l) % 13) * 0.37f - 1.9f;
}

static Tensor matrix(std::size_t rows, std::size_t cols, std::size_t salt) {
  Tensor t({rows, cols});
  for (std::size_t i = 0; i < rows; ++i)
    for (std::size_t j = 0; j < cols; ++j)
      t.data[i * cols + j] = fill(i, j, salt);
  return t;
}

static Tensor tensor4(std::vector<std::size_t> s, std::size_t salt) {
  Tensor t(s);
  std::size_t idx = 0;
  for (std::size_t a = 0; a < s[0]; ++a)
    for (std::size_t b = 0; b < s[1]; ++b)
      for (std::size_t c = 0; c < s[2]; ++c)
        for (std::size_t d = 0; d < s[3]; ++d)
          t.data[idx++] = fill(a + salt, b, c, d);
  return t;
}

static QTensor qmatrix(std::size_t rows, std::size_t cols, unsigned mul,
                       unsigned add) {
  std::vector<std::int8_t> v(rows * cols);
  for (std::size_t i = 0; i < v.size(); ++i)
    v[i] = std::int8_t(int((i * mul + add) % 256) - 128);
  return QTensor({rows, cols}, std::move(v));
}

// Same parameters as @qmatmul_256 in benchmarks/kernels.mlir.
static const QuantParams kQuant{-7, 12, 1276901417, 9, 4};

// Per-element tolerance |got - expected| <= tol(i). Empty: bit-exact only.
using Tolerance = std::function<double(std::size_t)>;

// A schedule that blocks the conv's channel dim moves that block loop
// outside kh/kw, so it sums the same products in a different order
// (README.md in this directory; docs/05-soundness.md assumes it does not).
// Any two summation orders of n products stay within 2 * gamma_n * sum |x*w|
// of each other, gamma_n = n*u / (1 - n*u), u = 2^-24 (Higham, Accuracy and
// Stability of Numerical Algorithms, 2nd ed., sec. 3.1): each is within
// gamma_n * sum |x*w| of the exact sum. That is the tolerance for conv2d.
static Tolerance reorder_bound(const Tensor &in, const Tensor &f) {
  Tensor ai(in.shape), af(f.shape);
  for (std::size_t i = 0; i < in.data.size(); ++i)
    ai.data[i] = std::abs(in.data[i]);
  for (std::size_t i = 0; i < f.data.size(); ++i)
    af.data[i] = std::abs(f.data[i]);
  // sum |x*w| per output, in float: its own relative rounding error (at most
  // gamma_n, ~3.4e-5 for n = 576) is covered by the 1e-3 slack below.
  auto abs_sum = std::make_shared<Tensor>(conv2d(ai, af));
  const double n = double(f.shape[0] * f.shape[1] * f.shape[2]);
  const double u = std::ldexp(1.0, -24);
  const double gamma = n * u / (1 - n * u);
  return [abs_sum, gamma](std::size_t i) {
    return 2 * gamma * double(abs_sum->data[i]) * (1 + 1e-3);
  };
}

// ---------------------------------------------------------------------------
// Benchmark cases
// ---------------------------------------------------------------------------

enum class Checked { Fail, BitExact, WithinBound };

struct Impl {
  std::string impl, config;
  std::function<Checked()> check; // empty for the reference itself
  std::function<void()> run;
  Checked checked = Checked::BitExact; // the reference is its own oracle
};


struct Case {
  std::string op, shape;
  double flops; // f32 flops, or int ops (2 per multiply-accumulate) for int8
  double bytes; // compulsory traffic: every operand read once, result written
                // once
  std::vector<Impl> impls;
};

static volatile std::uint64_t sink; // keeps reference results alive

template <typename T, int N, typename Ref>
static std::vector<Impl> impls_for(KernelSet<T, N> set, std::vector<T> &a,
                                   const std::vector<std::size_t> &a_shape,
                                   std::vector<T> &b,
                                   const std::vector<std::size_t> &b_shape,
                                   Ref ref, Tolerance tol = nullptr) {
  auto expected = std::make_shared<decltype(ref())>(ref());
  std::vector<Impl> out;
  out.push_back({"cpp-ref", "clang++ -O3 -ffp-contract=off", nullptr, [ref] {
                   auto r = ref();
                   std::uint64_t bits = 0;
                   std::memcpy(&bits, &r.data.back(), sizeof(r.data.back()));
                   sink = sink + bits;
                 }});
  for (auto [suffix, fn] : set.by_suffix) {
    const ConfigInfo &cfg = config_of(suffix);
    MemRef<T, N> ma = wrap<T, N>(a, a_shape), mb = wrap<T, N>(b, b_shape);
    std::string label = std::string(set.name) + " " + cfg.impl;
    auto check = [=]() mutable {
      MemRef<T, N> o;
      fn(&o, &ma, &mb);
      bool ok = true;
      for (int d = 0; d < N; ++d)
        if (o.sizes[d] != std::int64_t(expected->shape[d])) {
          std::fprintf(stderr, "FAIL %s: result dim %d is %lld, expected %zu\n",
                       label.c_str(), d, (long long)o.sizes[d],
                       expected->shape[d]);
          ok = false;
        }
      if (!ok) {
        std::free(o.allocated);
        return Checked::Fail;
      }
      const T *got = o.aligned + o.offset;
      const std::size_t n = expected->data.size();
      std::size_t differ = 0, over = 0, first = 0;
      double worst = 0; // largest |err| / tol
      for (std::size_t i = 0; i < n; ++i) {
        if (std::memcmp(&got[i], &expected->data[i], sizeof(T)) == 0)
          continue;
        if (!differ)
          first = i;
        ++differ;
        if (!tol) {
          ++over;
          continue;
        }
        const double err = std::abs(double(got[i]) - double(expected->data[i]));
        worst = std::max(worst, err / tol(i));
        if (!(err <= tol(i)))
          ++over;
      }
      const double got_first = double(got[first]);
      std::free(o.allocated);
      if (over) {
        std::fprintf(stderr,
                     "FAIL %s: %zu/%zu values differ from nanodsp::ref%s; "
                     "first at %zu: got %.9g, expected %.9g\n",
                     label.c_str(), over, n,
                     tol ? " by more than the reordering bound" : "", first,
                     got_first, double(expected->data[first]));
        return Checked::Fail;
      }
      if (differ) {
        std::printf("PASS %-20s %-15s %zu/%zu values differ from "
                    "nanodsp::ref, all within the reordering bound "
                    "(worst %.3f of it)\n",
                    set.name, cfg.impl, differ, n, worst);
        return Checked::WithinBound;
      }
      std::printf("PASS %-20s %-15s %zu values bit-identical to "
                  "nanodsp::ref\n",
                  set.name, cfg.impl, n);
      return Checked::BitExact;
    };
    auto run = [=]() mutable {
      MemRef<T, N> o;
      fn(&o, &ma, &mb);
      std::free(o.allocated);
    };
    out.push_back({cfg.impl, cfg.config, check, run});
  }
  return out;
}

// Inputs live as long as the cases (descriptors point into them).
struct Inputs {
  std::vector<Tensor> f;
  std::vector<QTensor> q;
};

static std::vector<Case> make_cases(Inputs &in) {
  std::vector<Case> cases;
  in.f.reserve(16);
  in.q.reserve(4);

  auto matmul_case = [&](KernelSet<float, 2> set, std::size_t n) {
    Tensor &a = in.f.emplace_back(matrix(n, n, 0));
    Tensor &b = in.f.emplace_back(matrix(n, n, 5));
    const double dn = double(n);
    std::string s = std::to_string(n);
    cases.push_back({"matmul", s + "x" + s + "x" + s, 2 * dn * dn * dn,
                     3 * dn * dn * 4,
                     impls_for(set, a.data, a.shape, b.data, b.shape,
                               [&a, &b] { return matmul(a, b); })});
  };
  matmul_case(matmul_64_set(), 64);
  matmul_case(matmul_128_set(), 128);
  matmul_case(matmul_256_set(), 256);
  matmul_case(matmul_512_set(), 512);

  auto conv_case = [&](KernelSet<float, 4> set, std::size_t hw, std::size_t c,
                       std::size_t f) {
    Tensor &x = in.f.emplace_back(tensor4({1, hw, hw, c}, 0));
    Tensor &w = in.f.emplace_back(tensor4({3, 3, c, f}, 2));
    const double o = double(hw - 2);
    const double flops = 2 * o * o * double(f) * 9 * double(c);
    const double bytes =
        4 * (double(hw * hw * c) + double(9 * c * f) + o * o * double(f));
    cases.push_back({"conv2d",
                     std::to_string(hw) + "x" + std::to_string(hw) + "x" +
                         std::to_string(c) + "->" + std::to_string(f),
                     flops, bytes,
                     impls_for(set, x.data, x.shape, w.data, w.shape,
                               [&x, &w] { return conv2d(x, w); },
                               reorder_bound(x, w))});
  };
  conv_case(conv2d_56_64_64_set(), 56, 64, 64);
  conv_case(conv2d_28_128_128_set(), 28, 128, 128);

  {
    const std::size_t n = 256;
    QTensor &a = in.q.emplace_back(qmatrix(n, n, 73, 11));
    QTensor &b = in.q.emplace_back(qmatrix(n, n, 151, 7));
    const double dn = double(n);
    cases.push_back({"qmatmul", "256x256x256", 2 * dn * dn * dn, 3 * dn * dn,
                     impls_for(qmatmul_256_set(), a.data, a.shape, b.data,
                               b.shape,
                               [&a, &b] { return qmatmul(a, b, kQuant); })});
  }
  return cases;
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

struct Timing {
  double best_ns;
  long iterations; // calls per sample
  int samples;
};

static double seconds(const std::function<void()> &f, long iters) {
  auto t0 = std::chrono::steady_clock::now();
  for (long i = 0; i < iters; ++i)
    f();
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(t1 - t0).count();
}

static Timing time_best(const std::function<void()> &f, int samples,
                        double min_sample_s) {
  f(); // warmup: page in the inputs and the result allocation
  long iters = 1;
  double t = seconds(f, iters);
  while (t < min_sample_s) {
    iters *= 2;
    t = seconds(f, iters);
  }
  double best = t / double(iters);
  for (int s = 1; s < samples; ++s)
    best = std::min(best, seconds(f, iters) / double(iters));
  return {best * 1e9, iters, samples};
}

// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
  bool check_only = false;
  const char *json_path = nullptr;
  int samples = 10;
  double min_time = 0.05;
  std::string filter;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--check")
      check_only = true;
    else if (arg == "--json" && i + 1 < argc)
      json_path = argv[++i];
    else if (arg == "--samples" && i + 1 < argc)
      samples = std::max(1, std::atoi(argv[++i]));
    else if (arg == "--min-time" && i + 1 < argc)
      min_time = std::atof(argv[++i]);
    else if (arg == "--filter" && i + 1 < argc)
      filter = argv[++i];
    else {
      std::fprintf(stderr,
                   "usage: %s [--check] [--json path] [--samples n] "
                   "[--min-time s] [--filter substring]\n",
                   argv[0]);
      return 2;
    }
  }

#if defined(__APPLE__)
  // Ask the scheduler for a performance core; macOS has no hard affinity.
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

  Inputs inputs;
  std::vector<Case> cases = make_cases(inputs);

  // Correctness first, for everything; nothing is timed if anything fails.
  int failures = 0;
  for (auto &c : cases)
    for (auto &impl : c.impls)
      if (impl.check && (impl.checked = impl.check()) == Checked::Fail)
        ++failures;
  if (failures) {
    std::fprintf(stderr, "%d implementation(s) disagree with nanodsp::ref\n",
                 failures);
    return 1;
  }
  if (check_only)
    return 0;

  std::FILE *json = nullptr;
  if (json_path && !(json = std::fopen(json_path, "w"))) {
    std::perror(json_path);
    return 2;
  }
  if (json)
    std::fprintf(json, "[\n");
  bool first = true;
  std::printf("\n%-8s %-14s %-15s %-12s %12s %10s\n", "op", "shape", "impl",
              "config", "best (us)", "GFLOP/s");
  for (auto &c : cases)
    for (auto &impl : c.impls) {
      std::string name = c.op + "/" + c.shape + "/" + impl.impl;
      if (!filter.empty() && name.find(filter) == std::string::npos)
        continue;
      Timing t = time_best(impl.run, samples, min_time);
      const double gflops = c.flops / t.best_ns;
      std::printf("%-8s %-14s %-15s %-12s %12.2f %10.2f\n", c.op.c_str(),
                  c.shape.c_str(), impl.impl.c_str(),
                  impl.impl == "cpp-ref" ? "-" : impl.config.c_str(),
                  t.best_ns / 1e3, gflops);
      std::fflush(stdout);
      if (!json)
        continue;
      std::fprintf(json,
                   "%s    {\"name\": \"%s\", \"op\": \"%s\", \"shape\": "
                   "\"%s\", \"impl\": \"%s\", \"config\": \"%s\",\n"
                   "     \"real_time\": %.1f, \"time_unit\": \"ns\", "
                   "\"aggregate\": \"min\", \"iterations\": %ld, "
                   "\"samples\": %d,\n"
                   "     \"flops\": %.0f, \"gflops\": %.3f, \"bytes\": %.0f, "
                   "\"intensity\": %.3f, \"checked\": \"%s\"}",
                   first ? "" : ",\n", name.c_str(), c.op.c_str(),
                   c.shape.c_str(), impl.impl.c_str(), impl.config.c_str(),
                   t.best_ns, t.iterations, t.samples, c.flops, gflops,
                   c.bytes, c.flops / c.bytes,
                   impl.checked == Checked::BitExact ? "bit-exact"
                                                     : "within-bound");
      first = false;
    }
  if (json) {
    std::fprintf(json, "\n  ]");
    std::fclose(json);
  }
  return 0;
}
