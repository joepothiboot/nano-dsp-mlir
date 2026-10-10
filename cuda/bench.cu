#include "harness.h"
#include "variants.cuh"

#include "../reference/nanodsp_ref.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

using namespace nanodsp::cuda;

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,                   \
                   cudaGetErrorString(err));                                   \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

struct Shape {
  int m;
  int n;
  int k;
};

struct Oracle {
  std::vector<float> exact_f32;
  std::vector<double> exact_f64;
  std::vector<double> magnitude;
};

static std::vector<float> fast_reference(const std::vector<float> &a,
                                         const std::vector<float> &b, Shape s) {
  std::vector<float> c(static_cast<std::size_t>(s.m) * s.n, 0.0f);

  for (int i = 0; i < s.m; ++i)
    for (int kk = 0; kk < s.k; ++kk) {
      float x = a[i * s.k + kk];

      for (int j = 0; j < s.n; ++j)
        c[i * s.n + j] += x * b[kk * s.n + j];
    }

  return c;
}

static Oracle make_oracle(const std::vector<float> &a,
                          const std::vector<float> &b, Shape s) {
  Oracle o;
  o.exact_f32 = fast_reference(a, b, s);
  o.exact_f64.assign(o.exact_f32.size(), 0.0);
  o.magnitude.assign(o.exact_f32.size(), 0.0);

  for (int i = 0; i < s.m; ++i)
    for (int kk = 0; kk < s.k; ++kk) {
      double x = a[i * s.k + kk];

      for (int j = 0; j < s.n; ++j) {
        double y = b[kk * s.n + j];
        o.exact_f64[i * s.n + j] += x * y;
        o.magnitude[i * s.n + j] += std::fabs(x) * std::fabs(y);
      }
    }

  if (s.m * s.n * s.k <= 512 * 512 * 512) {
    nanodsp::ref::Tensor ta({std::size_t(s.m), std::size_t(s.k)}, a);
    nanodsp::ref::Tensor tb({std::size_t(s.k), std::size_t(s.n)}, b);

    if (nanodsp::ref::matmul(ta, tb).data != o.exact_f32)
      throw std::runtime_error("fast reference differs from nanodsp::ref");
  }

  return o;
}

static double bound_used(const std::vector<float> &got, const Oracle &o,
                         int k) {
  double u = std::ldexp(1.0, -24);
  double gamma = k * u / (1 - k * u);
  double worst = 0.0;

  for (std::size_t i = 0; i < got.size(); ++i) {
    if (o.magnitude[i] == 0.0)
      continue;

    double err = std::fabs(got[i] - o.exact_f64[i]);
    worst = std::max(worst, err / (gamma * o.magnitude[i]));
  }

  return worst;
}

static std::string context_json(const cudaDeviceProp &prop) {
  int driver = 0;
  int runtime = 0;
  int cublas = 0;
  CHECK_CUDA(cudaDriverGetVersion(&driver));
  CHECK_CUDA(cudaRuntimeGetVersion(&runtime));
  cublasGetVersion(cublas_handle(), &cublas);

  char buf[512];
  std::snprintf(buf, sizeof buf,
                "  \"context\": {\"device\": \"%s\", \"compute_capability\": "
                "\"%d.%d\", \"cuda_driver\": %d, \"cuda_runtime\": %d, "
                "\"cublas\": %d, \"nvcc\": \"%d.%d\", \"timing\": \"one launch "
                "+ synchronize per call; best of %d samples of >= 50 ms\"},",
                prop.name, prop.major, prop.minor, driver, runtime, cublas,
                __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, kSamples);

  return buf;
}

static void run_shape(Shape s, bool timed, const std::string &device,
                      std::vector<std::string> &rows) {
  std::vector<float> a = inexact(s.m, s.k, 1);
  std::vector<float> b = inexact(s.k, s.n, 2);
  Oracle oracle = make_oracle(a, b, s);

  std::size_t bytes_a = a.size() * sizeof(float);
  std::size_t bytes_b = b.size() * sizeof(float);
  std::size_t bytes_c = oracle.exact_f32.size() * sizeof(float);
  float *da = nullptr;
  float *db = nullptr;
  float *dc = nullptr;
  CHECK_CUDA(cudaMalloc(&da, bytes_a));
  CHECK_CUDA(cudaMalloc(&db, bytes_b));
  CHECK_CUDA(cudaMalloc(&dc, bytes_c));
  CHECK_CUDA(cudaMemcpy(da, a.data(), bytes_a, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(db, b.data(), bytes_b, cudaMemcpyHostToDevice));

  std::vector<float> got(oracle.exact_f32.size());

  for (const Variant &v : kVariants) {
    if (!fits(v, s.m, s.n, s.k))
      continue;

    CHECK_CUDA(cudaMemset(dc, 0, bytes_c));
    v.launch(da, db, dc, s.m, s.n, s.k);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(got.data(), dc, bytes_c, cudaMemcpyDeviceToHost));

    double used = bound_used(got, oracle, s.k);
    bool exact = std::memcmp(got.data(), oracle.exact_f32.data(), bytes_c) == 0;
    std::string checked = exact ? "bit-exact" : "within-bound";

    if ((v.bit_exact && !exact) || used > 1.0) {
      std::fprintf(stderr, "FAIL %-8s %s: %s, bound used %.3f\n", v.name,
                   shape_name(s.m, s.n, s.k).c_str(),
                   exact ? "bit-exact" : "not bit-exact", used);
      std::exit(1);
    }

    std::fprintf(stderr, "PASS %-8s %-16s %s (bound used %.4f)\n", v.name,
                 shape_name(s.m, s.n, s.k).c_str(), checked.c_str(), used);

    if (!timed)
      continue;

    Timing t = time_samples([&] {
      v.launch(da, db, dc, s.m, s.n, s.k);
      CHECK_CUDA(cudaDeviceSynchronize());
    });
    rows.push_back(json_row(std::string("cuda-") + v.name, device, s.m, s.n,
                            s.k, t, checked, used));
  }

  CHECK_CUDA(cudaFree(da));
  CHECK_CUDA(cudaFree(db));
  CHECK_CUDA(cudaFree(dc));
}

static void profile(int size) {
  Shape s{size, size, size};
  std::vector<float> a = inexact(s.m, s.k, 1);
  std::vector<float> b = inexact(s.k, s.n, 2);
  std::size_t bytes = a.size() * sizeof(float);
  float *da = nullptr;
  float *db = nullptr;
  float *dc = nullptr;
  CHECK_CUDA(cudaMalloc(&da, bytes));
  CHECK_CUDA(cudaMalloc(&db, bytes));
  CHECK_CUDA(cudaMalloc(&dc, bytes));
  CHECK_CUDA(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));

  for (const Variant &v : kVariants) {
    v.launch(da, db, dc, s.m, s.n, s.k);
    CHECK_CUDA(cudaDeviceSynchronize());
  }

  CHECK_CUDA(cudaFree(da));
  CHECK_CUDA(cudaFree(db));
  CHECK_CUDA(cudaFree(dc));
}

int main(int argc, char **argv) {
  std::string mode = argc > 1 ? argv[1] : "bench";

  if (cublasCreate(&cublas_handle()) != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "cublasCreate failed\n");
    return 1;
  }

  if (mode == "--profile") {
    profile(kProfileSize);
    return 0;
  }

  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

  std::vector<std::string> rows;
  run_shape({384, 256, 640}, false, prop.name, rows);
  run_shape({64, 128, 32}, false, prop.name, rows);

  if (mode == "--check") {
    for (int n : kBenchSizes)
      run_shape({n, n, n}, false, prop.name, rows);

    return 0;
  }

  for (int n : kBenchSizes)
    run_shape({n, n, n}, true, prop.name, rows);

  std::printf("{\n%s\n  \"benchmarks\": [\n", context_json(prop).c_str());

  for (std::size_t i = 0; i < rows.size(); ++i)
    std::printf("%s%s\n", rows[i].c_str(), i + 1 < rows.size() ? "," : "");

  std::printf("  ]\n}\n");
  return 0;
}
