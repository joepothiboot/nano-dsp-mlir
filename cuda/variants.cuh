#pragma once

#include "kernels.cuh"

#include <cublas_v2.h>

namespace nanodsp::cuda {

struct Variant {
  const char *name;
  int divisor_m;
  int divisor_n;
  int divisor_k;
  bool bit_exact;
  void (*launch)(const float *a, const float *b, float *c, int m, int n, int k);
};

inline cublasHandle_t &cublas_handle() {
  static cublasHandle_t handle = nullptr;
  return handle;
}

inline void launch_naive(const float *a, const float *b, float *c, int m, int n,
                         int k) {
  dim3 block(kTile, kTile);
  dim3 grid((n + kTile - 1) / kTile, (m + kTile - 1) / kTile);
  naive<<<grid, block>>>(a, b, c, m, n, k);
}

inline void launch_tiled(const float *a, const float *b, float *c, int m, int n,
                         int k) {
  dim3 block(kTile, kTile);
  dim3 grid(n / kTile, m / kTile);
  tiled<<<grid, block>>>(a, b, c, m, n, k);
}

inline void launch_blocked(const float *a, const float *b, float *c, int m,
                           int n, int k) {
  int threads = (kBlockM / kThreadM) * (kBlockN / kThreadN);
  dim3 grid(n / kBlockN, m / kBlockM);
  blocked<<<grid, threads>>>(a, b, c, m, n, k);
}

template <bool kInterleaved>
inline void launch_vec(const float *a, const float *b, float *c, int m, int n,
                       int k) {
  int threads = (kVecM / kVecThreadM) * (kVecN / kVecThreadN);
  dim3 grid(n / kVecN, m / kVecM);
  vec<kInterleaved><<<grid, threads>>>(a, b, c, m, n, k);
}

inline void launch_cublas(const float *a, const float *b, float *c, int m,
                          int n, int k) {
  const float one = 1.0f;
  const float zero = 0.0f;
  cublasSgemm(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one, b, n, a,
              k, &zero, c, n);
}

inline constexpr Variant kVariants[] = {
    {"naive", 1, 1, 1, true, launch_naive},
    {"tiled", kTile, kTile, kTile, true, launch_tiled},
    {"blocked", kBlockM, kBlockN, kBlockK, true, launch_blocked},
    {"vec", kVecM, kVecN, kVecK, true, launch_vec<false>},
    {"vec-interleaved", kVecM, kVecN, kVecK, true, launch_vec<true>},
    {"cublas", 1, 1, 1, false, launch_cublas},
};

inline bool fits(const Variant &v, int m, int n, int k) {
  return m % v.divisor_m == 0 && n % v.divisor_n == 0 && k % v.divisor_k == 0;
}

}
