#pragma once

#include "constants.h"

namespace nanodsp::cuda {

__device__ __forceinline__ float mul_add(float acc, float x, float y) {
  return __fadd_rn(acc, __fmul_rn(x, y));
}

__device__ __forceinline__ float4 load4(const float *p) {
  return *reinterpret_cast<const float4 *>(p);
}

__device__ __forceinline__ void store4(float *p, float4 v) {
  *reinterpret_cast<float4 *>(p) = v;
}

__device__ __forceinline__ void unpack4(float4 v, float *out) {
  out[0] = v.x;
  out[1] = v.y;
  out[2] = v.z;
  out[3] = v.w;
}

__global__ void naive(const float *a, const float *b, float *c, int m, int n,
                      int k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= m || col >= n)
    return;

  float acc = 0.0f;

  for (int kk = 0; kk < k; ++kk)
    acc = mul_add(acc, a[row * k + kk], b[kk * n + col]);

  c[row * n + col] = acc;
}

__global__ void tiled(const float *a, const float *b, float *c, int, int n,
                      int k) {
  __shared__ float as[kTile][kTile];
  __shared__ float bs[kTile][kTile];

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int row = blockIdx.y * kTile + ty;
  int col = blockIdx.x * kTile + tx;
  float acc = 0.0f;

  for (int t = 0; t < k; t += kTile) {
    as[ty][tx] = a[row * k + t + tx];
    bs[ty][tx] = b[(t + ty) * n + col];
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kTile; ++kk)
      acc = mul_add(acc, as[ty][kk], bs[kk][tx]);

    __syncthreads();
  }

  c[row * n + col] = acc;
}

__global__ void __launch_bounds__((kBlockM / kThreadM) * (kBlockN / kThreadN))
    blocked(const float *__restrict__ a, const float *__restrict__ b,
            float *__restrict__ c, int, int n, int k) {
  constexpr int threads = (kBlockM / kThreadM) * (kBlockN / kThreadN);

  __shared__ float as[kBlockK][kBlockM];
  __shared__ float bs[kBlockK][kBlockN];

  int tid = threadIdx.x;
  int trow = tid / (kBlockN / kThreadN) * kThreadM;
  int tcol = tid % (kBlockN / kThreadN) * kThreadN;
  const float *ablock = a + blockIdx.y * kBlockM * k;
  const float *bblock = b + blockIdx.x * kBlockN;

  float acc[kThreadM][kThreadN] = {};
  float af[kThreadM];
  float bf[kThreadN];

  for (int t = 0; t < k; t += kBlockK) {
#pragma unroll
    for (int i = tid; i < kBlockM * kBlockK; i += threads)
      as[i % kBlockK][i / kBlockK] =
          ablock[(i / kBlockK) * k + t + i % kBlockK];

#pragma unroll
    for (int i = tid; i < kBlockK * kBlockN; i += threads)
      bs[i / kBlockN][i % kBlockN] =
          bblock[(t + i / kBlockN) * n + i % kBlockN];

    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kBlockK; ++kk) {
#pragma unroll
      for (int i = 0; i < kThreadM; ++i)
        af[i] = as[kk][trow + i];

#pragma unroll
      for (int j = 0; j < kThreadN; ++j)
        bf[j] = bs[kk][tcol + j];

#pragma unroll
      for (int i = 0; i < kThreadM; ++i)
#pragma unroll
        for (int j = 0; j < kThreadN; ++j)
          acc[i][j] = mul_add(acc[i][j], af[i], bf[j]);
    }

    __syncthreads();
  }

  int crow = blockIdx.y * kBlockM + trow;
  int ccol = blockIdx.x * kBlockN + tcol;

#pragma unroll
  for (int i = 0; i < kThreadM; ++i)
#pragma unroll
    for (int j = 0; j < kThreadN; ++j)
      c[(crow + i) * n + ccol + j] = acc[i][j];
}

template <bool kInterleaved>
__global__ void __launch_bounds__((kVecM / kVecThreadM) * (kVecN / kVecThreadN))
    vec(const float *__restrict__ a, const float *__restrict__ b,
        float *__restrict__ c, int, int n, int k) {
  __shared__ __align__(16) float as[kVecK][kVecM];
  __shared__ __align__(16) float bs[kVecK][kVecN];

  int tid = threadIdx.x;
  int trow = tid / (kVecN / kVecThreadN) * kVecThreadM;
  int tx = tid % (kVecN / kVecThreadN);
  int cols[2] = {tx * kVecThreadN, tx * kVecThreadN + 4};

  if (kInterleaved) {
    cols[0] = tx * 4;
    cols[1] = kVecN / 2 + tx * 4;
  }

  int arow = tid / (kVecK / 4);
  int acol = tid % (kVecK / 4) * 4;
  int brow = tid / (kVecN / 4);
  int bcol = tid % (kVecN / 4) * 4;
  const float *ablock = a + blockIdx.y * kVecM * k;
  const float *bblock = b + blockIdx.x * kVecN;

  float acc[kVecThreadM][kVecThreadN] = {};
  float af[kVecThreadM];
  float bf[kVecThreadN];
  float staged[4];

  for (int t = 0; t < k; t += kVecK) {
    unpack4(load4(&ablock[arow * k + t + acol]), staged);

#pragma unroll
    for (int i = 0; i < 4; ++i)
      as[acol + i][arow] = staged[i];

    store4(&bs[brow][bcol], load4(&bblock[(t + brow) * n + bcol]));
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kVecK; ++kk) {
#pragma unroll
      for (int i = 0; i < kVecThreadM; i += 4)
        unpack4(load4(&as[kk][trow + i]), &af[i]);

      unpack4(load4(&bs[kk][cols[0]]), &bf[0]);
      unpack4(load4(&bs[kk][cols[1]]), &bf[4]);

#pragma unroll
      for (int i = 0; i < kVecThreadM; ++i)
#pragma unroll
        for (int j = 0; j < kVecThreadN; ++j)
          acc[i][j] = mul_add(acc[i][j], af[i], bf[j]);
    }

    __syncthreads();
  }

  int crow = blockIdx.y * kVecM + trow;
  int ccol = blockIdx.x * kVecN;

#pragma unroll
  for (int i = 0; i < kVecThreadM; ++i)
#pragma unroll
    for (int g = 0; g < 2; ++g)
      store4(&c[(crow + i) * n + ccol + cols[g]],
             make_float4(acc[i][4 * g], acc[i][4 * g + 1], acc[i][4 * g + 2],
                         acc[i][4 * g + 3]));
}

}
