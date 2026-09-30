// Scalar C++ reference for the dsp ops.
//
// The oracle side of the differential tests and the baseline for
// benchmarks/. Written as the plainest possible loop nest: no SIMD, no
// tiling, reduction order kh -> kw -> c (conv) and k (matmul), matching
// linalg.generic after -convert-linalg-to-loops. Build with
// -ffp-contract=off so `acc += a * b` is not fused into an FMA; otherwise the
// rounding differs from the unfused Mojo kernels and the MLIR pipeline.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace nanodsp::ref {

struct Tensor {
  std::vector<std::size_t> shape;
  std::vector<float> data;

  explicit Tensor(std::vector<std::size_t> s, float fill = 0.0f)
      : shape(std::move(s)), data(numel(shape), fill) {}
  Tensor(std::vector<std::size_t> s, std::vector<float> values)
      : shape(std::move(s)), data(std::move(values)) {
    if (data.size() != numel(shape))
      throw std::invalid_argument("Tensor: value count does not match shape");
  }

  static std::size_t numel(const std::vector<std::size_t> &s) {
    std::size_t n = 1;
    for (std::size_t d : s)
      n *= d;
    return n;
  }
};

inline Tensor add(const Tensor &a, const Tensor &b) {
  if (a.shape != b.shape)
    throw std::invalid_argument("add: operand shapes differ");
  Tensor r(a.shape);
  for (std::size_t i = 0; i < a.data.size(); ++i)
    r.data[i] = a.data[i] + b.data[i];
  return r;
}

// NaN-propagating, like arith.maximumf: NaN < 0 is false, so NaN passes.
inline Tensor relu(const Tensor &a) {
  Tensor r(a.shape);
  for (std::size_t i = 0; i < a.data.size(); ++i)
    r.data[i] = a.data[i] < 0.0f ? 0.0f : a.data[i];
  return r;
}

inline Tensor matmul(const Tensor &a, const Tensor &b) {
  if (a.shape.size() != 2 || b.shape.size() != 2 || a.shape[1] != b.shape[0])
    throw std::invalid_argument("matmul: expected (MxK) * (KxN)");
  const std::size_t m = a.shape[0], k = a.shape[1], n = b.shape[1];
  Tensor r({m, n});
  for (std::size_t i = 0; i < m; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (std::size_t kk = 0; kk < k; ++kk)
        acc += a.data[i * k + kk] * b.data[kk * n + j];
      r.data[i * n + j] = acc;
    }
  return r;
}

// NHWC input x HWCF filter -> NHWF output, 'valid' padding, no kernel flip.
inline Tensor conv2d(const Tensor &in, const Tensor &f, std::size_t sh = 1,
                     std::size_t sw = 1, std::size_t dh = 1,
                     std::size_t dw = 1) {
  if (in.shape.size() != 4 || f.shape.size() != 4 || in.shape[3] != f.shape[2])
    throw std::invalid_argument("conv2d: expected NHWC x HWCF");
  const std::size_t nb = in.shape[0], h = in.shape[1], w = in.shape[2],
                    c = in.shape[3];
  const std::size_t kh = f.shape[0], kw = f.shape[1], nf = f.shape[3];
  if (h < (kh - 1) * dh + 1 || w < (kw - 1) * dw + 1)
    throw std::invalid_argument("conv2d: filter is larger than the input");
  const std::size_t oh = (h - (kh - 1) * dh - 1) / sh + 1;
  const std::size_t ow = (w - (kw - 1) * dw - 1) / sw + 1;
  Tensor r({nb, oh, ow, nf});
  for (std::size_t n = 0; n < nb; ++n)
    for (std::size_t y = 0; y < oh; ++y)
      for (std::size_t x = 0; x < ow; ++x)
        for (std::size_t fo = 0; fo < nf; ++fo) {
          float acc = 0.0f;
          for (std::size_t ky = 0; ky < kh; ++ky)
            for (std::size_t kx = 0; kx < kw; ++kx)
              for (std::size_t ch = 0; ch < c; ++ch) {
                const std::size_t iy = y * sh + ky * dh, ix = x * sw + kx * dw;
                acc += in.data[((n * h + iy) * w + ix) * c + ch] *
                       f.data[((ky * kw + kx) * c + ch) * nf + fo];
              }
          r.data[((n * oh + y) * ow + x) * nf + fo] = acc;
        }
  return r;
}

// ---------------------------------------------------------------------------
// Quantized int8 (dsp.qmatmul)
// ---------------------------------------------------------------------------

struct QTensor {
  std::vector<std::size_t> shape;
  std::vector<std::int8_t> data;

  QTensor(std::vector<std::size_t> s, std::vector<std::int8_t> values)
      : shape(std::move(s)), data(std::move(values)) {
    if (data.size() != Tensor::numel(shape))
      throw std::invalid_argument("QTensor: value count does not match shape");
  }
};

// Per-tensor quantization of a qmatmul; see dsp.qmatmul in DSPOps.td.
struct QuantParams {
  std::int32_t lhs_zp, rhs_zp;
  std::int32_t multiplier; // Q0.31, normalized to [2^30, 2^31)
  std::int32_t shift;      // right shift, [0, 31]
  std::int32_t out_zp;
};

// clamp(out_zp + ((acc * multiplier + 2^(s-1)) >> s), -128, 127), s = 31 +
// shift. >> on a negative int64 is an arithmetic shift (defined since
// C++20), so this rounds half up: -100.5 -> -100.
inline std::int8_t requantize(std::int32_t acc, const QuantParams &q) {
  const int s = 31 + q.shift;
  const std::int64_t scaled =
      (std::int64_t{acc} * q.multiplier + (std::int64_t{1} << (s - 1))) >> s;
  return static_cast<std::int8_t>(
      std::clamp<std::int64_t>(q.out_zp + scaled, -128, 127));
}

inline QTensor qmatmul(const QTensor &a, const QTensor &b,
                       const QuantParams &q) {
  if (a.shape.size() != 2 || b.shape.size() != 2 || a.shape[1] != b.shape[0])
    throw std::invalid_argument("qmatmul: expected (MxK) * (KxN)");
  const std::size_t m = a.shape[0], k = a.shape[1], n = b.shape[1];
  QTensor r({m, n}, std::vector<std::int8_t>(m * n));
  for (std::size_t i = 0; i < m; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      std::int32_t acc = 0;
      for (std::size_t kk = 0; kk < k; ++kk)
        acc += (std::int32_t{a.data[i * k + kk]} - q.lhs_zp) *
               (std::int32_t{b.data[kk * n + j]} - q.rhs_zp);
      r.data[i * n + j] = requantize(acc, q);
    }
  return r;
}

} // namespace nanodsp::ref
