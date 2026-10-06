#include "../../reference/memref.h"
#include "../../reference/nanodsp_ref.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace nanodsp::ref;
using nanodsp::MemRef;
using nanodsp::wrap;

extern "C" {
void *_mlir_memref_to_llvm_alloc(std::uint64_t size) {
  return std::malloc(static_cast<std::size_t>(size));
}

void *_mlir_memref_to_llvm_aligned_alloc(std::uint64_t alignment,
                                         std::uint64_t size) {
  return std::aligned_alloc(static_cast<std::size_t>(alignment),
                            static_cast<std::size_t>(size));
}

void _mlir_memref_to_llvm_free(void *p) { std::free(p); }

void _mlir_ciface_matmul(MemRef<float, 2> *, MemRef<float, 2> *,
                         MemRef<float, 2> *);
void _mlir_ciface_conv2d(MemRef<float, 4> *, MemRef<float, 4> *,
                         MemRef<float, 4> *);
void _mlir_ciface_add_relu(MemRef<float, 2> *, MemRef<float, 2> *,
                           MemRef<float, 2> *);
void _mlir_ciface_qmatmul_golden(MemRef<std::int8_t, 2> *,
                                 MemRef<std::int8_t, 2> *,
                                 MemRef<std::int8_t, 2> *);
void _mlir_ciface_qmatmul(MemRef<std::int8_t, 2> *, MemRef<std::int8_t, 2> *,
                          MemRef<std::int8_t, 2> *);
#ifdef NANODSP_LOCAL_KERNELS
void _mlir_ciface_matmul_local(MemRef<float, 2> *, MemRef<float, 2> *,
                               MemRef<float, 2> *);
void _mlir_ciface_qmatmul_local(MemRef<std::int8_t, 2> *,
                                MemRef<std::int8_t, 2> *,
                                MemRef<std::int8_t, 2> *);
#endif
}

static float fill(std::size_t i, std::size_t j, std::size_t k = 0,
                  std::size_t l = 0) {
  return float((7 * i + 3 * j + 5 * k + 11 * l) % 13) * 0.37f - 1.9f;
}

static int failures = 0;

template <typename T>
static void compare(const char *name, const T *actual,
                    const std::vector<T> &expected) {
  std::size_t bad = 0;

  for (std::size_t i = 0; i < expected.size(); ++i)
    bad += std::memcmp(&actual[i], &expected[i], sizeof(T)) != 0;

  std::printf("%s %-16s %zu/%zu values bit-identical\n", bad ? "FAIL" : "PASS",
              name, expected.size() - bad, expected.size());
  failures += bad != 0;
}

int main() {
  {
    Tensor a({64, 96}), b({96, 48});

    for (std::size_t i = 0; i < 64; ++i)
      for (std::size_t j = 0; j < 96; ++j)
        a.data[i * 96 + j] = fill(i, j);

    for (std::size_t i = 0; i < 96; ++i)
      for (std::size_t j = 0; j < 48; ++j)
        b.data[i * 48 + j] = fill(j, i);

    auto ma = wrap<float, 2>(a.data, a.shape);
    auto mb = wrap<float, 2>(b.data, b.shape);
    MemRef<float, 2> out;
    _mlir_ciface_matmul(&out, &ma, &mb);
    compare("matmul", out.aligned, matmul(a, b).data);
  }

  {
    Tensor in({1, 12, 12, 3}), f({3, 3, 3, 8});

    for (std::size_t h = 0; h < 12; ++h)
      for (std::size_t w = 0; w < 12; ++w)
        for (std::size_t c = 0; c < 3; ++c)
          in.data[(h * 12 + w) * 3 + c] = fill(0, h, w, c);

    for (std::size_t h = 0; h < 3; ++h)
      for (std::size_t w = 0; w < 3; ++w)
        for (std::size_t c = 0; c < 3; ++c)
          for (std::size_t o = 0; o < 8; ++o)
            f.data[((h * 3 + w) * 3 + c) * 8 + o] = fill(o, c, h, w);

    auto mi = wrap<float, 4>(in.data, in.shape);
    auto mf = wrap<float, 4>(f.data, f.shape);
    MemRef<float, 4> out;
    _mlir_ciface_conv2d(&out, &mi, &mf);
    compare("conv2d", out.aligned, conv2d(in, f).data);
  }

  {
    Tensor a({6, 40}), b({6, 40});

    for (std::size_t i = 0; i < 6; ++i)
      for (std::size_t j = 0; j < 40; ++j) {
        a.data[i * 40 + j] = fill(i, j);
        b.data[i * 40 + j] = fill(j, i + 5);
      }
    auto ma = wrap<float, 2>(a.data, a.shape);
    auto mb = wrap<float, 2>(b.data, b.shape);
    MemRef<float, 2> out;
    _mlir_ciface_add_relu(&out, &ma, &mb);
    compare("add+relu", out.aligned, relu(add(a, b)).data);
  }

  {
    QTensor a({2, 3}, {-128, 0, 127, 10, -7, 50});
    QTensor b({3, 4}, {1, -3, 127, -128, 4, 2, 0, 9, -1, 5, -55, 30});
    auto ma = wrap<std::int8_t, 2>(a.data, a.shape);
    auto mb = wrap<std::int8_t, 2>(b.data, b.shape);
    MemRef<std::int8_t, 2> out;
    _mlir_ciface_qmatmul_golden(&out, &ma, &mb);
    compare("qmatmul golden", out.aligned,
            std::vector<std::int8_t>{-23, 57, -128, 127, -4, 13, -105, 27});
  }

  {
    std::vector<std::int8_t> av(32 * 128), bv(128 * 64);

    for (std::size_t i = 0; i < av.size(); ++i)
      av[i] = std::int8_t(int((i * 73 + 11) % 256) - 128);

    for (std::size_t i = 0; i < bv.size(); ++i)
      bv[i] = std::int8_t(int((i * 151 + 7) % 256) - 128);

    QTensor a({32, 128}, av), b({128, 64}, bv);
    auto ma = wrap<std::int8_t, 2>(a.data, a.shape);
    auto mb = wrap<std::int8_t, 2>(b.data, b.shape);
    MemRef<std::int8_t, 2> out;
    _mlir_ciface_qmatmul(&out, &ma, &mb);
    compare("qmatmul", out.aligned,
            qmatmul(a, b, {-7, 12, 1276901417, 9, 4}).data);
  }
#ifdef NANODSP_LOCAL_KERNELS
  {
    Tensor a({128, 256}), b({256, 128});

    for (std::size_t i = 0; i < 128; ++i)
      for (std::size_t j = 0; j < 256; ++j)
        a.data[i * 256 + j] = fill(i, j);

    for (std::size_t i = 0; i < 256; ++i)
      for (std::size_t j = 0; j < 128; ++j)
        b.data[i * 128 + j] = fill(j, i);

    auto ma = wrap<float, 2>(a.data, a.shape);
    auto mb = wrap<float, 2>(b.data, b.shape);
    MemRef<float, 2> out;
    _mlir_ciface_matmul_local(&out, &ma, &mb);
    compare("matmul local", out.aligned, matmul(a, b).data);
  }

  {
    std::vector<std::int8_t> av(128 * 1024), bv(1024 * 128);

    for (std::size_t i = 0; i < av.size(); ++i)
      av[i] = std::int8_t(int((i * 73 + 11) % 256) - 128);

    for (std::size_t i = 0; i < bv.size(); ++i)
      bv[i] = std::int8_t(int((i * 151 + 7) % 256) - 128);

    QTensor a({128, 1024}, av), b({1024, 128}, bv);
    auto ma = wrap<std::int8_t, 2>(a.data, a.shape);
    auto mb = wrap<std::int8_t, 2>(b.data, b.shape);
    MemRef<std::int8_t, 2> out;
    _mlir_ciface_qmatmul_local(&out, &ma, &mb);
    compare("qmatmul local", out.aligned,
            qmatmul(a, b, {-7, 12, 1276901417, 9, 4}).data);
  }
#endif
  return failures == 0 ? 0 : 1;
}
