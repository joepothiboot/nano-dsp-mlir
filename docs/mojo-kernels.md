# Mojo kernels 🔥

`mojo/nanodsp/` implements the `dsp` ops as a small Mojo library. It
exists for two reasons:

1. 🔍 **A second oracle for the compiler.** The MLIR pipeline and the Mojo
   kernels are written independently from the same op spec
   (`DSPOps.td`). If they disagree on a golden value, one of them is wrong.
2. 🏁 **A hand-written baseline.** Once Stage 3 lands, the benchmark puts
   compiler-generated code head to head with a hand-written SIMD kernel and a
   scalar C++ loop nest on the same shapes.

## 🧩 API

```mojo
from nanodsp import Tensor, add, relu, matmul, conv2d, qmatmul, QuantParams

var a = Tensor[DType.float32]([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
var b = Tensor[DType.float32]([3, 2], fill=1.0)
var c = matmul(a, b)          # raises on a shape mismatch
```

- `Tensor[dtype]` is an owned, contiguous, row-major buffer plus a shape.
  There are no strides, views or broadcasting, because the dialect has none.
- Kernels are generic over `dtype`. The SIMD width comes from
  `simd_width_of[dtype]()` at compile time, so the same source compiles to
  NEON on Apple silicon and AVX on x86.
- Semantics match the dialect exactly: `add` rejects mismatched shapes instead
  of broadcasting, `relu` propagates NaN (`arith.maximumf`), and `conv2d` is
  NHWC x HWCF cross-correlation with 'valid' padding and optional
  strides/dilations.

## ⚡ SIMD strategy

Each kernel vectorizes the innermost contiguous dimension and handles the
remainder with a scalar tail:

| Kernel    | Vectorized over | Inner step                               |
| --------- | --------------- | ---------------------------------------- |
| add, relu | flat index      | load, op, store                          |
| matmul    | N (columns)     | `c[i, :] += a[i, k] * b[k, :]` (i-k-j)   |
| conv2d    | F (filters)     | `out[n, y, x, :] += in[...] * f[..., :]` |

matmul and conv2d share one helper, `_axpy`. It keeps multiply and add
**unfused** and preserves the reduction order of the naive loop nest (k for
matmul; kh, kw, c for conv), which is also the order `linalg.generic` uses
after `-convert-linalg-to-loops`. That means results can be compared for
exact equality, not just within a tolerance. The C++ reference is built with
`-ffp-contract=off` so the compiler doesn't fuse `acc += a * b` into an FMA.

📊 Untiled throughput on an Apple M2 (`pixi run bench`, best of 3-5 runs, one
core):

| Shape                       | Time    | GFLOP/s |
| --------------------------- | ------- | ------- |
| matmul 256 x 256 x 256      | 1.51 ms | 22.2    |
| matmul 512 x 512 x 512      | 14.4 ms | 18.6    |
| conv2d 56 x 56 x 64 -> 64   | 13.6 ms | 15.8    |
| conv2d 28 x 28 x 128 -> 128 | 9.7 ms  | 20.6    |

Throughput drops at 512, most likely because a row of `b` stops staying in
L1 across the k loop; Stage 3 tiling is meant to confirm and fix that.

This is untiled on purpose. Cache tiling is what Stage 3 derives from the
machine model, and the untiled kernel is the baseline it has to beat.

## 🧪 Tests

`mojo/tests/test_kernels.mojo` has two kinds of test:

- **Golden:** the same inputs and expected outputs as `test/Integration/`
  (and `reference/test_reference.cpp`).
- **Differential:** SIMD kernel vs. a naive loop nest on odd shapes (such as
  F = 11 or N = 19), so the scalar tail runs as well as the SIMD body. Inputs
  are small integers, so every partial sum is exact in f32.

Error paths (broadcast attempts, inner-dimension mismatch, channel mismatch)
are checked with `assert_raises`.

## 🔜 Next

- Tiled `matmul` with the tile sizes from `docs/02-tiling-model.md`, so the
  hand-written and compiler-derived schedules can be compared directly.
- `parallelize` over output rows.
- A benchmark table: MLIR (untiled / scheduled), Mojo, and C++ on the same
  shapes.
