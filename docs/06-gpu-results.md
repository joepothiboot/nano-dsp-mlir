# GPU Results

Measurements of the Mojo GPU kernels ([`mojo-gpu.md`](mojo-gpu.md)). Like
[`03-results.md`](03-results.md), these are one machine's snapshot, not a
general ranking.

**Status:** Apple M2 GPU and NVIDIA Tesla T4 (free Colab) measured, with
cuBLAS as the T4 baseline. Every Mojo row on both GPUs is bit-exact.

## Method

- `pixi run bench-gpu` (`benchmarks/bench_gpu.mojo`), built with
  `--fp-mode contract=off`. Raw results: `benchmarks/results/gpu-m2.json`.
- Before timing, every kernel's output is compared **bit for bit** with the
  CPU kernel on the same inexact inputs. A mismatch stops the run. Every row
  below passed.
- Same timing policy as the CPU benchmarks: one warmup, ten samples of at
  least 50 ms. One call is one launch plus `synchronize`, with operands
  already on the device; copies are not timed.
- Cells: median time ± sample standard deviation, then throughput from the
  best sample (as in the CPU harness).
- Apple M2, 10-core GPU, macOS 27.0.1, Mojo 1.1.0 with `max-core` 26.6.0.
- NVIDIA Tesla T4 (Turing, compute capability 7.5), driver 580.82.07, on a
  free Colab runtime, same Mojo; cuBLAS through CuPy (`benchmarks/bench_cublas.py`), TF32 off.
  Raw results: `benchmarks/results/gpu-t4.json`, `cublas-t4.json`. A shared
  cloud GPU is noisier than a local one; the spreads below show where.

## Apple M2 GPU

| op and shape | naive | tiled | blocked |
| --- | ---: | ---: | ---: |
| matmul 256³ | 795.5 µs ± 12.6 µs, 44 GFLOP/s | 586.2 µs ± 27.3 µs, 62 GFLOP/s | 351.5 µs ± 25.7 µs, 110 GFLOP/s |
| matmul 512³ | 3.86 ms ± 20.5 µs, 70 GFLOP/s | 2.70 ms ± 8.9 µs, 100 GFLOP/s | 893.0 µs ± 35.7 µs, 333 GFLOP/s |
| matmul 1024³ | 28.27 ms ± 53.3 µs, 76 GFLOP/s | 18.86 ms ± 55.2 µs, 114 GFLOP/s | 4.29 ms ± 28.5 µs, 509 GFLOP/s |
| matmul 2048³ | 223.80 ms ± 452.7 µs, 77 GFLOP/s | 148.78 ms ± 278.7 µs, 116 GFLOP/s | 31.36 ms ± 32.5 µs, 549 GFLOP/s |

| op and shape | one thread per output |
| --- | ---: |
| conv2d 56×56×64 → 64 | 3.68 ms ± 29.6 µs, 60 GFLOP/s |
| conv2d 28×28×128 → 128 | 3.22 ms ± 13.9 µs, 62 GFLOP/s |

### Against the CPU (same machine, `03-results.md`)

| op and shape | C++ reference | MLIR scheduled | Mojo CPU, best | Mojo GPU, best |
| --- | ---: | ---: | ---: | ---: |
| matmul 512³ | 109.06 ms | 7.99 ms | 5.75 ms (tiled) | 0.89 ms (blocked) |
| conv2d 56×56×64 → 64 | 89.29 ms | 10.76 ms | 14.84 ms | 3.68 ms |
| conv2d 28×28×128 → 128 | 87.85 ms | 25.62 ms | 13.11 ms | 3.22 ms |

All medians; all bit-exact against the reference except scheduled MLIR
conv2d, which is within the reordering bound (`03-results.md`).

### What bit-exactness costs here

The same benchmark built with Mojo's default `contract=fast`, which fuses
multiply-add, ran at 544 GFLOP/s for blocked 2048³ against 549 unfused: no
difference beyond noise at any size. On the M2 these kernels are limited by
memory traffic, not arithmetic, so keeping multiply and add separate costs
nothing measurable. (The fused build's outputs were not checked against the
reference; it was a timing run only.) That may differ on a GPU whose kernels
here reach its arithmetic limit.

### Reading it

- The register-blocked kernel is 7× the naive one at 2048³; shared-memory
  tiling alone gives 1.5×. Reuse in registers is what matters on this GPU.
- 256³ is dominated by launch and synchronize latency (all variants are
  0.35–0.8 ms for 34 MFLOP).
- No Apple GPU library baseline is included (MPS was not measured), so these
  numbers show the kernels' progression, not how close they are to the best
  possible on the M2.

## NVIDIA Tesla T4

`pixi run test-gpu` passed on the T4 (all 8 tests, including the odd shapes
and the FMA negative control), and every benchmark row below was
bit-checked against the CPU kernel before timing: the NVIDIA path keeps
multiply and add separate under `contract=off` too.

| op and shape | naive | tiled | blocked | cuBLAS | blocked / cuBLAS |
| --- | ---: | ---: | ---: | ---: | ---: |
| matmul 256³ | 74.9 µs ± 27.8 µs, 455 GFLOP/s | 62.7 µs ± 0.3 µs, 540 GFLOP/s | 41.9 µs ± 0.3 µs, 808 GFLOP/s | 53.2 µs ± 2.6 µs, 642 GFLOP/s | 126% |
| matmul 512³ | 516.1 µs ± 9.4 µs, 524 GFLOP/s | 421.4 µs ± 5.2 µs, 644 GFLOP/s | 150.3 µs ± 0.9 µs, 1795 GFLOP/s | 87.6 µs ± 2.9 µs, 3109 GFLOP/s | 58% |
| matmul 1024³ | 4.66 ms ± 54.4 µs, 466 GFLOP/s | 3.69 ms ± 47.1 µs, 589 GFLOP/s | 1.11 ms ± 6.8 µs, 1953 GFLOP/s | 611.7 µs ± 7.6 µs, 3529 GFLOP/s | 55% |
| matmul 2048³ | 42.82 ms ± 4.31 ms, 405 GFLOP/s | 35.20 ms ± 594.3 µs, 492 GFLOP/s | 9.13 ms ± 177.0 µs, 1892 GFLOP/s | 4.44 ms ± 29.6 µs, 3881 GFLOP/s | 49% |

| op and shape | one thread per output |
| --- | ---: |
| conv2d 56×56×64 → 64 | 569.8 µs ± 18.9 µs, 380 GFLOP/s |
| conv2d 28×28×128 → 128 | 623.1 µs ± 36.0 µs, 326 GFLOP/s |

`blocked / cuBLAS` is the throughput ratio from the best samples; the ratio
of medians is the same to the percent.

### cuBLAS correctness

cuBLAS sums in its own order and may fuse multiply-add, so it is checked
against a float64 product within the reordering bound, not bit for bit
(`bench_cublas.py`). The largest fraction of the bound any output used:

| size | 256³ | 512³ | 1024³ | 2048³ |
| --- | ---: | ---: | ---: | ---: |
| bound used | 0.25% | 0.25% | 0.05% | 0.07% |

A dropped or wrong product would use more than 100%.

### Reading it

- The register-blocked Mojo kernel reaches about half of cuBLAS from 512³
  up (49–58%), while keeping the reference's summation order and unfused
  multiply-add. cuBLAS is free to do neither. How much of the gap that
  discipline costs on the T4 was not measured: unlike on the M2, no fused
  build was timed here.
- At 256³ the Mojo kernel is faster than cuBLAS (126%). At that size both are
  dominated by launch and synchronize latency, and the CuPy call adds Python
  overhead the Mojo launch doesn't have, so this says little about the
  kernels themselves.
- From 512³ up, register blocking is 3.4–4.7× the naive kernel on the T4
  (4.7–7.1× on the M2); shared-memory tiling alone gives 1.2–1.3× (1.4–1.5×
  on the M2).
- The T4 runs the blocked matmul 3.4–5.4× faster than the M2 GPU from 512³
  up (1892 against 549 GFLOP/s at 2048³), and conv2d 6.5× and 5.2× faster
  by median time.
- Naive 2048³ has a large spread (± 4.31 ms, 10% of its median) while its
  best sample (42.4 ms) is close to the median: occasional slow samples,
  most likely from sharing the cloud GPU.
