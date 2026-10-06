# GPU Results

Measurements of the Mojo GPU kernels ([`mojo-gpu.md`](mojo-gpu.md)). Like
[`03-results.md`](03-results.md), these are one machine's snapshot, not a
general ranking.

**Status:** Apple M2 GPU measured. **NVIDIA T4 and cuBLAS: pending** (the
run needs a cloud GPU; the steps are in [`mojo-gpu.md`](mojo-gpu.md#nvidia-on-colab-or-kaggle-t4)).

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

## NVIDIA T4 — pending

| op and shape | Mojo naive | Mojo tiled | Mojo blocked | cuBLAS |
| --- | ---: | ---: | ---: | ---: |
| matmul 256³ – 2048³ | pending | pending | pending | pending |
| conv2d (both shapes) | pending | — | — | — |

To fill in: run the steps in `mojo-gpu.md`, save the files as
`benchmarks/results/gpu-t4.json` and `cublas-t4.json`, and paste the output
of `python3 scripts/gpu_table.py`. Report for each size the Mojo blocked
kernel as a percentage of cuBLAS, and cuBLAS's `bound_used`: cuBLAS is
checked within the reordering bound, not bit for bit.
