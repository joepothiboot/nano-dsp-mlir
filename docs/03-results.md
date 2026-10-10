# CPU Performance Results

A same-machine comparison of the scheduled MLIR kernels, the Mojo kernels and
the scalar C++ reference, against ceilings measured on the same core. It is a
local snapshot, not a cross-machine claim.

## Environment and method

- Apple M2 (8 GB), macOS 27.0.1, arm64, single-threaded. Commit `a54dab3`.
- Homebrew LLVM 23.1.1; Mojo 1.1.0; MLIR schedule target `host-neon`.
- MLIR and C++ were measured with `pixi run bench-mlir`; Mojo with
  `pixi run bench`. Both use one warmup, ten samples, a 50 ms minimum sample.
  Raw results: `benchmarks/results/cpu-m2.json` and
  `benchmarks/results/mojo-cpu-m2.txt`.
- Shapes and deterministic input patterns match across harnesses. C++, MLIR,
  ordinary Mojo kernels, and allocation-inclusive tiled Mojo include output
  allocation. Reused-output tiled Mojo is reported separately.
- The C++ harness checks every MLIR output bit for bit against its scalar
  reference before timing; all rows passed, conv2d included. Mojo's
  correctness is checked by `pixi run test-mojo` separately; the Mojo
  benchmark itself does not verify each timed output.
- Taken on 2026-10-10 with background load around 4 on the 8-core machine. Two
  back-to-back MLIR/C++ runs agreed within 3%, and every row's median is within
  1% of its best sample, so the load did not reach the measured core.

## Ceilings

Measured by `benchmarks/ceilings.cpp` on the same core (see
[`../benchmarks/README.md`](../benchmarks/README.md#-ceilings)):

| Ceiling                      |         Value |
| ---------------------------- | ------------: |
| NEON unfused multiply + add  |  54.8 GFLOP/s |
| NEON fused multiply-add      | 111.0 GFLOP/s |
| STREAM-style triad bandwidth |     66.4 GB/s |

The kernels keep multiply and add separate to stay bit-exact, so the unfused
figure is the compute ceiling that applies. Percentages below are best-sample
throughput over that ceiling.

## Results

Cells show median per-call time ± sample standard deviation from one run.

| Operation and shape    |      C++ reference |       MLIR untiled |   MLIR scheduled |          Mojo API | Mojo tiled, allocating | Mojo tiled, reused |
| ---------------------- | -----------------: | -----------------: | ---------------: | ----------------: | ---------------------: | -----------------: |
| matmul 64³             |   113.66 ± 0.89 µs |   119.50 ± 0.68 µs |  14.18 ± 0.07 µs |   29.35 ± 0.15 µs |                      — |                  — |
| matmul 128³            |   1.159 ± 0.004 ms |   1.294 ± 0.006 ms | 114.21 ± 0.97 µs |  208.87 ± 0.67 µs |                      — |                  — |
| matmul 256³            |  11.913 ± 0.068 ms |  13.160 ± 0.045 ms | 919.41 ± 2.66 µs |  1.619 ± 0.067 ms |       715.73 ± 3.37 µs |   710.79 ± 4.12 µs |
| matmul 512³            | 105.197 ± 0.536 ms | 112.079 ± 0.226 ms | 7.586 ± 0.034 ms | 14.504 ± 0.895 ms |       5.653 ± 0.025 ms |   5.652 ± 0.022 ms |
| conv2d 56×56×64 → 64   |  86.772 ± 0.276 ms |  89.624 ± 0.255 ms | 8.805 ± 0.018 ms | 14.690 ± 0.079 ms |                      — |                  — |
| conv2d 28×28×128 → 128 |  84.316 ± 0.542 ms |  85.842 ± 0.213 ms | 5.524 ± 0.029 ms | 12.751 ± 0.091 ms |                      — |                  — |
| qmatmul 256³           |   8.726 ± 0.044 ms |   8.064 ± 0.019 ms | 820.55 ± 4.31 µs |  1.842 ± 0.009 ms |                      — |                  — |

Share of the unfused ceiling (54.8 GFLOP/s), f32 rows only:

| Operation and shape    | MLIR scheduled | Mojo API | Mojo tiled |
| ---------------------- | -------------: | -------: | ---------: |
| matmul 64³             |            68% |      33% |          — |
| matmul 128³            |            67% |      37% |          — |
| matmul 256³            |            67% |      38% |        86% |
| matmul 512³            |            65% |      34% |        87% |
| conv2d 56×56×64 → 64   |            45% |      27% |          — |
| conv2d 28×28×128 → 128 |            66% |      29% |          — |

qmatmul is integer work, so no ceiling here applies to it; scheduled MLIR
reaches 41 GOP/s.

## Reading it

- Scheduled MLIR is 13.9× faster than the C++ reference at matmul 512³ by
  median, and steady at 65–68% of the ceiling from 64³ to 512³. Untiled MLIR
  is close to the reference.
- The 4×16 tiled Mojo kernel reaches 87% of the ceiling and is 1.34× faster
  than scheduled MLIR at 512³. The gap points at the MLIR schedule's known
  limits: the accumulator tile is reloaded on every `k` step and register
  tiles nest inside the L1 reduction loops (README, known limitations).
- Scheduled MLIR beats the ordinary Mojo kernels on both conv2d shapes (1.67×
  and 2.31×) and on qmatmul (2.24×).
- conv2d 56×56×64 → 64 sits at 45% against 66% for the 28×28 shape; its
  schedule is the weakest of the f32 kernels here.
- Allocating and reusing the output make no measurable difference for tiled
  Mojo at these sizes.

## Interpretation and limits

- The Mojo `matmul_tiled` tile size is fixed at 4×16; no Mojo tile sweep was
  performed. The MLIR cache-tile sweep is in
  [`02-tiling-model.md`](02-tiling-model.md).
- Ordinary Mojo matmul and tiled matmul use different loop strategies. These
  measurements compare implementations; they do not isolate compiler quality.
- Ten samples are a useful warning signal, not a confidence interval.
- The C++/MLIR harness requests interactive QoS on macOS; the Mojo harness
  does not set an explicit QoS class. macOS offers no hard CPU affinity, so
  this is a remaining scheduling difference.
- These are call timings, not a complete model or application pipeline.
- No GPU is involved. These results are for the Apple M2 CPU's NEON path;
  GPU measurements are in [`06-gpu-results.md`](06-gpu-results.md).

Detailed command options, correctness checks, and roofline methodology are in
[`../benchmarks/README.md`](../benchmarks/README.md).
