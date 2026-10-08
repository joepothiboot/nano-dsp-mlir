# Initial Performance Results

This is a first same-machine comparison of the Mojo, MLIR, and scalar C++
implementations. It is a local snapshot, not a cross-machine claim or a
completed tile-size sweep.

## Environment and method

- Apple M2, macOS 27.0.1, arm64, single-threaded.
- Homebrew LLVM 23.1.1; Mojo 1.1.0; MLIR schedule target `host-neon`.
- MLIR and C++ were measured with `pixi run bench-mlir`; Mojo with
  `pixi run bench`. Both use one warmup, ten samples, a 50 ms minimum sample,
  and report the minimum per-call time.
- Shapes and deterministic input patterns match across harnesses. C++, MLIR,
  ordinary Mojo kernels, and allocation-inclusive tiled Mojo include output
  allocation. Reused-output tiled Mojo is reported separately.
- The C++ harness checks MLIR outputs against its scalar reference before
  timing. Mojo's correctness is checked by `pixi run test-mojo` separately;
  the Mojo benchmark itself does not verify each timed output.
- The numbers below were taken before the cache tile stopped blocking
  `conv2d` channels (see [`05-soundness.md`](05-soundness.md)). That schedule
  was only within the reordering bound; the current one is bit-exact, and its
  timings have not been re-measured.

## Results

Cells show median per-call time ± sample standard deviation from one run.
Each harness also reports the best sample, but medians are used for comparisons.

| Operation and shape    |      C++ reference |       MLIR untiled |     MLIR scheduled |          Mojo API | Mojo tiled, allocating | Mojo tiled, reused |
| ---------------------- | -----------------: | -----------------: | -----------------: | ----------------: | ---------------------: | -----------------: |
| matmul 64³             |  120.01 ± 11.71 µs |   122.59 ± 0.69 µs |    14.51 ± 0.24 µs |  30.57 ± 18.93 µs |                      — |                  — |
| matmul 128³            |   1.211 ± 0.018 ms |   1.342 ± 0.006 ms |   116.70 ± 0.52 µs | 254.53 ± 22.30 µs |                      — |                  — |
| matmul 256³            |  12.091 ± 0.265 ms |  14.108 ± 1.021 ms | 940.63 ± 880.30 µs |  1.737 ± 0.148 ms |       730.07 ± 4.04 µs |   722.92 ± 2.43 µs |
| matmul 512³            | 109.059 ± 0.290 ms | 116.476 ± 1.097 ms |   7.992 ± 0.093 ms | 14.749 ± 0.109 ms |       5.754 ± 0.646 ms |   5.858 ± 0.168 ms |
| conv2d 56×56×64 → 64   |  89.289 ± 0.358 ms |  91.429 ± 0.405 ms |  10.755 ± 0.119 ms | 14.837 ± 0.095 ms |                      — |                  — |
| conv2d 28×28×128 → 128 |  87.848 ± 2.723 ms |  87.824 ± 1.005 ms |  25.623 ± 1.805 ms | 13.107 ± 0.155 ms |                      — |                  — |
| qmatmul 256³           |   9.162 ± 0.375 ms |   8.344 ± 1.270 ms | 878.28 ± 231.56 µs |  1.868 ± 0.009 ms |                      — |                  — |

For the larger matmuls, scheduled MLIR is about 13.6× faster than the C++
reference by median at 512³; allocation-inclusive tiled Mojo is about 18.7×
faster.
Untiled MLIR is close to the reference, while the ordinary Mojo `matmul` is
faster than the reference but slower than the scheduled implementations.
The 4×16 tiled Mojo kernel is about 1.39× faster than scheduled MLIR at 512³
by median. The allocation-versus-reuse difference is smaller than the sample
spread at 512³, so this run cannot establish whether allocation matters at
that size.

There is no single winner across all measured operations. Mojo is faster than
scheduled MLIR on the 28×28 conv2d case, while scheduled MLIR is faster on the
larger 56×56 case and on qmatmul. These are prompts for investigation, not a
general ranking.

## Interpretation and limits

- The schedule materially improves the measured kernels over untiled MLIR and
  the scalar reference on this host.
- The Mojo `matmul_tiled` tile size is fixed at 4×16 here; no tile sweep was
  performed. This result does not establish the best tile for the M2.
- Ordinary Mojo matmul and tiled matmul use different loop strategies. These
  measurements compare implementations; they do not isolate compiler quality.
- The sample standard deviations expose substantial noise in some cases,
  notably matmul 256³. Ten samples are a useful warning signal, not a
  confidence interval; repeat runs before making a performance claim.
- The C++/MLIR harness requests interactive QoS on macOS; the Mojo harness
  does not set an explicit QoS class. macOS offers no hard CPU affinity, so
  this is a remaining scheduling difference.
- These are call timings, not a complete model or application pipeline.
- No GPU is involved. These results are for the Apple M2 CPU's NEON path;
  GPU measurements are in [`06-gpu-results.md`](06-gpu-results.md).

Detailed command options, correctness checks, and roofline methodology are in
[`../benchmarks/README.md`](../benchmarks/README.md).
