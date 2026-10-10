# CUDA and Triton on an NVIDIA T4

Hand-written CUDA and Triton matmuls next to the Mojo GPU kernels
([`06-gpu-results.md`](06-gpu-results.md)) and cuBLAS, all measured in one
Colab session on one GPU. Like the other results pages, this is one machine's
snapshot, not a general ranking.

## Method

- NVIDIA Tesla T4 (Turing, compute capability 7.5), driver 580.82.07, free
  Colab runtime. CUDA 13.0 (nvcc 13.0.88), cuBLAS 13.1, Triton 3.6.0 with
  PyTorch 2.11, Mojo 1.1. Raw results:
  [`benchmarks/results/t4-2026-10-10/`](../benchmarks/results/t4-2026-10-10/).
- `scripts/run-t4.sh` builds `cuda/bench.cu`, checks every kernel, benchmarks
  CUDA, Triton and Mojo, and runs Nsight Compute, in that order, in one session.
- Same inputs and timing policy as the Mojo benchmarks: one warmup, ten
  samples of at least 50 ms, one launch plus `synchronize` per call, operands
  already on the device. Cells are best-sample throughput; the 2048³ table also
  gives median ± sample standard deviation.
- **Correctness before timing.** Every CUDA kernel's output is compared bit for
  bit with the scalar C++ reference (`reference/nanodsp_ref.h`), at the four
  benchmark sizes and two non-square shapes (384×640·640×256 and
  64×32·32×128). The kernels write every multiply-add as
  `__fadd_rn(acc, __fmul_rn(a, b))`, which the compiler never fuses, and keep
  the reference's ascending-`k` order. cuBLAS and Triton are checked against
  the reordering bound γₖ·|A|·|B| instead (`bound_used` in the JSON; all
  ≤ 0.005 of the bound).

## Kernels

| kernel           | block tile    | per thread  | idea                                             |
| ---------------- | ------------- | ----------- | ------------------------------------------------ |
| naive            | 16×16 outputs | 1 output    | one thread per output, operands from global      |
| tiled            | 16×16×16      | 1 output    | shared-memory tiles of A and B                   |
| blocked          | 64×64×16      | 4×4 outputs | register blocking, same shape as the Mojo kernel |
| vec              | 128×128×8     | 8×8 outputs | `float4` global and shared loads                 |
| vec, interleaved | 128×128×8     | 8×8 outputs | `vec` with B read in two interleaved halves      |
| Triton           | autotuned     | —           | `tl.dot`, `input_precision="ieee"`               |
| cuBLAS           | chosen by lib | —           | `cublasSgemm`, timed in the same harness         |

## Results

GFLOP/s (best sample) and share of cuBLAS at the same size:

| kernel                |        256³ |        512³ |       1024³ |       2048³ |
| --------------------- | ----------: | ----------: | ----------: | ----------: |
| CUDA naive            |   470 (42%) |   533 (14%) |   470 (13%) |   414 (11%) |
| Mojo naive            |   445 (39%) |   513 (13%) |   456 (12%) |   389 (10%) |
| CUDA tiled            |   690 (61%) |   845 (22%) |   765 (21%) |   625 (16%) |
| Mojo tiled            |   535 (47%) |   632 (16%) |   567 (16%) |   469 (12%) |
| CUDA blocked          |   723 (64%) |  1611 (41%) |  1794 (49%) |  1713 (44%) |
| Mojo blocked          |   816 (72%) |  1780 (46%) |  1912 (52%) |  1818 (47%) |
| CUDA vec              |   304 (27%) |  1284 (33%) |  2337 (64%) |  2620 (68%) |
| CUDA vec, interleaved |   306 (27%) |  1279 (33%) |  2352 (64%) |  2642 (68%) |
| Triton                |   691 (61%) |  2551 (66%) |  3264 (89%) | 3888 (101%) |
| cuBLAS                | 1128 (100%) | 3888 (100%) | 3652 (100%) | 3866 (100%) |

At 2048³:

| kernel                |         median ± sd | GFLOP/s | / cuBLAS | check        |
| --------------------- | ------------------: | ------: | -------: | ------------ |
| CUDA naive            | 42.32 ms ± 11.18 ms |     414 |      11% | bit-exact    |
| Mojo naive            | 44.21 ms ± 364.7 µs |     389 |      10% | bit-exact    |
| CUDA tiled            |  27.48 ms ± 98.9 µs |     625 |      16% | bit-exact    |
| Mojo tiled            | 36.79 ms ± 721.4 µs |     469 |      12% | bit-exact    |
| CUDA blocked          |  10.09 ms ± 78.4 µs |    1713 |      44% | bit-exact    |
| Mojo blocked          |  9.46 ms ± 249.0 µs |    1818 |      47% | bit-exact    |
| CUDA vec              |   6.61 ms ± 41.2 µs |    2620 |      68% | bit-exact    |
| CUDA vec, interleaved |   6.58 ms ± 51.3 µs |    2642 |      68% | bit-exact    |
| Triton                |  4.77 ms ± 432.1 µs |    3888 |     101% | within-bound |
| cuBLAS                |   4.48 ms ± 61.4 µs |    3866 |     100% | within-bound |

Triton's autotuner picked 32×64×32 at 256³, 64×64×32 at 512³, 64×64×16 at
1024³ and 128×128×16 (8 warps) at 2048³.

An earlier session the same day (before the interleaved variant and the Triton
fix) measured cuBLAS at 3922 GFLOP/s and `vec` at 2676 at 2048³, within 2% of
the numbers above.

## Nsight Compute at 1024³

`ncu` locks the clocks below what the benchmark runs at, so only the counters
are compared here, not the durations.

| kernel                           |         grid | waves / SM | regs | achieved occ. | warp cycles / issued inst. | SM throughput | smem ld conflicts | smem st conflicts |
| -------------------------------- | -----------: | ---------: | ---: | ------------: | -------------------------: | ------------: | ----------------: | ----------------: |
| naive                            |         4096 |       25.6 |   52 |           99% |                       35.1 |           63% |                 0 |                 0 |
| tiled                            |         4096 |       25.6 |   36 |           99% |                       31.5 |           74% |                 0 |                 0 |
| blocked                          |          256 |        1.6 |   64 |           82% |                       14.5 |           60% |                 0 |         7,864,320 |
| vec                              |           64 |        0.8 |  128 |           35% |                        6.2 |           71% |         4,194,304 |           262,144 |
| vec, interleaved                 |           64 |        0.8 |  103 |           35% |                        6.2 |           70% |                 0 |           262,144 |
| cuBLAS (`volta_sgemm_128x64_nn`) | 640 (8×16×5) |        4.0 |  122 |           47% |                        6.8 |           89% |            39,459 |           258,879 |

## Reading it

- **The progression is the textbook one.** `naive` stalls on the global-memory
  queue (27.7 of its 35 cycles per instruction, per `ncu`); `tiled` moves the
  pressure to the shared-memory queue (17.5 cycles); register blocking cuts the
  cycles per instruction to 14.5 and then 6.2, close to cuBLAS's 6.8.
- **Same algorithm, two languages.** CUDA and Mojo `blocked` use the same
  64×64×16 tile and 4×4 per thread; Mojo is 6% faster at 2048³. CUDA `tiled` is
  33% faster than Mojo `tiled` at the same size. Neither gap has been traced
  to the generated code yet.
- **`vec` loses at small sizes because it can't fill the GPU.** A 128×128 tile
  gives 4 blocks at 256³ and 64 at 1024³, under one wave on the T4's 40 SMs.
  Triton's autotuner picks 64×64 or smaller up to 1024³ for the same reason.
- **Bank conflicts were not the bottleneck.** Reading B in two interleaved
  halves removed all 4.2M shared-load conflicts in `vec` and freed 25
  registers per thread, and the time did not change.
- **cuBLAS splits `k`.** Its grid is 8×16×5: the reduction is cut into five
  slices to reach four waves at 1024³. Split-K changes the summation order, so
  it is off the table for these bit-exact kernels; this is one concrete cost of
  bit-exactness on this GPU.
- **Triton matches cuBLAS at 2048³** by best sample (3888 against 3866
  GFLOP/s); by median it is 6% slower, with a wide spread (± 432 µs). It is
  not bit-exact even with `enable_fp_fusion=False`; the likely reason is that
  `tl.dot` itself emits fused multiply-adds (not verified in the generated
  code).
- **Open question: the cost of keeping multiply and add separate.** The
  bit-exact kernels issue two instructions per multiply-add where cuBLAS and
  Triton issue one fused instruction. On the M2 GPU the kernels were
  memory-bound and fusion made no difference ([`06`](06-gpu-results.md)); on
  the T4 it may be part of the remaining 32%. This profile does not include
  instruction counts, so it is untested; a fused build of `vec` would answer it.

## Reproduce

On a Colab T4 runtime, from a checkout (or an uploaded tarball of one):

```
!bash scripts/run-t4.sh
```

It writes `build/t4-results.tar.gz` with `env.txt`, `cuda-t4.json`,
`triton-t4.json`, `mojo-t4.json` and `ncu-t4.csv`. `RUN_MOJO=0` skips the Mojo
install and run.

## Limits

- One GPU, a shared cloud instance, one session.
- Square sizes that are multiples of 128; the tiled kernels do no bounds checks.
- f32 on the CUDA cores only; tensor cores are not used here.
- `ncu` was run at 1024³ only.
