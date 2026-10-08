# Benchmarks ⏱️

Four harnesses live here:

- `bench_kernels.mojo`: allocation-inclusive Mojo API timings and tiled
  matmul timings (`pixi run bench`).
- `bench_gpu.mojo`: the Mojo GPU matmul and conv2d, bit-checked against the
  CPU kernels before timing (`pixi run bench-gpu`). Saved runs are in
  `results/`; see [`docs/mojo-gpu.md`](../docs/mojo-gpu.md).
- `bench_cublas.py`: cuBLAS on the same matmul inputs, checked within the
  reordering bound below; needs an NVIDIA GPU and CuPy.
- `kernels.mlir` + `harness.cpp` + `ceilings.cpp`: the Stage 5 comparison of
  MLIR-compiled kernels, untiled and scheduled, against the scalar C++
  reference, with measured roofline ceilings. The rest of this file is about
  this one.

## ▶️ Running

```bash
./test.sh                # builds build/bin/nanodsp-opt
pixi run bench-check     # build + correctness only, no timing (CI runs this)
pixi run bench-mlir      # build + check + time; writes build/bench/results.json
```

Both call `scripts/bench.sh`. Environment variables:

| Variable       | Default                 | Meaning                                                        |
| -------------- | ----------------------- | -------------------------------------------------------------- |
| `NANODSP_OPT`  | `build/bin/nanodsp-opt` | compiler driver to use                                         |
| `LLVM_BIN`     | Homebrew LLVM `bin/`    | `mlir-translate`, `opt`, `llc`, `clang++`, `llvm-objdump`      |
| `BENCH_IR_OPT` | `-O3`                   | `opt` level for the kernels before `llc`; `none` skips `opt`   |
| `BENCH_CPU`    | `native`                | `-mcpu` for `opt`, `llc` and `clang++`                         |
| `BENCH_ARGS`   | (none)                  | extra harness flags: `--samples N`, `--min-time S`, `--filter` |

## 🧩 What is compared

Shapes match `bench_kernels.mojo`:

| Op        | Shapes                                                                         | Element type |
| --------- | ------------------------------------------------------------------------------ | ------------ |
| `matmul`  | square, n = 64, 128, 256, 512                                                  | f32          |
| `conv2d`  | 3×3 valid, NHWC × HWCF: 56×56×64 → 64, 28×28×128 → 128                         | f32          |
| `qmatmul` | 256 × 256 × 256, the quantization parameters of `test/Hexagon/kernels-i8.mlir` | int8         |

Three implementations of each:

| `impl`           | `config`                        | How it is built                                                                      |
| ---------------- | ------------------------------- | ------------------------------------------------------------------------------------ |
| `cpp-ref`        | `clang++ -O3 -ffp-contract=off` | `reference/nanodsp_ref.h` called directly                                            |
| `mlir-untiled`   | `none`                          | `nanodsp-opt -convert-dsp-to-linalg -nanodsp-lower-to-llvm`                          |
| `mlir-scheduled` | `host-neon`                     | same, plus `-nanodsp-optimize=target=host-neon` (the `TargetModel`-derived schedule) |

`pixi run bench` separately measures the Mojo implementation on the same
shapes and deterministic input patterns. It reports `matmul`, `conv2d`, and
`qmatmul` with result allocation included, plus 4×16 tiled matmul both with a
fresh output allocation per call and with a reused output. Keep those tiled
modes distinct: allocation-inclusive timings are the closer API-level
comparison; reused-output timings isolate repeated calls more closely. The
Mojo run is not linked into the C++ harness, so its measurements appear in
terminal output rather than `results.json`.

Both MLIR configurations then go through
`mlir-translate --mlir-to-llvmir | opt -O3 | llc -O3 -filetype=obj`.

### 🔗 Linking both configurations into one binary

`scripts/bench.sh` compiles `kernels.mlir` once per configuration. Before
compiling, a `sed` over the `func.func` names renames every function from
`@matmul_64` to `@matmul_64_untiled` or `@matmul_64_scheduled`. With
`llvm.emit_c_interface`, that gives distinct `_mlir_ciface_matmul_64_<config>`
entry points, and both objects link into one `harness` binary. The kernel list
and the configurations are X-macros at the top of `harness.cpp`.

### ⚙️ Why `opt -O3`

`llc` runs no IR-level optimization. Without `opt`, the untiled kernels load
and store the output element on every step of the reduction, because nothing
promotes the accumulator to a register. `clang -O3` does that for the C++
reference, so skipping `opt` compares two different middle ends. On the M2
used for development, untiled `matmul` 512 took 4.2× as long as the reference
without `opt` and 1.07× as long with it. `opt` adds no fast-math flags, so it
can neither reassociate nor contract. The harness checks every result anyway,
and `bench.sh` fails if the kernel objects contain any FMA instruction.
`BENCH_IR_OPT=none` restores the `llc`-only build.

## ✅ Correctness before timing

The harness computes every result with `nanodsp::ref` first and compares
before it times anything. Any mismatch prints the first bad index and exits
non-zero, and nothing is timed.

- **`matmul`, `qmatmul`**: bit-exact, for both MLIR configurations.
- **`conv2d`**: bit-exact, for both MLIR configurations.

The harness still accepts a result within a reordering bound if it is not
bit-identical: any two summation orders of the same `n` products differ by at
most `2·γₙ·Σ|x·w|` per output, with `γₙ = n·u / (1 − n·u)`, `u = 2⁻²⁴`
(Higham, _Accuracy and Stability of Numerical Algorithms_, sec. 3.1). A
dropped or wrong product would exceed it. The `checked` field of each result
says which test passed. Scheduled `conv2d` used to need it, when the cache tile
blocked the channel dim; see `docs/05-soundness.md`.

## ⏱️ Timing method

- Single thread. On macOS the thread asks for `QOS_CLASS_USER_INTERACTIVE`,
  which steers it to a performance core. macOS has no hard affinity, so this
  is a request, not a guarantee.
- One warmup call. Then the calls per sample double until one sample takes at
  least `--min-time` (default 0.05 s). The JSON records the best (minimum),
  median, and sample standard deviation of per-call times over `--samples`
  samples (default 10). Console throughput uses the best time; use the median
  and spread when comparing implementations.
- The Mojo benchmark uses the same warmup, 50 ms minimum sample, 10 samples,
  and timing statistics. Both harnesses are single-threaded.
- Each MLIR call returns a freshly `malloc`ed result, which the timed loop
  frees. The reference allocates its result `std::vector` the same way.
- C++ and MLIR calls include result allocation. Ordinary Mojo kernels also
  include allocation; tiled Mojo matmul reports both allocation and output
  reuse.
- The machine is not isolated: other processes add noise. The minimum is the
  statistic least affected by it; compare the median and sample standard
  deviation to judge how representative that best case is.

The harnesses retain sample statistics but not every sample. Ten samples give
only a rough view of variability, not a confidence interval. The recorded M2
comparison is in
[`docs/03-results.md`](../docs/03-results.md).

## 📏 Ceilings

`ceilings.cpp` measures, on the same machine and thread setup:

| Key                 | Loop                                                                     | Reported                |
| ------------------- | ------------------------------------------------------------------------ | ----------------------- |
| `triad_bw`          | `a[i] = b[i] + s·c[i]`, three 128 MiB float arrays (far beyond L2 + SLC) | best of 10, plus median |
| `neon_fma_peak`     | `vfmaq_f32` into 24 independent accumulators                             | best of 5               |
| `neon_mul_add_peak` | `vmulq_f32` + `vaddq_f32`, unfused, 24 accumulators                      | best of 5               |

Triad bytes are counted the STREAM way: 12 B per element (two reads, one
write), no write-allocate traffic. The compute ceiling that applies to the f32
kernels here is `neon_mul_add_peak`, not `neon_fma_peak`: the kernels keep
multiply and add separate to stay bit-exact, which halves the peak by
construction. Neither ceiling applies to `qmatmul`, whose work is integer.

Nothing here is a datasheet number. If a ceiling can't be measured on a host
(no NEON), its value is `null`.

## 🗂️ `results.json`

Modeled on Google Benchmark's JSON output. The shape of the file, with the
measured values replaced by placeholders (`0.0`, `"..."`):

```json
{
  "context": {
    "date": "...",
    "cpu": "Apple M2",
    "num_cpus": 8,
    "threads": 1,
    "memory_bytes": 8589934592,
    "caches": { "l1d_bytes": 131072, "l2_bytes": 16777216, "note": "..." },
    "os": "...",
    "commit": "abc1234",
    "dirty": false,
    "nanodsp_opt": "...",
    "llc": "...",
    "clang++": "...",
    "kernel_flags": "...",
    "cxx_flags": "...",
    "timing": "..."
  },
  "ceilings": {
    "threads": 1,
    "triad_bw": {
      "value": 0.0,
      "unit": "GB/s",
      "median": 0.0,
      "runs": 10,
      "array_bytes": 0,
      "method": "..."
    },
    "neon_fma_peak": {
      "value": 0.0,
      "unit": "GFLOP/s",
      "accumulators": 24,
      "method": "..."
    },
    "neon_mul_add_peak": {
      "value": 0.0,
      "unit": "GFLOP/s",
      "accumulators": 24,
      "method": "..."
    }
  },
  "benchmarks": [
    {
      "name": "matmul/512x512x512/mlir-scheduled",
      "op": "matmul",
      "shape": "512x512x512",
      "impl": "mlir-scheduled",
      "config": "host-neon",
      "real_time": 0.0,
      "median_time": 0.0,
      "sample_stddev": 0.0,
      "time_unit": "ns",
      "aggregate": "min",
      "iterations": 0,
      "samples": 10,
      "ops": 268435456,
      "rate": 0.0,
      "rate_unit": "GFLOP/s",
      "bytes": 3145728,
      "intensity": 85.333,
      "checked": "bit-exact"
    }
  ]
}
```

Per benchmark:

| Field                   | Meaning                                                                                               |
| ----------------------- | ----------------------------------------------------------------------------------------------------- |
| `name`                  | `op/shape/impl`, unique within a run                                                                  |
| `shape`                 | `MxNxK` for matmul and qmatmul; `HxWxC->F` for conv2d (batch 1, 3×3 filter, valid padding)            |
| `impl`                  | `cpp-ref`, `mlir-untiled` or `mlir-scheduled`; new implementations (Mojo, schedule sweeps) add values |
| `config`                | schedule or target that distinguishes runs of one `impl` (`none`, `host-neon`, ...)                   |
| `real_time`             | best per-call time in `time_unit` (always `ns`); `aggregate` says it is a minimum                     |
| `median_time`           | median per-call time across samples, in `time_unit`                                                   |
| `sample_stddev`         | sample standard deviation of per-call time, in `time_unit`                                            |
| `iterations`, `samples` | calls per timed sample, and number of samples                                                         |
| `ops`                   | `2·M·N·K` (matmul/qmatmul), `2·OH·OW·F·9·C` (conv2d); qmatmul counts integer multiply/add operations  |
| `rate`, `rate_unit`     | `ops / real_time`; `GFLOP/s` for floating point and `GOP/s` for qmatmul                               |
| `bytes`                 | compulsory traffic: each operand read once and the result written once, at the element size           |
| `intensity`             | `ops / bytes`, the roofline x-coordinate                                                              |
| `checked`               | `bit-exact`, or `within-bound` (see above); the reference is `bit-exact` by definition                |

`bytes` is the minimum any implementation must move, not a measurement of
what it does move. It places each kernel on the roofline; it doesn't say
whether the kernel reaches it.
