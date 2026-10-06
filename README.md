# nano-dsp-mlir

An out-of-tree MLIR compiler for a small tensor DSL, plus a Mojo kernel library
that implements the same ops on CPU and GPU.

The compiler defines a `dsp` dialect (`add`, `relu`, `matmul`, `conv2d` and the
int8 `qmatmul`), lowers it to `linalg.generic`, tiles and vectorizes it with a
Transform-dialect schedule derived from a target model, and lowers the result
to LLVM. The MLIR pipeline, the Mojo kernels and a scalar C++ reference are all
tested against the same golden values, and the optimized code must match the
unoptimized code bit for bit.

Demo: https://joepothiboot.github.io/nano-dsp-mlir/ traces one matmul from the
`dsp` dialect to NEON and AVX2 machine code.

## Pipeline

```
dsp dialect          dsp.conv2d, dsp.matmul, ...      value semantics, no loops
  -> linalg          linalg.generic on tensors        destination-passing style
  -> tiled/vector    scf.for + vector on tensors      schedule applied
  -> memref          bufferized                       allocations explicit
  -> LLVM            llvm dialect -> machine code
```

Tiling decisions come from a `TargetModel` (vector width, register count, cache
size). An analytical working-set model in `lib/Schedule/TileSizeModel.cpp` picks
the largest tile that fits a fraction of L1 and snaps it to register-width
multiples; see [docs/02-tiling-model.md](docs/02-tiling-model.md). The chosen
tiles are emitted as a Transform-dialect schedule, so a schedule can be printed,
edited and checked in, or replaced with a hand-written one, without rebuilding
the compiler.

Supported targets: `host-neon`, `x86-avx2`, `hexagon-hvx128`.

## Status

| Area                                                  | Status                                                    |
| ----------------------------------------------------- | --------------------------------------------------------- |
| `dsp` dialect and lowering to `linalg.generic`        | Done                                                      |
| Tiling and vectorization (Transform dialect)          | Done, bit-exact                                           |
| Bufferization and lowering to LLVM                    | Done (upstream passes)                                    |
| Local memory: VTCM promotion with double-buffered DMA | Functional; DMAs lowered to synchronous copies            |
| Benchmark harness, MLIR vs Mojo vs C++                | First measurements on Apple M2                            |
| Mojo CPU kernels and C++ reference                    | Done (Mojo 1.1)                                           |
| Mojo GPU kernels (matmul in 3 variants, conv2d)       | Bit-exact on Apple M2 and NVIDIA T4, compared with cuBLAS |
| Python DSL front end, tile-size sweep                 | Not started                                               |

GPU results: on an NVIDIA T4 the register-blocked matmul reaches 1.9 TFLOP/s,
49–58% of cuBLAS from 512³ up. On an Apple M2 GPU it reaches 549 GFLOP/s at
2048³. Details in [docs/06-gpu-results.md](docs/06-gpu-results.md).

## Building

Requirements: LLVM/MLIR 21 or newer (tested with 23.1.1), CMake, Ninja, `lit`
and `FileCheck`. On macOS, `brew install llvm` is enough.

```bash
./test.sh
```

`test.sh` finds Homebrew LLVM, configures, builds `build/bin/nanodsp-opt` and
runs the lit suite. To configure manually:

```bash
cmake -G Ninja -B build \
  -DMLIR_DIR=$LLVM_INSTALL/lib/cmake/mlir \
  -DLLVM_EXTERNAL_LIT=$(which lit)
cmake --build build --target check-nanodsp
```

The lit suite covers dialect parsing and verification, lowering structure
(FileCheck), generated schedules, and end-to-end execution with `mlir-runner`,
including a test that diffs scheduled and unscheduled results bit for bit.

## Usage

```bash
build/bin/nanodsp-opt input.mlir \
  -convert-dsp-to-linalg \
  -nanodsp-optimize=target=host-neon \
  -nanodsp-lower-to-llvm
```

- `-nanodsp-optimize=target=<name>` generates and applies a schedule for the
  target. Use `schedule-file=schedules/matmul-8x12-neon.mlir` to apply a
  hand-written one instead.
- `-nanodsp-emit-schedule=target=<name>` appends the generated schedule to the
  module so it can be inspected.
- `-nanodsp-lower-to-llvm` bufferizes and lowers to the LLVM dialect. Add
  `local-target=hexagon-hvx128` to promote cache tiles to local memory (see
  [docs/scratchpad-dma.md](docs/scratchpad-dma.md)).

## Mojo kernels and C++ reference

The Mojo toolchain is pinned to 1.1 and installed through
[pixi](https://pixi.sh).

```bash
pixi run test-mojo        # golden and differential tests for the Mojo library
pixi run test-reference   # golden tests for the C++ reference
pixi run test-gpu         # GPU kernels vs CPU kernels, bit for bit (needs a GPU)
pixi run bench            # Mojo CPU benchmarks
pixi run bench-gpu        # GPU benchmarks, writes build/bench/results-gpu.json
pixi run bench-check      # MLIR kernels vs the C++ reference, no timing
pixi run bench-mlir       # MLIR vs C++ benchmark
```

Kernels keep the reduction order of the naive loop nest and keep multiply and
add unfused (`--fp-mode contract=off` in Mojo, `-ffp-contract=off` in C++), so
all three implementations can be compared exactly. The GPU kernels run on Apple
silicon (macOS 15+) and NVIDIA Turing or newer; running on a free Colab T4 is
described in [docs/mojo-gpu.md](docs/mojo-gpu.md).

## Other tools

- Demo page: `python3 scripts/gen_demo.py` runs the real tools, fills
  `demo/template.html` with their output and writes it with `demo/style.css`
  and `demo/app.js` to `build/demo/`. It is redeployed on every push to `main`.
- Hexagon: `docker build --platform linux/amd64 -t nanodsp-hexagon docker/hexagon`,
  then `scripts/run-hexagon.sh` runs the kernels on an emulated Hexagon V68 with
  HVX and checks them against the C++ reference. The scalar and integer HVX
  builds pass; the f32 HVX builds need a newer QEMU. See
  [docs/hexagon-target.md](docs/hexagon-target.md).

## Repository layout

```
include/nanodsp/, lib/   dialect, lowering, target model, schedule passes
tools/nanodsp-opt/       the compiler driver
test/                    lit tests (Dialect, Conversion, Schedule, Integration, Hexagon)
schedules/               hand-written Transform-dialect schedules
mojo/nanodsp/            Mojo package (CPU kernels, layout API, GPU kernels)
mojo/tests/              Mojo tests
reference/               header-only scalar C++ reference
benchmarks/              benchmark harnesses and results
docker/hexagon/          Hexagon toolchain and QEMU image
scripts/                 demo generator, benchmark and Hexagon scripts
docs/                    design notes
```

## Documentation

- [docs/02-tiling-model.md](docs/02-tiling-model.md): working-set model for tile sizes
- [docs/03-results.md](docs/03-results.md): MLIR, Mojo and C++ measurements on M2
- [docs/04-schedule-ir-diff.md](docs/04-schedule-ir-diff.md): IR before and after scheduling
- [docs/05-soundness.md](docs/05-soundness.md): why the schedules preserve results
- [docs/06-gpu-results.md](docs/06-gpu-results.md): GPU measurements, M2 and T4 vs cuBLAS
- [docs/mojo-kernels.md](docs/mojo-kernels.md), [docs/mojo-api-design.md](docs/mojo-api-design.md), [docs/mojo-gpu.md](docs/mojo-gpu.md): Mojo library design
- [docs/quantization.md](docs/quantization.md): `dsp.qmatmul` semantics and rounding
- [docs/hexagon-target.md](docs/hexagon-target.md), [docs/scratchpad-dma.md](docs/scratchpad-dma.md): Hexagon target and local memory
- [benchmarks/README.md](benchmarks/README.md): benchmark methodology and options

## Known limitations

- No loop interchange: register tiles nest inside the L1 reduction loops.
- Tile sizes must divide loop extents; there is no masking or peeling.
- The accumulator tile is reloaded on every `k` step instead of being hoisted.
- `cacheFraction = 0.5` has not been validated against hardware.
- Local-memory DMAs are lowered to synchronous copies; QEMU models neither VTCM
  nor DMA timing. The output tile is not promoted.

## License

MIT. See [LICENSE](LICENSE).
