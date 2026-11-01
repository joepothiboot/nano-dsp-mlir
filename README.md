# nano-dsp-mlir

An experimental compiler playground for learning and showcasing MLIR, tensor
kernel optimization, and low-level code generation.

The project follows a small set of tensor operations from a custom `dsp`
dialect through MLIR lowering and scheduling, with CPU and GPU kernels to
compare against. It is a practice and exploration project, not a production
compiler.

**[See the interactive matmul demo](https://joepothiboot.github.io/nano-dsp-mlir/)** — trace a matrix multiplication from the `dsp` dialect to generated code.

## What it explores

- Defining an MLIR dialect for operations such as `add`, `relu`, `matmul`,
  `conv2d`, and quantized `qmatmul`.
- Lowering tensor operations to `linalg`, then tiling and vectorizing them with
  Transform-dialect schedules.
- Building a simple target model to explore how tile sizes relate to vector
  widths, registers, and cache.
- Comparing MLIR-generated code with scalar C++ and Mojo kernels; the repo also
  includes GPU, CUDA, Triton, Hexagon, and MN1 experiments.

## Build and test

You need LLVM/MLIR 21 or newer, CMake, Ninja, `lit`, and `FileCheck`. On macOS,
install LLVM with Homebrew, then run:

```bash
brew install llvm ninja
pip3 install lit
./test.sh
```

This builds `build/bin/nanodsp-opt` and runs the MLIR test suite.

## Try the compiler

Run the compiler pipeline on an example:

```bash
build/bin/nanodsp-opt test/Integration/end-to-end.mlir \
  -convert-dsp-to-linalg \
  -nanodsp-optimize=target=host-neon \
  -nanodsp-lower-to-llvm
```

Change `host-neon` to another supported target, or inspect the hand-written
schedule in `schedules/`.

## Explore the project

- `include/` and `lib/` — dialect, lowering, and scheduling passes
- `tools/` — compiler and translation tools
- `test/` — MLIR tests and end-to-end examples
- `mojo/`, `cuda/`, and `triton/` — kernel experiments
- `reference/` — scalar C++ reference implementation
- `benchmarks/` — benchmark harnesses and results
- `docs/` — design notes, experiments, and measurements

For a closer look, start with the [tiling model](docs/02-tiling-model.md),
[schedule IR walkthrough](docs/04-schedule-ir-diff.md), or
[GPU experiments](docs/06-gpu-results.md).

## License

MIT. See [LICENSE](LICENSE).
