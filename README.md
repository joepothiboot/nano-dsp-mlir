# nano-dsp-mlir ⚡

A small MLIR compiler that lowers a tiny image/math DSL all the way down to
hardware-aware LLVM IR, plus a Mojo kernel library for the same ops on CPU
and GPU. Every implementation produces the same numbers, bit for bit. Built
as a portfolio project for compiler and kernel-library roles focused on
hardware optimization.

🚀 **[Try the demo live](https://joepothiboot.github.io/nano-dsp-mlir/)**: traces
a matmul from the `dsp` dialect to NEON and AVX2 machine code, with bit-exact
proof that the optimization didn't change the answer.

```
blur(image) + bias        →  dsp.conv2d / dsp.add        →  linalg.generic
                           →  tiled + vectorized loops    →  memref (buffers)
                           →  LLVM IR                     →  machine code
```

It isn't a production compiler, and it isn't trying to be. It's a deliberately
small, fully-tested pipeline built to do one thing well: **make a
hardware-aware optimization decision (L1 cache tiling, SIMD width) and prove it
didn't change the program's output.**

---

## 💡 Why this project exists

Most "I did the MLIR Toy tutorial" projects stop at "it lowers to LLVM and
runs." This one goes further:

1. 📐 Tile sizes and vector widths are **derived from an explicit machine
   model** (`docs/02-tiling-model.md`): not hardcoded, and not naively queried
   at runtime. The derivation is in `docs/02-tiling-model.md`; the full
   architecture write-up (`docs/00-architecture.md`) is still planned.
2. 📋 The optimization pass is written as **data (a Transform-dialect
   schedule)**, not baked into a C++ pass, so a tuning sweep is a shell loop
   instead of 40 rebuilds.
3. 🧪 Every optimization is backed by a **differential test**: the same
   program run before and after tiling/vectorization must produce identical
   (or provably-bounded) numeric output. A fast but wrong compiler is useless,
   so correctness tests outnumber structural tests in this repo.

---

## 🏗️ Architecture

Six IR levels, each with a stated invariant. Full diagram:
[`docs/architecture.excalidraw`](docs/architecture.excalidraw) (open at
[excalidraw.com](https://excalidraw.com) → File → Open).

```
┌─────────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────────────┐   ┌─────────────┐   ┌─────────────┐
│ L0  DSL     │──▶│ L1  dsp      │──▶│ L2  linalg     │──▶│ L3  Tiled +       │──▶│ L4  memref  │──▶│ L5  LLVM IR │
│ Python AST  │   │ dialect      │   │ .generic on    │   │ vectorized        │   │ (buffers,   │   │ (.o / JIT)  │
│ subset      │   │ (conv2d/add/ │   │ tensors (DPS)  │   │ scf.for +         │   │ ownership   │   │             │
│             │   │ matmul/relu) │   │                │   │ vector.*          │   │ resolved)   │   │             │
└─────────────┘   └──────────────┘   └───────────────┘   └────────▲──────────┘   └─────────────┘   └─────────────┘
                                                                    │
                                                          ┌─────────┴─────────┐
                                                          │  TargetModel  +   │
                                                          │  Transform Schedule│
                                                          │ (schedules/*.mlir) │
                                                          │  "schedule = data" │
                                                          └────────────────────┘
```

| Level | Dialects                           | Invariant                                       |
| ----- | ---------------------------------- | ----------------------------------------------- |
| L0    | Python AST subset                  | static shapes, f32 only, no control flow        |
| L1    | `dsp`                              | value semantics, whole-array ops, no loops      |
| L2    | `linalg` on tensors                | destination-passing style, explicit affine maps |
| L3    | `linalg`/`scf`/`vector` on tensors | schedule already applied, still no memory       |
| L4    | `memref`, `scf`, `vector`          | aliasing resolved, allocations explicit         |
| L5    | `llvm`                             | native vector widths, no `memref` left          |

Full contracts: [`docs/01-ir-contracts.md`](docs/01-ir-contracts.md) (planned).

---

## 🧭 Key design decisions

Two judgment calls drive the whole project. The second is worked through in
[`docs/02-tiling-model.md`](docs/02-tiling-model.md); the full argument for both
is planned for `docs/00-architecture.md`.

**1. Transform dialect (policy) + C++ (mechanism), not one or the other.**
C++ owns what the compiler _can_ do (`dsp → linalg` conversion, since no
upstream op exists for that). The Transform dialect owns what it _chose_ to do
(tile sizes, fusion, vectorization) as a checked-in `.mlir` schedule file. One
flag (`-nanodsp-optimize`) runs a generated default;
`-nanodsp-optimize=schedule-file=...` swaps in a hand-tuned one, same code
path.

**2. Tile sizes are derived, not hardcoded or runtime-queried.**
A compile-time `TargetModel` (L1 size, vector width, register count) feeds an
analytical working-set model (`lib/Schedule/TileSizeModel.cpp`) that solves
for the largest tile that fits `α × L1d`, then snaps to register-width
multiples ([`docs/02-tiling-model.md`](docs/02-tiling-model.md)). Validating it
against a brute-force sweep, with a model-vs-measured comparison in
`docs/03-results.md`, remains future work; that document currently records the
first cross-implementation kernel measurements, not a tile-model validation.

---

## 🗂️ Directory structure

```
nano-dsp-mlir/
├── include/nanodsp/
│   ├── Dialect/DSP/IR/       # dsp.{add,relu,matmul,conv2d,qmatmul} — ODS + verifiers
│   ├── Conversion/DSPToLinalg/  # dsp -> linalg.generic lowering
│   └── Schedule/             # TargetModel, TileSizeModel, schedule passes
├── lib/                      # .cpp for everything above
├── tools/nanodsp-opt/        # the compiler CLI (mlir-opt clone + our dialect/passes)
├── test/
│   ├── Dialect/DSP/          # op parsing, verification, canonicalization
│   ├── Conversion/DSPToLinalg/  # lowering structure (FileCheck)
│   ├── Schedule/             # generated schedules and scheduled IR (FileCheck)
│   ├── Integration/          # end-to-end execution via mlir-runner
│   └── Hexagon/              # on-target harness: kernels vs the C++ reference
├── schedules/                # hand-written Transform-dialect schedules
├── mojo/
│   ├── nanodsp/              # Mojo package: Tensor[dtype] + SIMD kernels; gpu.mojo for GPU
│   └── tests/                # golden + differential tests
├── reference/                # scalar C++ oracle (header-only) + golden-value test
├── docker/hexagon/           # Hexagon cross toolchain + qemu image
├── scripts/                  # demo page generator, run-hexagon.sh
├── benchmarks/               # Mojo + MLIR/C++ kernel benchmarks
├── docs/                     # design notes
├── pixi.toml                 # Mojo toolchain + task runner
└── test.sh                   # configure + build + run check-nanodsp
```

Planned, not in the repo yet: `frontend/`.

### 🔥 Three implementations, one set of numbers

Every `dsp` op exists three times: lowered through MLIR, as a SIMD Mojo
kernel (`mojo/nanodsp/`), and as a plain scalar C++ loop nest
(`reference/`). All three are tested against the same golden values (the ones
in `test/Integration/`), and the Mojo kernels are also checked against a naive
loop nest on odd sizes so every SIMD tail path runs. See
[`docs/mojo-kernels.md`](docs/mojo-kernels.md).

On top of the kernels sits a small trait-based API: a `TensorLike` trait, a
borrowed `TensorView` whose origin ties it to the buffer it reads, and a
`matmul_tiled` that is generic over both and takes its register tile shape
and SIMD width as compile-time parameters. It stays bit-exact with the other
two implementations because it tiles only the output dims. See
[`docs/mojo-api-design.md`](docs/mojo-api-design.md).

That includes `dsp.qmatmul`, an int8 matmul with zero points and fixed-point
requantization, the way DSP and NPU integer pipelines compute a quantized
layer. See [`docs/quantization.md`](docs/quantization.md).

The Mojo `matmul` and `conv2d` also run on a GPU (`mojo/nanodsp/gpu.mojo`),
with the same bits: a naive, a shared-memory tiled, and a register-blocked
matmul, all summing in the reference's order with multiply and add unfused.
On an Apple M2 GPU the blocked matmul reaches 549 GFLOP/s at 2048³, and
512³ runs in 0.89 ms against 5.75 ms for the best CPU kernel. NVIDIA T4 and
cuBLAS numbers are pending. See [`docs/mojo-gpu.md`](docs/mojo-gpu.md) and
[`docs/06-gpu-results.md`](docs/06-gpu-results.md).

---

## 🚀 Getting started

### 📋 Prerequisites

Pin the exact revision. MLIR's transform-dialect and pass APIs change a lot
between releases.

LLVM/MLIR 21 or newer is required (the code uses the `Op::create(builder, ...)`
API); CI and local development use 23.1.1. On macOS, `brew install llvm` is
enough and `./test.sh` finds it automatically. Otherwise build from source:

```bash
# built with -DLLVM_ENABLE_PROJECTS="mlir"
git clone --branch llvmorg-23.1.1 https://github.com/llvm/llvm-project
```

You'll need `MLIR_DIR` pointing at the install, plus `lit` and `FileCheck` on
`PATH`.

### 🔨 Build

```bash
cmake -G Ninja -B build \
  -DMLIR_DIR=$LLVM_INSTALL/lib/cmake/mlir \
  -DLLVM_EXTERNAL_LIT=$(which lit)
cmake --build build
```

### 🧪 Test

```bash
cmake --build build --target check-nanodsp
```

This runs three tiers: dialect verification, lowering-structure checks
(FileCheck), and differential correctness tests (optimized-vs-baseline
execution via `mlir-runner`).

### ▶️ Run the compiler directly

```bash
build/bin/nanodsp-opt input.mlir \
  -convert-dsp-to-linalg \
  -nanodsp-optimize=target=host-neon \
  -nanodsp-lower-to-llvm
```

- `-nanodsp-optimize` tiles and vectorizes every `linalg.generic` with a
  schedule generated from a `TargetModel` (`host-neon`, `x86-avx2` or
  `hexagon-hvx128`).
  `-nanodsp-optimize=schedule-file=schedules/matmul-8x12-neon.mlir` applies a
  hand-written schedule instead.
- `-nanodsp-emit-schedule=target=...` appends the generated schedule to the
  module, so you can read it, edit it, and check it in.
- `-nanodsp-lower-to-llvm` bufferizes and runs the upstream lowering to the
  LLVM dialect. Drop `-nanodsp-optimize` to get the unscheduled scalar loops.

The Stage 5 benchmark harness compares the scalar C++ reference with untiled
and scheduled MLIR kernels, and `pixi run bench` measures the Mojo kernels.
The current measured results are in [`docs/03-results.md`](docs/03-results.md),
with detailed harness methodology in [`benchmarks/README.md`](benchmarks/README.md).
A broader schedule/tile sweep and model-vs-measured analysis remain future work.

### 🖥️ Demo page

🌐 **Live:** https://joepothiboot.github.io/nano-dsp-mlir/ (redeployed on every
push to `main` by [`deploy-pages.yml`](.github/workflows/deploy-pages.yml))

```bash
python3 scripts/gen_demo.py   # writes build/demo/index.html
```

One page that traces `demo/matmul.mlir` from the `dsp` dialect to NEON and
AVX2 machine code, shows the tile sizes each target model picks, and reports
the bit-exact results. The script runs the real tools (`nanodsp-opt`, `llc`,
`mlir-runner`, the lit suite) and injects their output into
`demo/template.html`, so nothing on the page is written by hand.

### 🔶 Hexagon (emulated)

```bash
docker build --platform linux/amd64 -t nanodsp-hexagon docker/hexagon
scripts/run-hexagon.sh
```

Compiles the kernels for a Hexagon V68 with HVX and runs them under
`qemu-hexagon`, checking every value bit for bit against the scalar C++
reference. The scalar and integer-HVX builds pass; the f32 HVX builds compile
but need a newer QEMU than the image has. Nothing here is timed. See
[`docs/hexagon-target.md`](docs/hexagon-target.md).

### 🧱 Local memory (scratchpad + DMA)

```bash
build/bin/nanodsp-opt input.mlir -convert-dsp-to-linalg \
  -nanodsp-optimize=target=hexagon-hvx128 \
  -nanodsp-lower-to-llvm=local-target=hexagon-hvx128
```

For a target with local memory (Hexagon VTCM), `-nanodsp-promote-local`
copies the input tiles of each cache tile into `#dsp.local` buffers with
double-buffered `memref.dma_start`/`dma_wait`, prefetching the next tile
before computing on the current one. `-nanodsp-lower-local` then turns the
DMAs into synchronous copies, since neither the host nor `qemu-hexagon` has a
DMA engine. Results stay bit-exact on the host and on the emulator; no
speedup is claimed. See [`docs/scratchpad-dma.md`](docs/scratchpad-dma.md).

### 🔥 Mojo kernels and C++ reference

The Mojo toolchain (pinned to 1.1) is installed through [pixi](https://pixi.sh):

```bash
pixi run test-mojo       # golden, differential and view/tiling tests for the Mojo library
pixi run test-reference  # golden tests for the C++ reference
pixi run bench           # Mojo kernel throughput
```

The MLIR-vs-C++ benchmark harness (`pixi run bench-check`,
`pixi run bench-mlir`) is described in
[`benchmarks/README.md`](benchmarks/README.md).

### 🖥️ GPU kernels (Mojo)

```bash
pixi run test-gpu        # GPU matmul and conv2d, bit for bit against the CPU kernels
pixi run bench-gpu       # writes build/bench/results-gpu.json
```

Needs a GPU: Apple silicon (macOS 15+, Xcode with its Metal toolchain) or
NVIDIA Turing and newer. Running on a free Colab or Kaggle T4, and the
cuBLAS baseline, are described in [`docs/mojo-gpu.md`](docs/mojo-gpu.md).

---

## 📊 Project status

| Stage | Scope                                                    | Status                                                            |
| ----- | -------------------------------------------------------- | ----------------------------------------------------------------- |
| 1     | Architecture + judgment calls                            | ✅ done                                                           |
| 2     | `dsp` dialect + lowering to `linalg.generic`             | ✅ done                                                           |
| 3     | Tiling + vectorization (Transform dialect schedule)      | ✅ done, bit-exact (`lib/Schedule/`, `test/Schedule/`)            |
| 4     | Bufferization + `linalg → scf → vector → LLVM`           | ✅ done (upstream passes, see `test/Integration/end-to-end.mlir`) |
| L     | Local memory: VTCM promotion + double-buffered DMA       | ✅ done, functional only (`docs/scratchpad-dma.md`)               |
| 5     | Benchmark harness + cross-implementation measurements    | ✅ harnesses and first M2 comparison (`benchmarks/`)              |
| M     | Mojo kernel library + C++ reference oracle               | ✅ done (`mojo/`, `reference/`; Mojo 1.1)                         |
| 6     | (planned) DSL frontend polish, autotuning sweep write-up | ⏳ not started                                                    |
| G     | Mojo GPU kernels: matmul (3 variants) + conv2d           | ✅ Apple M2, bit-exact (`docs/mojo-gpu.md`); NVIDIA T4 pending    |

---

## 📚 Docs index

Docs that exist:

- [`docs/03-results.md`](docs/03-results.md): first Mojo/MLIR/C++ measurements and limitations
- [`docs/mojo-gpu.md`](docs/mojo-gpu.md): the Mojo GPU kernels: API, why they stay bit-exact, how to run them on Apple silicon and a free T4
- [`docs/06-gpu-results.md`](docs/06-gpu-results.md): Apple M2 GPU measurements; NVIDIA T4 and cuBLAS pending
- [`docs/mojo-kernels.md`](docs/mojo-kernels.md): the Mojo library: design, SIMD strategy, how it's tested
- [`docs/mojo-api-design.md`](docs/mojo-api-design.md): the Mojo `TensorLike` trait, borrowed views and origins, compile-time tiles, and why only i/j are tiled
- [`docs/02-tiling-model.md`](docs/02-tiling-model.md): the working-set derivation
- [`docs/04-schedule-ir-diff.md`](docs/04-schedule-ir-diff.md): before/after IR for one matmul
- [`docs/05-soundness.md`](docs/05-soundness.md): why tiling/vectorization can't change results
- [`docs/quantization.md`](docs/quantization.md): `dsp.qmatmul` semantics, rounding, lowering
- [`docs/hexagon-target.md`](docs/hexagon-target.md): the Hexagon HVX target model and the emulated bit-exact run
- [`docs/scratchpad-dma.md`](docs/scratchpad-dma.md): promoting cache tiles to local memory (VTCM) with double-buffered DMA, and lowering it for host/emulator

Planned:

- [`docs/00-architecture.md`](docs/00-architecture.md): full Stage 1 plan and both judgment-call tradeoffs
- [`docs/01-ir-contracts.md`](docs/01-ir-contracts.md): the L0–L5 invariant table
- [`docs/06-amendments.md`](docs/06-amendments.md): deviations from the original plan, and why
- [`docs/07-mlir-for-js-devs.md`](docs/07-mlir-for-js-devs.md): MLIR concepts explained via JS/Babel analogies
- [`docs/architecture.excalidraw`](docs/architecture.excalidraw): the diagram above, editable

## 🚧 Known limitations

- No loop interchange yet. Register tiles nest inside L1 reduction loops
  instead of outside; that's the first knob for the autotuning stage.
- Tile sizes must divide the loop extents (no masking or peeling yet), so an
  awkward or prime extent loses register blocking.
- The accumulator tile is re-read and re-written on every `k` step; hoisting
  it out of the `k` loop isn't done yet.
- `cacheFraction = 0.5` is a starting estimate, not yet validated against
  hardware. That validation is the point of Stage 6.
- Local-memory promotion is functional only: DMAs are lowered to synchronous
  copies, QEMU models neither VTCM nor DMA timing, and nothing maps
  `memref.dma_start` to a real DMA engine. The output tile is not promoted,
  and cache tiles are sized against L2, not VTCM (promotion falls back to
  single buffering when double buffers don't fit).

## 📜 License

MIT. See [`LICENSE`](LICENSE).
