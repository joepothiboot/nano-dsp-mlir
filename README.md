# nano-dsp-mlir

A small MLIR compiler that lowers a tiny image/math DSL down to hardware-aware
LLVM IR — built as a portfolio project for hardware-optimization-focused
compiler roles.

```
blur(image) + bias        →  dsp.conv2d / dsp.add        →  linalg.generic
                           →  tiled + vectorized loops    →  memref (buffers)
                           →  LLVM IR                     →  machine code
```

It is not a production compiler. It is a deliberately small, fully-tested
pipeline built to demonstrate one specific thing well: **taking a
hardware-aware optimization decision (L1 cache tiling, SIMD width) and proving
it did not change the program's output.**

---

## Why this project exists

Most "I did the MLIR Toy tutorial" portfolio projects stop at "it lowers to
LLVM and runs." The differentiator here is:

1. Tile sizes and vector widths are **derived from an explicit machine model**
   (`docs/02-tiling-model.md`), not hardcoded, and not naively queried at
   runtime — see the tradeoff writeup in `docs/00-architecture.md`.
2. The optimization pass is written as **data (a Transform-dialect schedule)**,
   not baked into a C++ pass — so a tuning sweep is a shell loop, not 40
   rebuilds.
3. Every optimization is backed by a **differential test**: the same program
   run before and after tiling/vectorization must produce identical (or
   provably-bounded) numeric output. A fast but wrong compiler is worthless,
   so correctness tests outnumber structural tests in this repo.

---

## Architecture

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

Full contracts: [`docs/01-ir-contracts.md`](docs/01-ir-contracts.md).

---

## Key design decisions

Two judgment calls drive the whole project; both are argued in full in
[`docs/00-architecture.md`](docs/00-architecture.md).

**1. Transform dialect (policy) + C++ (mechanism), not one or the other.**
C++ owns what the compiler _can_ do (`dsp → linalg` conversion — no upstream
op exists for that). The Transform dialect owns what it _chose_ to do (tile
sizes, fusion, vectorization) as a checked-in `.mlir` schedule file. One flag
(`-nanodsp-optimize`) runs a generated default; `-schedule-file=...` swaps in
a hand-tuned one, same code path.

**2. Tile sizes are derived, not hardcoded or runtime-queried.**
A compile-time `TargetModel` (L1 size, vector width, register count) feeds an
analytical working-set model (`lib/Schedule/TileSizeModel.cpp`) that solves
for the largest tile that fits `α × L1d`, then snaps to register-width
multiples. Validated against a brute-force sweep in `benchmark/sweep.py` —
model-vs-measured is the highest-value artifact in the repo
(`docs/03-results.md`).

---

## Directory structure

```
nano-dsp-mlir/
├── include/nanodsp/
│   ├── Dialect/DSP/IR/       # dsp.{add,relu,matmul,conv2d} — ODS + verifiers
│   └── Conversion/DSPToLinalg/  # dsp -> linalg.generic lowering
├── lib/                      # .cpp for everything above
├── tools/nanodsp-opt/        # the compiler CLI (mlir-opt clone + our dialect/passes)
├── test/
│   ├── Dialect/DSP/          # op parsing, verification, canonicalization
│   ├── Conversion/DSPToLinalg/  # lowering structure (FileCheck)
│   └── Integration/          # end-to-end execution via mlir-runner
└── test.sh                   # configure + build + run check-nanodsp
```

Planned, not yet in the repo: `Schedule/` (TargetModel, TileSizeModel),
`schedules/` (transform-dialect schedules), `frontend/`, `benchmark/`, `docs/`.

---

## Getting started

### Prerequisites

Pin the exact revision — MLIR's transform-dialect and pass APIs churn between
releases:

```bash
# LLVM/MLIR 20.1.x, built with -DLLVM_ENABLE_PROJECTS="mlir"
git clone --branch llvmorg-20.1.0 https://github.com/llvm/llvm-project
```

You'll need `MLIR_DIR` pointing at the install, plus `lit` and `FileCheck` on
`PATH`.

### Build

```bash
cmake -G Ninja -B build \
  -DMLIR_DIR=$LLVM_INSTALL/lib/cmake/mlir \
  -DLLVM_EXTERNAL_LIT=$(which lit)
cmake --build build
```

### Test

```bash
cmake --build build --target check-nanodsp
```

This runs three tiers: dialect verification, lowering-structure checks
(FileCheck), and differential correctness tests (optimized-vs-baseline
execution via `mlir-runner`).

### Run the compiler directly

```bash
build/bin/nanodsp-opt input.mlir \
  -convert-dsp-to-linalg \
  -one-shot-bufferize="bufferize-function-boundaries" \
  -convert-linalg-to-loops -convert-vector-to-llvm -convert-func-to-llvm \
  -reconcile-unrealized-casts
```

The `-nanodsp-optimize` / `-nanodsp-apply-schedule` passes and the benchmark
sweep described above are planned (Stages 3 and 5) and not implemented yet.

---

## Project status

| Stage | Scope                                                    | Status                                                         |
| ----- | -------------------------------------------------------- | -------------------------------------------------------------- |
| 1     | Architecture + judgment calls                            | done                                                           |
| 2     | `dsp` dialect + lowering to `linalg.generic`             | done                                                           |
| 3     | Tiling + vectorization (Transform dialect schedule)      | not started                                                    |
| 4     | Bufferization + `linalg → scf → vector → LLVM`           | done (upstream passes, see `test/Integration/end-to-end.mlir`) |
| 5     | Benchmark harness                                        | not started                                                    |
| 6     | (planned) DSL frontend polish, autotuning sweep write-up | not started                                                    |

---

## Docs index

- [`docs/00-architecture.md`](docs/00-architecture.md) — full Stage 1 plan and both judgment-call tradeoffs
- [`docs/01-ir-contracts.md`](docs/01-ir-contracts.md) — the L0–L5 invariant table
- [`docs/02-tiling-model.md`](docs/02-tiling-model.md) — the working-set derivation
- [`docs/03-results.md`](docs/03-results.md) — sweep plots, model vs. measured
- [`docs/04-schedule-ir-diff.md`](docs/04-schedule-ir-diff.md) — before/after IR for one matmul
- [`docs/05-soundness.md`](docs/05-soundness.md) — why tiling/vectorization can't change results
- [`docs/06-amendments.md`](docs/06-amendments.md) — deviations from the original plan, and why
- [`docs/07-mlir-for-js-devs.md`](docs/07-mlir-for-js-devs.md) — MLIR concepts explained via JS/Babel analogies
- [`docs/architecture.excalidraw`](docs/architecture.excalidraw) — the diagram above, editable

## Known limitations

- No loop interchange yet — register tiles nest inside L1 reduction loops
  rather than outside; first knob for the autotuning stage.
- Convolution vectorization is not claimed bit-exact (channel-reduction
  reassociation); matmul and elementwise paths are.
- `l1Fraction = 0.5` is a starting estimate, not yet validated against
  hardware — that validation is the point of Stage 6.

## License

MIT (or your choice — update before publishing).
