# nano-dsp-mlir

Small out-of-tree MLIR compiler: a `dsp` dialect (`add`, `relu`, `matmul`,
`conv2d` on static-shape f32 tensors, plus the int8 `qmatmul`) lowered to
`linalg.generic`, then down to LLVM via upstream passes, plus a SIMD Mojo
library implementing the same ops. See `README.md` for the full pitch and roadmap.

## Layout

- `include/nanodsp/Dialect/DSP/IR/*.td` — ODS for the dialect and ops
- `include/nanodsp/Dialect/DSP/IR/DSPConstants.h` — quantization limits shared
  by the `qmatmul` verifier and its lowering
- `include/nanodsp/Conversion/Passes.td` — `-convert-dsp-to-linalg` pass
- `include/nanodsp/Schedule/` + `lib/Schedule/` — Stage 3: `TargetModel`,
  `TileSizeModel`, schedule generation, `-nanodsp-optimize`,
  `-nanodsp-emit-schedule`, and the `-nanodsp-lower-to-llvm` pipeline
  (= `-nanodsp-bufferize` + `-nanodsp-lower-bufferized-to-llvm`);
  `LocalMemory.cpp` holds `-nanodsp-promote-local` / `-nanodsp-lower-local`
  (VTCM tiles in `#dsp.local`, double-buffered DMA; see
  `docs/scratchpad-dma.md`)
- `schedules/` — hand-written Transform-dialect schedules
- `lib/` — C++ for the above (verifiers, canonicalizers, lowering patterns)
- `tools/nanodsp-opt/` — `mlir-opt`-style driver with our dialect registered
- `test/` — lit tests: `Dialect/` (parse/verify/canonicalize),
  `Conversion/` (FileCheck on lowering structure), `Schedule/` (generated
  schedules, scheduled IR), `Integration/` (execute via `mlir-runner`, check
  numeric output; `Integration/Schedule/bit-exact.mlir` diffs scheduled vs
  unscheduled results bit for bit)
- `mojo/nanodsp/` — Mojo package: `Tensor[dtype]` + SIMD `add`/`relu`/
  `matmul`/`conv2d` (`kernels.mojo`) and `qmatmul` (`quant.mojo`) with the
  same semantics as the dialect ops; `TensorLike`/`TensorView`
  (`layout.mojo`) and the generic `matmul_tiled` (see
  `docs/mojo-api-design.md`); `gpu.mojo` runs `matmul` (naive, tiled,
  blocked) and `conv2d` on a GPU, bit-exact with the CPU kernels (see
  `docs/mojo-gpu.md`). It is not re-exported from the package root, so the
  CPU kernels don't need `max-core`
- `mojo/nanodsp/constants.mojo` — int8 limits and GPU tile sizes
- `mojo/tests/` — golden (same values as `test/Integration/`) and
  differential (SIMD vs naive loop nest) tests
- `reference/` — header-only scalar C++ oracle + golden-value test
- `cuda/` — CUDA matmuls (naive, tiled, blocked, vec) and the T4 harness
  (`bench.cu`, cuBLAS in the same harness); `triton/matmul.py` — Triton
  matmul; `scripts/run-t4.sh` runs both plus Mojo and `ncu` on Colab (see
  `docs/07-cuda-triton.md`). CUDA kernels stay bit-exact via
  `__fmul_rn`/`__fadd_rn`; constants in `cuda/constants.h`
- `docker/hexagon/`, `scripts/run-hexagon.sh`, `test/Hexagon/` — emulated Hexagon V68
  (HVX) run of the kernels, bit-checked against `reference/`; needs Docker and
  is not part of `check-nanodsp` (see `docs/hexagon-target.md`)
- `benchmarks/` — Mojo, MLIR and C++ benchmarks, ceilings, raw results in
  `benchmarks/results/` (see `docs/03-results.md`, `docs/06-gpu-results.md`)
- `docs/` — design notes (the README index says which exist)

## Build & test

```bash
./test.sh   # configure + build + check-nanodsp (auto-detects Homebrew LLVM)
```

Targets LLVM/MLIR 21+ (tested on Homebrew 23.1.1); use `Op::create(builder,
...)`, not the deprecated `builder.create<Op>(...)`. Build output goes to
`build/` (gitignored). `test.sh` works around lit's unquoted paths when the
checkout lives under a directory containing spaces.

```bash
pixi run test-mojo       # Mojo kernel tests (installs Mojo 1.1 via pixi)
pixi run test-reference  # C++ reference golden tests
pixi run bench           # Mojo kernel benchmarks
pixi run test-gpu        # GPU kernels (needs a GPU; not in test-mojo or CI)
pixi run bench-gpu       # GPU benchmark -> build/bench/results-gpu.json
```

## Conventions

- Lowering emits `linalg.generic` (not named ops) in destination-passing style;
  Stage 3 tiling relies on this uniformity.
- Every new op or lowering gets a dialect test, a conversion test, and an
  integration test that checks actual output values. New ops also get a Mojo
  kernel, a C++ reference, and the same golden values in all three places.
- Kernels keep multiply and add unfused and the reduction order of the naive
  loop nest, so they can be compared bit-exactly; the C++ reference is built
  with `-ffp-contract=off` and Mojo with `--fp-mode contract=off` (Mojo's
  default fuses into FMA) for the same reason. Mojo kernels tile only output
  dims, never the reduction. GPU kernels may split the reduction into
  consecutive chunks walked in order (shared-memory tiles), never reorder it.
- Stage 3 schedules must stay bit-exact: register tiles keep reduction dims at
  1 and vectorize to separate `mulf`/`addf` (no `vector.contract`, no FMA).
  Any schedule change must keep `test/Integration/Schedule/bit-exact.mlir`
  passing. Tile sizes divide loop extents (no masking yet).
- Mojo is pinned to 1.1 in `pixi.toml`: `def` only (no `fn`), `comptime` not
  `alias`, stdlib imports as `std.*`, struct parameters as `Self.dtype`, and
  `unsafe_load`/`unsafe_store`/`unsafe_offset` on pointers. Keep builds free of
  deprecation warnings. GPU APIs come from the `max-core` package
  (`from max.gpu import ...`, `from max.gpu.host import DeviceContext`);
  kernel arguments must be fixed-width (`Int32`, not `Int`).
- `dsp.qmatmul` requantizes in integer arithmetic only, rounding half up
  (TFLite single rounding); see `docs/quantization.md`. ODS `I32Attr`
  accessors return `uint32_t`: sign-extend explicitly when reading signed
  attributes.
- The C++ reference builds as C++20 (arithmetic `>>` on negative integers is
  defined there).
- Format C++ with clang-format (`.clang-format`), Mojo with `mojo format`, and
  other files with Prettier (`.prettierrc.json`).
- No comments in code (lit `RUN`/`CHECK` lines and pragmas excepted); explain
  design in `docs/`. Leave a blank line between logical blocks, avoid nested
  ternaries, keep files under ~500 lines, and put shared values in the
  constants files above.
