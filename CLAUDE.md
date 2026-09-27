# nano-dsp-mlir

Small out-of-tree MLIR compiler: a `dsp` dialect (`add`, `relu`, `matmul`,
`conv2d` on static-shape f32 tensors) lowered to `linalg.generic`, then down to
LLVM via upstream passes, plus a SIMD Mojo library implementing the same four
ops. See `README.md` for the full pitch and roadmap.

## Layout

- `include/nanodsp/Dialect/DSP/IR/*.td` — ODS for the dialect and ops
- `include/nanodsp/Conversion/Passes.td` — `-convert-dsp-to-linalg` pass
- `lib/` — C++ for the above (verifiers, canonicalizers, lowering patterns)
- `tools/nanodsp-opt/` — `mlir-opt`-style driver with our dialect registered
- `test/` — lit tests: `Dialect/` (parse/verify/canonicalize),
  `Conversion/` (FileCheck on lowering structure), `Integration/`
  (execute via `mlir-runner`, check numeric output)
- `mojo/nanodsp/` — Mojo package: `Tensor[dtype]` + SIMD `add`/`relu`/
  `matmul`/`conv2d` with the same semantics as the dialect ops
- `mojo/tests/` — golden (same values as `test/Integration/`) and
  differential (SIMD vs naive loop nest) tests
- `reference/` — header-only scalar C++ oracle + golden-value test
- `benchmarks/` — Mojo kernel throughput; MLIR/C++ comparison planned
- `docs/` — design notes (most files in the README index are still planned)

## Build & test

```bash
./test.sh   # configure + build + check-nanodsp (auto-detects Homebrew LLVM)
```

Targets LLVM/MLIR 20.1.x. Build output goes to `build/` (gitignored).

```bash
pixi run test-mojo       # Mojo kernel tests (installs Mojo 1.1 via pixi)
pixi run test-reference  # C++ reference golden tests
pixi run bench           # Mojo kernel benchmarks
```

## Conventions

- Lowering emits `linalg.generic` (not named ops) in destination-passing style;
  Stage 3 tiling relies on this uniformity.
- Every new op or lowering gets a dialect test, a conversion test, and an
  integration test that checks actual output values. New ops also get a Mojo
  kernel, a C++ reference, and the same golden values in all three places.
- Kernels keep multiply and add unfused and the reduction order of the naive
  loop nest, so they can be compared bit-exactly; the C++ reference is built
  with `-ffp-contract=off` for the same reason.
- Stage 3 (Transform-dialect tiling/vectorization) is not implemented yet, and
  the Stage 5 MLIR-vs-Mojo-vs-C++ benchmark comparison is planned. Most of the
  `docs/` files in the README index are planned, not present.
- Mojo is pinned to 1.1 in `pixi.toml`: `def` only (no `fn`), `comptime` not
  `alias`, stdlib imports as `std.*`, struct parameters as `Self.dtype`, and
  `unsafe_load`/`unsafe_store`/`unsafe_offset` on pointers. Keep builds free of
  deprecation warnings.
- Format non-C++ files with Prettier (`.prettierrc.json`).
