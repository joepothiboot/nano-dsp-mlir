# nano-dsp-mlir

Small out-of-tree MLIR compiler: a `dsp` dialect (`add`, `relu`, `matmul`,
`conv2d` on static-shape f32 tensors) lowered to `linalg.generic`, then down to
LLVM via upstream passes. See `README.md` for the full pitch and roadmap.

## Layout

- `include/nanodsp/Dialect/DSP/IR/*.td` — ODS for the dialect and ops
- `include/nanodsp/Conversion/Passes.td` — `-convert-dsp-to-linalg` pass
- `lib/` — C++ for the above (verifiers, canonicalizers, lowering patterns)
- `tools/nanodsp-opt/` — `mlir-opt`-style driver with our dialect registered
- `test/` — lit tests: `Dialect/` (parse/verify/canonicalize),
  `Conversion/` (FileCheck on lowering structure), `Integration/`
  (execute via `mlir-runner`, check numeric output)

## Build & test

```bash
./test.sh   # configure + build + check-nanodsp (auto-detects Homebrew LLVM)
```

Targets LLVM/MLIR 20.1.x. Build output goes to `build/` (gitignored).

## Conventions

- Lowering emits `linalg.generic` (not named ops) in destination-passing style;
  Stage 3 tiling relies on this uniformity.
- Every new op or lowering gets a dialect test, a conversion test, and an
  integration test that checks actual output values.
- Stages 3 (Transform-dialect tiling/vectorization) and 5 (benchmarks) are not
  implemented yet; `docs/` referenced in the README is also planned, not present.
- Format non-C++ files with Prettier (`.prettierrc.json`).
