---
name: add-dsp-op
description: Checklist for adding a new op to the dsp dialect and lowering it to linalg.
---

Rough steps — follow the existing ops (e.g. `dsp.matmul`) as the template:

1. **Define** the op in `include/nanodsp/Dialect/DSP/IR/DSPOps.td`
   (arguments, results, assembly format, `hasVerifier` if shapes need checks).
2. **Implement** verifier / canonicalizer / builders in
   `lib/Dialect/DSP/IR/DSPOps.cpp`.
3. **Lower** it in `lib/Conversion/DSPToLinalg/DSPToLinalg.cpp` as a single
   `linalg.generic` with a `tensor.empty` (plus `linalg.fill` for reductions)
   destination.
4. **Test** at all three levels:
   - `test/Dialect/DSP/` — round-trip in `ops.mlir`, bad shapes in `invalid.mlir`
   - `test/Conversion/DSPToLinalg/<op>.mlir` — FileCheck the generic's maps/iterators
   - `test/Integration/DSPToLinalg/<op>.mlir` — run and check numeric output
5. **Mirror** it outside MLIR, reusing the integration test's golden values:
   - Mojo kernel in `mojo/nanodsp/kernels.mojo` (export it from
     `__init__.mojo`), with golden, differential (odd sizes, so the SIMD tail
     runs) and error-path tests in `mojo/tests/test_kernels.mojo`
   - scalar version in `reference/nanodsp_ref.h` plus a golden check in
     `reference/test_reference.cpp`
6. Run the `build-and-test` skill, then `pixi run test-mojo` and
   `pixi run test-reference`.
