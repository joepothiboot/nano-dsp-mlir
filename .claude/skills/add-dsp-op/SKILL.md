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
5. Run the `build-and-test` skill.
