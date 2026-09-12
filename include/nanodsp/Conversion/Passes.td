#ifndef NANODSP_CONVERSION_PASSES_TD
#define NANODSP_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertDSPToLinalg : Pass<"convert-dsp-to-linalg"> {
  let summary = "Lower the 'dsp' dialect to linalg.generic over tensors (L1 -> L2).";
  let description = [{
    Full conversion of the `dsp` dialect. Every op becomes a single
    `linalg.generic` in destination-passing style with an explicit
    `tensor.empty` (and `linalg.fill` for reductions) destination.

    Deliberately emits `linalg.generic` and not named ops: the Stage 3
    transform schedule tiles a uniform op class. Use
    `-linalg-specialize-generic-ops` to raise back to named ops if a library
    call path is ever wanted.
  }];

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::tensor::TensorDialect"
  ];
}

#endif // NANODSP_CONVERSION_PASSES_TD