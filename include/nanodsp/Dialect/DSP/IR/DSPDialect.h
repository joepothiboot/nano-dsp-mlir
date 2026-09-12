
//===- DSPDialect.h - nano-dsp-mlir 'dsp' dialect ---------------*- C++ -*-===//
#ifndef NANODSP_DIALECT_DSP_IR_DSPDIALECT_H
#define NANODSP_DIALECT_DSP_IR_DSPDIALECT_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"

namespace mlir {
namespace nanodsp {

/// Shared 'valid'-convolution output-extent formula. Used by both the
/// Conv2DOp verifier/builder and the DSPToLinalg lowering so the two can never
/// disagree about what the op means.
inline int64_t computeConv2DOutputDim(int64_t inputDim, int64_t kernelDim,
                                      int64_t stride, int64_t dilation) {
  return (inputDim - (kernelDim - 1) * dilation - 1) / stride + 1;
}

} // namespace nanodsp
} // namespace mlir

#include "nanodsp/Dialect/DSP/IR/DSPOpsDialect.h.inc"

#endif // NANODSP_DIALECT_DSP_IR_DSPDIALECT_H
