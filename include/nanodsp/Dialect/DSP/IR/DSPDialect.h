#ifndef NANODSP_DIALECT_DSP_IR_DSPDIALECT_H
#define NANODSP_DIALECT_DSP_IR_DSPDIALECT_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"

namespace mlir {
namespace nanodsp {

inline int64_t computeConv2DOutputDim(int64_t inputDim, int64_t kernelDim,
                                      int64_t stride, int64_t dilation) {
  return (inputDim - (kernelDim - 1) * dilation - 1) / stride + 1;
}

}
}

#include "nanodsp/Dialect/DSP/IR/DSPOpsDialect.h.inc"

#endif
