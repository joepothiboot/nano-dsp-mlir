#ifndef NANODSP_SCHEDULE_PASSES_H
#define NANODSP_SCHEDULE_PASSES_H

#include "nanodsp/Dialect/DSP/IR/DSPDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace nanodsp {

#define GEN_PASS_DECL
#include "nanodsp/Schedule/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "nanodsp/Schedule/Passes.h.inc"

void registerNanoDSPPipelines();

}
}

#endif
