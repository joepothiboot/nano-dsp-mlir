#ifndef NANODSP_DIALECT_DSP_IR_DSPOPS_H
#define NANODSP_DIALECT_DSP_IR_DSPOPS_H

#include "nanodsp/Dialect/DSP/IR/DSPDialect.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "nanodsp/Dialect/DSP/IR/DSPOps.h.inc"

#endif // NANODSP_DIALECT_DSP_IR_DSPOPS_H