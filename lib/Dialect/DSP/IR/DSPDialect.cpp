#include "nanodsp/Dialect/DSP/IR/DSPDialect.h"
#include "nanodsp/Dialect/DSP/IR/DSPAttrs.h"
#include "nanodsp/Dialect/DSP/IR/DSPOps.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::nanodsp;

#include "nanodsp/Dialect/DSP/IR/DSPOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "nanodsp/Dialect/DSP/IR/DSPAttrs.cpp.inc"

void DSPDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "nanodsp/Dialect/DSP/IR/DSPOps.cpp.inc"
      >();
  registerAttributes();
}

void DSPDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "nanodsp/Dialect/DSP/IR/DSPAttrs.cpp.inc"
      >();
}
