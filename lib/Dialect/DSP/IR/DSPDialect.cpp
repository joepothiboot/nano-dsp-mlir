#include "nanodsp/Dialect/DSP/IR/DSPDialect.h"
#include "nanodsp/Dialect/DSP/IR/DSPOps.h"

using namespace mlir;
using namespace mlir::nanodsp;

#include "nanodsp/Dialect/DSP/IR/DSPOpsDialect.cpp.inc"

void DSPDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "nanodsp/Dialect/DSP/IR/DSPOps.cpp.inc"
      >();
}