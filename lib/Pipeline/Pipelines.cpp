#include "nanodsp/Pipeline/Pipelines.h"
#include "nanodsp/Conversion/DSPToLinalg/DSPToLinalg.h"
#include "nanodsp/Schedule/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::nanodsp;

void mlir::nanodsp::buildNanoDSPFullPipeline(OpPassManager &pm) {
  NanoDSPOptimizeOptions opt;
  buildNanoDSPOptimizePipeline(pm, opt);

  pm.addPass(createConvertLinalgToLoopsPass());
}