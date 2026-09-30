#include "nanodsp/Conversion/Passes.h"
#include "nanodsp/Dialect/DSP/IR/DSPOps.h"
#include "nanodsp/Schedule/Passes.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  // Transform-dialect ops (structured.tile_using_for, vectorize, ...) and the
  // interface implementations they rely on live in dialect extensions.
  mlir::registerAllExtensions(registry);
  mlir::registerAllPasses();

  registry.insert<mlir::nanodsp::DSPDialect>();
  mlir::nanodsp::registerNanoDSPConversionPasses();
  mlir::nanodsp::registerNanoDSPSchedulePasses();
  mlir::nanodsp::registerNanoDSPPipelines();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "nano-dsp-mlir optimizer driver\n", registry));
}