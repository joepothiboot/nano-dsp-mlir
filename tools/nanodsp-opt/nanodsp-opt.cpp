#include "nanodsp/Conversion/Passes.h"
#include "nanodsp/Dialect/DSP/IR/DSPOps.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();

  registry.insert<mlir::nanodsp::DSPDialect>();
  mlir::nanodsp::registerNanoDSPConversionPasses();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "nano-dsp-mlir optimizer driver\n", registry));
}