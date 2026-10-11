#include "nanodsp/Target/MN1/TranslateToMN1.h"

#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

int main(int argc, char **argv) {
  mlir::nanodsp::registerToMN1Translation();

  return mlir::failed(mlir::mlirTranslateMain(
             argc, argv, "nano-dsp-mlir translation driver\n"))
             ? 1
             : 0;
}
