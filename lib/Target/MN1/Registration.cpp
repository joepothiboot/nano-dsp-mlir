#include "Translator.h"

#include "nanodsp/Dialect/DSP/IR/DSPOps.h"
#include "nanodsp/Target/MN1/TranslateToMN1.h"

#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include <string>

using namespace mlir;

LogicalResult mlir::nanodsp::translateToMN1(Operation *op,
                                            llvm::raw_ostream &os) {
  auto module = dyn_cast<ModuleOp>(op);

  if (!module)
    return op->emitError("MN1 translation runs on a module");

  std::string text;
  llvm::raw_string_ostream buffer(text);

  if (failed(mn1::Translator(buffer).translate(module)))
    return failure();

  os << text;
  return success();
}

void mlir::nanodsp::registerToMN1Translation() {
  TranslateFromMLIRRegistration registration(
      "mlir-to-mn1",
      "Translate scheduled, promoted nano-dsp IR to MN1 assembly",
      translateToMN1, [](DialectRegistry &registry) {
        registry.insert<affine::AffineDialect, arith::ArithDialect,
                        func::FuncDialect, linalg::LinalgDialect,
                        memref::MemRefDialect, scf::SCFDialect, ub::UBDialect,
                        vector::VectorDialect, DSPDialect>();
      });
}
