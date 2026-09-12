set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name NanoDSPConversion)
add_public_tablegen_target(NanoDSPConversionPassIncGen)
add_mlir_doc(Passes NanoDSPConversionPasses ./ -gen-pass-doc)
