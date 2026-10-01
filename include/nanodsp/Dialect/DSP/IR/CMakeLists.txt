add_mlir_dialect(DSPOps dsp)
add_mlir_doc(DSPOps DSPDialect Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS DSPAttrs.td)
mlir_tablegen(DSPAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=dsp)
mlir_tablegen(DSPAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=dsp)
add_public_tablegen_target(MLIRDSPAttrsIncGen)
