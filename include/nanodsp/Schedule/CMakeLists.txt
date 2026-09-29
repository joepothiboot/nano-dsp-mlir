set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name NanoDSPSchedule)
add_public_tablegen_target(NanoDSPSchedulePassIncGen)
add_mlir_doc(Passes NanoDSPSchedulePasses ./ -gen-pass-doc)
