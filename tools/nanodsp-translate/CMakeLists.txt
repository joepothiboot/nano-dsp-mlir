add_llvm_executable(nanodsp-translate nanodsp-translate.cpp)
llvm_update_compile_flags(nanodsp-translate)
target_link_libraries(nanodsp-translate PRIVATE
  NanoDSPTargetMN1
  MLIRTranslateLib
)
mlir_check_all_link_libraries(nanodsp-translate)
