# tools/nanodsp-opt/CMakeLists.txt
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_llvm_executable(nanodsp-opt nanodsp-opt.cpp)
llvm_update_compile_flags(nanodsp-opt)
target_link_libraries(nanodsp-opt PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  NanoDSPDialect
  NanoDSPConversionDSPToLinalg
  NanoDSPPipeline
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIROptLib
)
mlir_check_all_link_libraries(nanodsp-opt)