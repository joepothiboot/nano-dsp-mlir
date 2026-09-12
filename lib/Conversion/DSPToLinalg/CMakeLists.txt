add_mlir_library(NanoDSPConversionDSPToLinalg
  DSPToLinalg.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/nanodsp/Conversion

  DEPENDS
  NanoDSPConversionPassIncGen

  LINK_LIBS PUBLIC
  NanoDSPDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
)