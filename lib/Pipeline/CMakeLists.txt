add_mlir_library(NanoDSPPipeline
  Pipelines.cpp
  DEPENDS NanoDSPConversionPassIncGen
  LINK_LIBS PUBLIC
  NanoDSPConversionDSPToLinalg
  MLIRFuncDialect
  MLIRLinalgTransforms
  MLIRTransforms
  MLIRPass
)