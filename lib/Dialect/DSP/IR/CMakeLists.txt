add_mlir_dialect_library(NanoDSPDialect
  DSPDialect.cpp
  DSPOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/nanodsp/Dialect/DSP/IR

  DEPENDS
  MLIRDSPOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRSideEffectInterfaces
)