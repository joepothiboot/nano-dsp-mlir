add_mlir_library(NanoDSPTargetMN1
  Control.cpp
  Memory.cpp
  RegisterPool.cpp
  Registration.cpp
  Scalars.cpp
  Translator.cpp
  Vectors.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/nanodsp/Target/MN1

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTranslateLib
  MLIRUBDialect
  MLIRVectorDialect
  NanoDSPDialect
)
