add_mlir_library(NanoDSPSchedule
  LocalMemory.cpp
  Passes.cpp
  ScheduleGen.cpp
  TargetModel.cpp
  TileSizeModel.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/nanodsp/Schedule

  DEPENDS
  NanoDSPSchedulePassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAnalysis
  MLIRArithDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransformOps
  MLIRMemRefDialect
  MLIRMemRefTransforms
  MLIRParser
  MLIRPass
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTransformDialect
  MLIRTransformDialectTransforms
  MLIRTransforms
  MLIRUBDialect
  MLIRVectorDialect
  MLIRVectorTransformOps
  NanoDSPDialect
)
