add_mlir_library(NanoDSPSchedule
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
  MLIRArithDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransformOps
  MLIRParser
  MLIRPass
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTransformDialect
  MLIRTransformDialectTransforms
  MLIRUBDialect
  MLIRVectorDialect
  MLIRVectorTransformOps
)
