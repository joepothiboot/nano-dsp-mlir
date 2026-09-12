#ifndef NANODSP_PIPELINE_PIPELINES_H
#define NANODSP_PIPELINE_PIPELINES_H

#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace nanodsp {

void buildNanoDSPFullPipeline(OpPassManager &pm);

} // namespace nanodsp
} // namespace mlir

#endif // NANODSP_PIPELINE_PIPELINES_H