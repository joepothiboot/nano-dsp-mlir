#ifndef NANODSP_SCHEDULE_PASSES_H
#define NANODSP_SCHEDULE_PASSES_H

#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace nanodsp {

struct NanoDSPOptimizeOptions {};

inline void buildNanoDSPOptimizePipeline(OpPassManager &,
                                         const NanoDSPOptimizeOptions &) {}

} // namespace nanodsp
} // namespace mlir

#endif // NANODSP_SCHEDULE_PASSES_H