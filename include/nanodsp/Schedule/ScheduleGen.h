#ifndef NANODSP_SCHEDULE_SCHEDULEGEN_H
#define NANODSP_SCHEDULE_SCHEDULEGEN_H

#include "nanodsp/Schedule/TargetModel.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Support/LLVM.h"

#include <string>

namespace mlir::nanodsp {

inline constexpr llvm::StringLiteral kScheduleTagAttr = "nanodsp.tag";

inline constexpr llvm::StringLiteral kCacheLoopAttr = "nanodsp.cache_loop";

SmallVector<linalg::GenericOp> tagScheduleTargets(Operation *root);

void stripScheduleTags(Operation *root);

std::string buildDefaultSchedule(ArrayRef<linalg::GenericOp> ops,
                                 const TargetModel &target);

}

#endif
