//===- ScheduleGen.h - Generate Transform-dialect schedules -----*- C++ -*-===//

#ifndef NANODSP_SCHEDULE_SCHEDULEGEN_H
#define NANODSP_SCHEDULE_SCHEDULEGEN_H

#include "nanodsp/Schedule/TargetModel.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Support/LLVM.h"

#include <string>

namespace mlir {
namespace nanodsp {

/// Attribute tying a schedule's `transform.structured.match` to one payload
/// op.
inline constexpr llvm::StringLiteral kScheduleTagAttr = "nanodsp.tag";

/// Unit attribute the schedule puts on the innermost loop of an op's
/// cache-tile nest, for targets with local memory. -nanodsp-promote-local
/// stages that loop's operand tiles through local memory.
inline constexpr llvm::StringLiteral kCacheLoopAttr = "nanodsp.cache_loop";

/// Tags every linalg.generic under `root` with "op0", "op1", ... in walk
/// order and returns them in that order.
SmallVector<linalg::GenericOp> tagScheduleTargets(Operation *root);

/// Removes the tags added by tagScheduleTargets.
void stripScheduleTags(Operation *root);

/// Textual `module attributes {transform.with_named_sequence}` holding a
/// `@__transform_main` that tiles (cache, then register level) and
/// vectorizes each op, with sizes from computeTileSizes. Ops the model cannot
/// handle are left alone. For a target with local memory, the innermost
/// cache-tile loop is also annotated with kCacheLoopAttr.
std::string buildDefaultSchedule(ArrayRef<linalg::GenericOp> ops,
                                 const TargetModel &target);

} // namespace nanodsp
} // namespace mlir

#endif // NANODSP_SCHEDULE_SCHEDULEGEN_H
