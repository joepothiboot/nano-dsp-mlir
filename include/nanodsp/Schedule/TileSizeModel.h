#ifndef NANODSP_SCHEDULE_TILESIZEMODEL_H
#define NANODSP_SCHEDULE_TILESIZEMODEL_H

#include "nanodsp/Schedule/TargetModel.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Support/LLVM.h"

namespace mlir::nanodsp {

struct TileSizes {
  SmallVector<int64_t> loopRanges;
  SmallVector<int64_t> cache;
  SmallVector<int64_t> reg;
  uint64_t cacheWorkingSetBytes = 0;
};

uint64_t computeWorkingSetBytes(linalg::LinalgOp op, ArrayRef<int64_t> tile);

FailureOr<TileSizes> computeTileSizes(linalg::LinalgOp op,
                                      const TargetModel &target);

}

#endif
