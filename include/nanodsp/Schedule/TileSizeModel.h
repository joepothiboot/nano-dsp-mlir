//===- TileSizeModel.h - Analytical tile-size selection ---------*- C++ -*-===//
//
// Two tiling levels per structured op, both derived from a TargetModel:
//
//   * register tile: what one vectorized step computes. Reduction dims are 1,
//     so accumulation follows the naive loop nest's order and scheduled code
//     stays bit-exact with the unscheduled lowering.
//   * cache tile: the register tile grown until the tile's working set would
//     exceed TargetModel::tileBudgetBytes().
//
// Tile sizes always divide the loop extents, so every tile has a static shape
// and vectorization needs no masking.
//
//===----------------------------------------------------------------------===//

#ifndef NANODSP_SCHEDULE_TILESIZEMODEL_H
#define NANODSP_SCHEDULE_TILESIZEMODEL_H

#include "nanodsp/Schedule/TargetModel.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace nanodsp {

struct TileSizes {
  /// Loop extents of the op, one per iterator.
  SmallVector<int64_t> loopRanges;
  /// Absolute extents of one cache tile / one register tile per iterator.
  SmallVector<int64_t> cache;
  SmallVector<int64_t> reg;
  /// Bytes touched by one cache tile, summed over all operands.
  uint64_t cacheWorkingSetBytes = 0;
};

/// Bytes of all operands touched by a tile with the given per-iterator
/// extents. Exposed for tests and docs.
uint64_t computeWorkingSetBytes(linalg::LinalgOp op, ArrayRef<int64_t> tile);

/// Fails if the op has dynamic loop ranges or a non-int/float element type.
FailureOr<TileSizes> computeTileSizes(linalg::LinalgOp op,
                                      const TargetModel &target);

} // namespace nanodsp
} // namespace mlir

#endif // NANODSP_SCHEDULE_TILESIZEMODEL_H
