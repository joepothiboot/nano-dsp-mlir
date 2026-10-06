#include "nanodsp/Dialect/DSP/IR/DSPAttrs.h"
#include "nanodsp/Schedule/Passes.h"
#include "nanodsp/Schedule/ScheduleGen.h"
#include "nanodsp/Schedule/TargetModel.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/CSE.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir {
namespace nanodsp {
#define GEN_PASS_DEF_NANODSPPROMOTELOCAL
#define GEN_PASS_DEF_NANODSPLOWERLOCAL
#include "nanodsp/Schedule/Passes.h.inc"
}
}

using namespace mlir;
using namespace mlir::nanodsp;

namespace {

struct DmaShape {
  int64_t numElements = 1;
  std::optional<int64_t> stride;
  int64_t eltsPerStride = 0;
};

std::optional<DmaShape> getDmaShape(ArrayRef<int64_t> tileShape,
                                    ArrayRef<int64_t> strides) {
  DmaShape dma;
  int64_t run = 1, rows = 0, rowStride = 0;

  for (int64_t d = tileShape.size() - 1; d >= 0; --d) {
    int64_t extent = tileShape[d], stride = strides[d];

    if (extent == 1)
      continue;

    if (ShapedType::isDynamic(stride))
      return std::nullopt;

    if (rows == 0 && stride == run) {
      run *= extent;
    } else if (rows == 0) {
      rows = extent;
      rowStride = stride;
    } else if (stride == rows * rowStride) {
      rows *= extent;
    } else {
      return std::nullopt;
    }
  }

  dma.numElements = run * std::max<int64_t>(rows, 1);

  if (rows != 0) {
    dma.stride = rowStride;
    dma.eltsPerStride = run;
  }

  return dma;
}

bool isReadOnlyView(Value view) {
  for (Operation *user : view.getUsers()) {
    if (auto subview = dyn_cast<memref::SubViewOp>(user)) {
      if (!isReadOnlyView(subview.getResult()))
        return false;

      continue;
    }

    if (isa<vector::TransferReadOp, memref::LoadOp>(user))
      continue;

    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(user)) {
      bool onlyInput =
          llvm::all_of(linalgOp->getOpOperands(), [&](OpOperand &operand) {
            return operand.get() != view || linalgOp.isDpsInput(&operand);
          });
      if (onlyInput)
        continue;
    }

    return false;
  }

  return true;
}

FailureOr<SmallVector<Operation *>> getLoopLocalSlice(Operation *op,
                                                      scf::ForOp loop) {
  BackwardSliceOptions options;
  options.omitBlockArguments = true;
  options.filter = [&](Operation *o) {
    return o->getBlock() == loop.getBody();
  };

  SetVector<Operation *> slice;

  if (failed(getBackwardSlice(op, &slice, options)))
    return failure();

  SmallVector<Operation *> ops(slice.begin(), slice.end());

  if (!llvm::all_of(ops, isPure))
    return failure();

  llvm::sort(ops,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  return ops;
}

bool dependsOn(Operation *op, ArrayRef<Operation *> slice, Value value) {
  auto reads = [&](Operation *o) {
    return llvm::is_contained(o->getOperands(), value);
  };

  return reads(op) || llvm::any_of(slice, reads);
}

struct TilePlan {
  memref::SubViewOp tile;
  SmallVector<int64_t> shape;
  DmaShape dma;
  bool varies = false;
  uint64_t bytes = 0;
};

SmallVector<TilePlan> planTiles(scf::ForOp loop) {
  SmallVector<TilePlan> plans;

  for (Operation &op : loop.getBody()->without_terminator()) {
    auto tile = dyn_cast<memref::SubViewOp>(&op);

    if (!tile || !loop.isDefinedOutsideOfLoop(tile.getSource()) ||
        !tile.hasUnitStride())
      continue;

    MemRefType srcType = tile.getSourceType();
    MemRefType tileType = tile.getType();

    if (tileType.getRank() != srcType.getRank() || !tileType.hasStaticShape() ||
        !tileType.getElementType().isIntOrFloat())
      continue;

    if (!isReadOnlyView(tile.getResult()))
      continue;

    SmallVector<int64_t> strides;
    int64_t offset;

    if (failed(srcType.getStridesAndOffset(strides, offset)))
      continue;

    std::optional<DmaShape> dma = getDmaShape(tileType.getShape(), strides);

    if (!dma)
      continue;

    FailureOr<SmallVector<Operation *>> slice = getLoopLocalSlice(tile, loop);

    if (failed(slice))
      continue;

    TilePlan plan;
    plan.tile = tile;
    plan.shape = llvm::to_vector(tileType.getShape());
    plan.dma = *dma;
    plan.varies = dependsOn(tile, *slice, loop.getInductionVar());
    plan.bytes = tileType.getNumElements() *
                 llvm::divideCeil(tileType.getElementTypeBitWidth(), 8);
    plans.push_back(std::move(plan));
  }

  return plans;
}

void replaceAndRetype(RewriterBase &rewriter, Value oldValue, Value newValue) {
  for (OpOperand &use : llvm::make_early_inc_range(oldValue.getUses())) {
    Operation *user = use.getOwner();
    auto subview = dyn_cast<memref::SubViewOp>(user);

    if (!subview) {
      rewriter.modifyOpInPlace(user, [&] { use.set(newValue); });
      continue;
    }

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(subview);
    MemRefType type = memref::SubViewOp::inferRankReducedResultType(
        subview.getType().getShape(), cast<MemRefType>(newValue.getType()),
        subview.getMixedOffsets(), subview.getMixedSizes(),
        subview.getMixedStrides());
    auto retyped = memref::SubViewOp::create(
        rewriter, subview.getLoc(), type, newValue, subview.getMixedOffsets(),
        subview.getMixedSizes(), subview.getMixedStrides());
    replaceAndRetype(rewriter, subview.getResult(), retyped.getResult());
    rewriter.eraseOp(subview);
  }
}

struct PromotedTile {
  memref::AllocOp buffer;
  memref::AllocOp tag;
  memref::DmaStartOp start;
  memref::DmaWaitOp wait;
};

PromotedTile promoteTile(RewriterBase &rewriter, const TilePlan &plan,
                         Operation *nest, const TargetModel &target) {
  MLIRContext *ctx = rewriter.getContext();
  memref::SubViewOp tile = plan.tile;
  Location loc = tile.getLoc();

  PromotedTile promoted;
  rewriter.setInsertionPoint(nest);
  auto bufferType = MemRefType::get(plan.shape, tile.getType().getElementType(),
                                    MemRefLayoutAttrInterface(),
                                    LocalMemorySpaceAttr::get(ctx));
  promoted.buffer = memref::AllocOp::create(
      rewriter, loc, bufferType,
      rewriter.getI64IntegerAttr(target.vectorBits / 8));
  promoted.tag = memref::AllocOp::create(
      rewriter, loc, MemRefType::get({1}, rewriter.getI32Type()));

  rewriter.setInsertionPointAfter(tile);
  SmallVector<Value> srcIndices =
      getValueOrCreateConstantIndexOp(rewriter, loc, tile.getMixedOffsets());
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  SmallVector<Value> dstIndices(plan.shape.size(), zero);
  Value numElements =
      arith::ConstantIndexOp::create(rewriter, loc, plan.dma.numElements);
  Value stride, eltsPerStride;

  if (plan.dma.stride) {
    stride = arith::ConstantIndexOp::create(rewriter, loc, *plan.dma.stride);
    eltsPerStride =
        arith::ConstantIndexOp::create(rewriter, loc, plan.dma.eltsPerStride);
  }

  promoted.start = memref::DmaStartOp::create(
      rewriter, loc, tile.getSource(), srcIndices, promoted.buffer, dstIndices,
      numElements, promoted.tag, ValueRange{zero}, stride, eltsPerStride);
  promoted.wait = memref::DmaWaitOp::create(rewriter, loc, promoted.tag,
                                            ValueRange{zero}, numElements);

  replaceAndRetype(rewriter, tile.getResult(), promoted.buffer.getResult());
  rewriter.eraseOp(tile);

  return promoted;
}

void eraseDeadDefs(RewriterBase &rewriter, SmallVector<Operation *> worklist) {
  llvm::SmallPtrSet<Operation *, 8> erased;
  auto erase = [&](Operation *op) {
    erased.insert(op);
    rewriter.eraseOp(op);
  };

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();

    if (!op || erased.contains(op))
      continue;

    bool deadAlloc = isa<memref::AllocOp>(op) &&
                     llvm::all_of(op->getUsers(), [](Operation *u) {
                       return isa<memref::DeallocOp>(u);
                     });
    if (!deadAlloc && !isOpTriviallyDead(op))
      continue;

    for (Value operand : op->getOperands())
      worklist.push_back(operand.getDefiningOp());

    if (deadAlloc)
      for (Operation *user : llvm::make_early_inc_range(op->getUsers()))
        erase(user);

    erase(op);
  }
}

LogicalResult hoistInvariantDma(RewriterBase &rewriter, scf::ForOp loop,
                                PromotedTile &promoted) {
  FailureOr<SmallVector<Operation *>> slice =
      getLoopLocalSlice(promoted.start, loop);
  if (failed(slice))
    return failure();

  rewriter.setInsertionPoint(loop);
  IRMapping mapping;

  for (Operation *op : *slice)
    rewriter.clone(*op, mapping);

  rewriter.clone(*promoted.start, mapping);
  rewriter.clone(*promoted.wait, mapping);
  SmallVector<Operation *> defs;

  for (Operation *op :
       {promoted.wait.getOperation(), promoted.start.getOperation()}) {
    for (Value operand : op->getOperands())
      defs.push_back(operand.getDefiningOp());

    rewriter.eraseOp(op);
  }

  eraseDeadDefs(rewriter, defs);

  return success();
}

LogicalResult multiBufferTile(RewriterBase &rewriter, PromotedTile &promoted) {
  FailureOr<memref::AllocOp> buffer =
      memref::multiBuffer(rewriter, promoted.buffer, 2, true);
  if (failed(buffer))
    return failure();

  promoted.buffer = *buffer;
  FailureOr<memref::AllocOp> tag =
      memref::multiBuffer(rewriter, promoted.tag, 2, true);
  if (failed(tag))
    return failure();

  promoted.tag = *tag;

  return success();
}

LogicalResult pipelineDmas(RewriterBase &rewriter, scf::ForOp loop,
                           MutableArrayRef<PromotedTile> tiles) {
  if (tiles.empty())
    return success();

  SmallVector<SmallVector<Operation *>> slices;

  for (PromotedTile &tile : tiles) {
    FailureOr<SmallVector<Operation *>> slice =
        getLoopLocalSlice(tile.start, loop);
    if (failed(slice))
      return failure();

    slices.push_back(std::move(*slice));
  }

  Value iv = loop.getInductionVar();
  auto cloneDmasFor = [&](Value iteration) {
    for (auto [tile, slice] : llvm::zip_equal(tiles, slices)) {
      IRMapping mapping;
      mapping.map(iv, iteration);

      for (Operation *op : slice) {
        Operation *clone = rewriter.clone(*op, mapping);
        SmallVector<Value> folded;

        if (succeeded(rewriter.tryFold(clone, folded)) && !folded.empty()) {
          mapping.map(op->getResults(), folded);
          rewriter.eraseOp(clone);
        }
      }

      rewriter.clone(*tile.start, mapping);
    }
  };

  rewriter.setInsertionPoint(loop);
  cloneDmasFor(loop.getLowerBound());

  Operation *firstStart = tiles.front().start;

  for (PromotedTile &tile : tiles)
    if (tile.start->isBeforeInBlock(firstStart))
      firstStart = tile.start;

  Location loc = loop.getLoc();
  rewriter.setInsertionPoint(firstStart);
  Value next = arith::AddIOp::create(rewriter, loc, iv, loop.getStep());
  Value hasNext = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::slt, next, loop.getUpperBound());
  auto prefetch = scf::IfOp::create(rewriter, loc, hasNext, false);
  rewriter.setInsertionPointToStart(prefetch.thenBlock());
  cloneDmasFor(next);

  SmallVector<Operation *> defs;

  for (PromotedTile &tile : tiles) {
    for (Value operand : tile.start->getOperands())
      defs.push_back(operand.getDefiningOp());

    rewriter.eraseOp(tile.start);
    tile.start = nullptr;
  }

  eraseDeadDefs(rewriter, defs);

  return success();
}

Operation *getOutermostLoop(scf::ForOp loop) {
  Operation *outer = loop;

  while (auto parent = dyn_cast<scf::ForOp>(outer->getParentOp()))
    outer = parent;

  return outer;
}

struct NanoDSPPromoteLocalPass
    : public mlir::nanodsp::impl::NanoDSPPromoteLocalBase<
          NanoDSPPromoteLocalPass> {
  using NanoDSPPromoteLocalBase::NanoDSPPromoteLocalBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::optional<TargetModel> target = lookupTarget(targetName, module);

    if (!target)
      return signalPassFailure();

    SmallVector<scf::ForOp> loops;
    module.walk([&](scf::ForOp loop) {
      if (loop->hasAttr(kCacheLoopAttr))
        loops.push_back(loop);
    });

    if (target->localMemBytes == 0)
      return;

    IRRewriter rewriter(&getContext());

    for (scf::ForOp loop : loops)
      if (failed(promoteLoop(rewriter, loop, *target)))
        return signalPassFailure();

    DominanceInfo domInfo;
    eliminateCommonSubExpressions(rewriter, domInfo, module);
  }

  LogicalResult promoteLoop(RewriterBase &rewriter, scf::ForOp loop,
                            const TargetModel &target) {
    SmallVector<TilePlan> plans = planTiles(loop);

    if (plans.empty())
      return success();

    auto requiredBytes = [&](bool doubled) {
      uint64_t bytes = 0;

      for (const TilePlan &plan : plans)
        bytes += plan.bytes * (doubled && plan.varies ? 2 : 1);

      return bytes;
    };

    bool doubled = doubleBuffer && requiredBytes(true) <= target.localMemBytes;

    if (requiredBytes(false) > target.localMemBytes)
      return loop.emitError()
             << "input tiles of this cache tile need " << requiredBytes(false)
             << " bytes of local memory, but target '" << target.name
             << "' has " << target.localMemBytes;

    Operation *nest = getOutermostLoop(loop);
    SmallVector<PromotedTile> promotedTiles, pipelined;

    for (const TilePlan &plan : plans) {
      PromotedTile promoted = promoteTile(rewriter, plan, nest, target);

      if (!plan.varies && failed(hoistInvariantDma(rewriter, loop, promoted)))
        return loop.emitError("could not hoist a loop-invariant tile DMA");

      if (plan.varies && doubled) {
        if (failed(multiBufferTile(rewriter, promoted)))
          return loop.emitError("could not double-buffer a tile");

        pipelined.push_back(promoted);
      }

      promotedTiles.push_back(promoted);
    }

    if (failed(pipelineDmas(rewriter, loop, pipelined)))
      return loop.emitError("could not pipeline the tile DMAs");

    rewriter.setInsertionPointAfter(nest);

    for (PromotedTile &promoted : promotedTiles) {
      memref::DeallocOp::create(rewriter, loop.getLoc(), promoted.buffer);
      memref::DeallocOp::create(rewriter, loop.getLoc(), promoted.tag);
    }

    return success();
  }
};

LogicalResult lowerDmaStart(RewriterBase &rewriter, memref::DmaStartOp dma) {
  auto dstType = dyn_cast<MemRefType>(dma.getDstMemRef().getType());
  auto srcType = cast<MemRefType>(dma.getSrcMemRef().getType());
  std::optional<int64_t> numElements =
      getConstantIntValue(dma.getNumElements());
  bool zeroDst = llvm::all_of(dma.getDstIndices(),
                              [](Value v) { return isConstantIntValue(v, 0); });
  if (!dstType || !dstType.hasStaticShape() ||
      dstType.getRank() != srcType.getRank() || !zeroDst || !numElements ||
      *numElements != dstType.getNumElements())
    return dma.emitOpError(
        "can only be lowered when it fills a whole statically shaped "
        "destination of the source's rank, starting at index 0");

  rewriter.setInsertionPoint(dma);
  Location loc = dma.getLoc();
  SmallVector<OpFoldResult> offsets = getAsOpFoldResult(dma.getSrcIndices());
  SmallVector<OpFoldResult> sizes =
      getAsIndexOpFoldResult(rewriter.getContext(), dstType.getShape());
  SmallVector<OpFoldResult> strides(dstType.getRank(),
                                    rewriter.getIndexAttr(1));
  Value src = memref::SubViewOp::create(rewriter, loc, dma.getSrcMemRef(),
                                        offsets, sizes, strides);
  linalg::CopyOp::create(rewriter, loc, ValueRange{src},
                         ValueRange{dma.getDstMemRef()});
  SmallVector<Operation *> defs;

  for (Value operand : dma->getOperands())
    defs.push_back(operand.getDefiningOp());

  rewriter.eraseOp(dma);
  eraseDeadDefs(rewriter, defs);

  return success();
}

struct NanoDSPLowerLocalPass
    : public mlir::nanodsp::impl::NanoDSPLowerLocalBase<NanoDSPLowerLocalPass> {
  using NanoDSPLowerLocalBase::NanoDSPLowerLocalBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    IRRewriter rewriter(&getContext());

    SmallVector<memref::DmaWaitOp> waits;
    module.walk([&](memref::DmaWaitOp op) { waits.push_back(op); });

    for (memref::DmaWaitOp wait : waits) {
      SmallVector<Operation *> defs;

      for (Value operand : wait->getOperands())
        defs.push_back(operand.getDefiningOp());

      rewriter.eraseOp(wait);
      eraseDeadDefs(rewriter, defs);
    }

    SmallVector<memref::DmaStartOp> starts;
    module.walk([&](memref::DmaStartOp op) { starts.push_back(op); });

    for (memref::DmaStartOp start : starts)
      if (failed(lowerDmaStart(rewriter, start)))
        return signalPassFailure();

    AttrTypeReplacer replacer;
    replacer.addReplacement([](MemRefType type) -> std::optional<Type> {
      if (!isa_and_nonnull<LocalMemorySpaceAttr>(type.getMemorySpace()))
        return std::nullopt;

      return MemRefType::get(type.getShape(), type.getElementType(),
                             type.getLayout());
    });
    replacer.recursivelyReplaceElementsIn(module, true, false, true);
  }
};

}
