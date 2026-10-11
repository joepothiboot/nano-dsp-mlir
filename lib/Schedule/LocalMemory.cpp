#include "LocalMemoryUtils.h"

#include "nanodsp/Dialect/DSP/IR/DSPAttrs.h"
#include "nanodsp/Schedule/Passes.h"
#include "nanodsp/Schedule/ScheduleGen.h"
#include "nanodsp/Schedule/TargetModel.h"

#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/CSE.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::nanodsp {
#define GEN_PASS_DEF_NANODSPPROMOTELOCAL
#define GEN_PASS_DEF_NANODSPLOWERLOCAL
#include "nanodsp/Schedule/Passes.h.inc"
}

using namespace mlir;
using namespace mlir::nanodsp;
using namespace mlir::nanodsp::local;

namespace {

struct TilePlan {
  memref::SubViewOp tile;
  SmallVector<int64_t> shape;
  DmaShape dma;
  Access access = Access::Read;
  bool varies = false;
  uint64_t bytes = 0;
};

SmallVector<TilePlan> planTiles(scf::ForOp loop, bool outputs) {
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

    std::optional<Access> access = classifyView(tile.getResult());

    if (!access || *access == Access::None)
      continue;

    if (*access != Access::Read && !outputs)
      continue;

    SmallVector<int64_t> strides;
    int64_t offset = 0;

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
    plan.access = *access;
    plan.varies = dependsOn(tile, *slice, loop.getInductionVar());
    plan.bytes = tileType.getNumElements() *
                 llvm::divideCeil(tileType.getElementTypeBitWidth(), 8);
    plans.push_back(std::move(plan));
  }

  return plans;
}

struct PromotedTile {
  memref::AllocOp buffer;
  memref::AllocOp tag;
  memref::DmaStartOp start;
  memref::DmaWaitOp wait;
  memref::DmaStartOp store;
  memref::DmaWaitOp storeWait;
};

memref::AllocOp allocLocal(RewriterBase &rewriter, Location loc,
                           ArrayRef<int64_t> shape, Type elementType,
                           const TargetModel &target) {
  auto type = MemRefType::get(shape, elementType, MemRefLayoutAttrInterface(),
                              LocalMemorySpaceAttr::get(rewriter.getContext()));
  return memref::AllocOp::create(
      rewriter, loc, type, rewriter.getI64IntegerAttr(target.vectorBits / 8));
}

memref::AllocOp allocTag(RewriterBase &rewriter, Location loc) {
  return memref::AllocOp::create(rewriter, loc,
                                 MemRefType::get({1}, rewriter.getI32Type()));
}

PromotedTile promoteTile(RewriterBase &rewriter, const TilePlan &plan,
                         scf::ForOp loop, Operation *nest,
                         const TargetModel &target) {
  memref::SubViewOp tile = plan.tile;
  Location loc = tile.getLoc();

  PromotedTile promoted;
  rewriter.setInsertionPoint(nest);
  promoted.buffer = allocLocal(rewriter, loc, plan.shape,
                               tile.getType().getElementType(), target);
  promoted.tag = allocTag(rewriter, loc);

  rewriter.setInsertionPointAfter(tile);
  SmallVector<Value> dramIndices =
      getValueOrCreateConstantIndexOp(rewriter, loc, tile.getMixedOffsets());
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  SmallVector<Value> localIndices(plan.shape.size(), zero);
  Value numElements =
      arith::ConstantIndexOp::create(rewriter, loc, plan.dma.numElements);
  Value source = tile.getSource();

  if (reads(plan.access)) {
    promoted.start =
        createDma(rewriter, loc, source, dramIndices, promoted.buffer,
                  localIndices, plan.dma, promoted.tag);
    promoted.wait = memref::DmaWaitOp::create(rewriter, loc, promoted.tag,
                                              ValueRange{zero}, numElements);
  }

  if (writes(plan.access)) {
    rewriter.setInsertionPoint(loop.getBody()->getTerminator());
    promoted.store = createDma(rewriter, loc, promoted.buffer, localIndices,
                               source, dramIndices, plan.dma, promoted.tag);
    promoted.storeWait = memref::DmaWaitOp::create(
        rewriter, loc, promoted.tag, ValueRange{zero}, numElements);
  }

  replaceAndRetype(rewriter, tile.getResult(), promoted.buffer.getResult());
  rewriter.eraseOp(tile);

  return promoted;
}

LogicalResult moveInvariantDma(RewriterBase &rewriter, scf::ForOp loop,
                               memref::DmaStartOp &start,
                               memref::DmaWaitOp &wait, bool after) {
  FailureOr<SmallVector<Operation *>> slice = getLoopLocalSlice(start, loop);
  if (failed(slice))
    return failure();

  if (after)
    rewriter.setInsertionPointAfter(loop);
  else
    rewriter.setInsertionPoint(loop);

  IRMapping mapping;

  for (Operation *op : *slice)
    rewriter.clone(*op, mapping);

  auto movedStart = cast<memref::DmaStartOp>(rewriter.clone(*start, mapping));
  auto movedWait = cast<memref::DmaWaitOp>(rewriter.clone(*wait, mapping));
  SmallVector<Operation *> defs;

  for (Operation *op : {wait.getOperation(), start.getOperation()}) {
    for (Value operand : op->getOperands())
      defs.push_back(operand.getDefiningOp());

    rewriter.eraseOp(op);
  }

  start = movedStart;
  wait = movedWait;
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

bool feedsCompute(Value view, Operation *within) {
  return llvm::any_of(view.getUsers(), [&](Operation *user) {
    if (!within->isAncestor(user))
      return false;

    if (auto subview = dyn_cast<memref::SubViewOp>(user))
      return feedsCompute(subview.getResult(), within);

    return isa<vector::TransferReadOp, vector::TransferWriteOp, memref::LoadOp,
               memref::StoreOp, linalg::LinalgOp>(user);
  });
}

SetVector<Value> mainMemoryOperands(scf::ForOp nest) {
  SetVector<Value> values;

  nest.walk([&](Operation *op) {
    if (isa<memref::DmaStartOp, memref::DmaWaitOp, memref::DeallocOp>(op))
      return;

    for (Value operand : op->getOperands()) {
      auto type = dyn_cast<MemRefType>(operand.getType());

      if (!type || type.getMemorySpace() ||
          !type.getElementType().isIntOrFloat())
        continue;

      if (!nest->isAncestor(operand.getParentRegion()->getParentOp()) &&
          feedsCompute(operand, nest))
        values.insert(operand);
    }
  });

  return values;
}

uint64_t bytesOf(MemRefType type) {
  return type.getNumElements() *
         llvm::divideCeil(type.getElementTypeBitWidth(), 8);
}

uint64_t localBytesInUse(Operation *func) {
  uint64_t bytes = 0;

  func->walk([&](memref::AllocOp alloc) {
    if (isa_and_nonnull<LocalMemorySpaceAttr>(alloc.getType().getMemorySpace()))
      bytes += bytesOf(alloc.getType());
  });

  return bytes;
}

LogicalResult stageNest(RewriterBase &rewriter, scf::ForOp nest,
                        const TargetModel &target) {
  SetVector<Value> values = mainMemoryOperands(nest);
  uint64_t inUse = localBytesInUse(nest->getParentOfType<func::FuncOp>());
  uint64_t bytes = 0;

  for (Value value : values) {
    auto type = cast<MemRefType>(value.getType());

    if (!memref::isStaticShapeAndContiguousRowMajor(type))
      return nest.emitError("cannot stage a non-contiguous buffer in local "
                            "memory for target '")
             << target.name << "'";

    bytes += bytesOf(type);
  }

  if (inUse + bytes > target.localMemBytes)
    return nest.emitError() << "operands of this loop nest need " << bytes
                            << " bytes of local memory on top of " << inUse
                            << " already allocated, but target '" << target.name
                            << "' has " << target.localMemBytes;

  Location loc = nest.getLoc();

  for (Value value : values) {
    std::optional<Access> access = classifyView(value, nest);

    if (!access)
      return nest.emitError("cannot classify how this loop nest uses a buffer");

    auto type = cast<MemRefType>(value.getType());
    DmaShape whole;
    whole.numElements = type.getNumElements();

    rewriter.setInsertionPoint(nest);
    memref::AllocOp buffer = allocLocal(rewriter, loc, type.getShape(),
                                        type.getElementType(), target);
    memref::AllocOp tag = allocTag(rewriter, loc);
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    SmallVector<Value> indices(type.getRank(), zero);
    Value numElements =
        arith::ConstantIndexOp::create(rewriter, loc, whole.numElements);

    if (reads(*access)) {
      createDma(rewriter, loc, value, indices, buffer, indices, whole, tag);
      memref::DmaWaitOp::create(rewriter, loc, tag, ValueRange{zero},
                                numElements);
    }

    replaceAndRetype(rewriter, value, buffer.getResult(), nest);
    rewriter.setInsertionPointAfter(nest);

    if (writes(*access)) {
      createDma(rewriter, loc, buffer, indices, value, indices, whole, tag);
      memref::DmaWaitOp::create(rewriter, loc, tag, ValueRange{zero},
                                numElements);
    }

    memref::DeallocOp::create(rewriter, loc, buffer);
    memref::DeallocOp::create(rewriter, loc, tag);
  }

  return success();
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
    bool outputs = promoteOutputs || !target->hasCache();

    for (scf::ForOp loop : loops)
      if (failed(promoteLoop(rewriter, loop, *target, outputs)))
        return signalPassFailure();

    if (!target->hasCache() && failed(stageNests(rewriter, module, *target)))
      return signalPassFailure();

    DominanceInfo domInfo;
    eliminateCommonSubExpressions(rewriter, domInfo, module);
  }

  static LogicalResult stageNests(RewriterBase &rewriter, ModuleOp module,
                                  const TargetModel &target) {
    for (auto func : module.getOps<func::FuncOp>()) {
      if (func.isExternal())
        continue;

      SmallVector<scf::ForOp> nests(func.getBody().getOps<scf::ForOp>());

      for (scf::ForOp nest : nests)
        if (failed(stageNest(rewriter, nest, target)))
          return failure();
    }

    return success();
  }

  LogicalResult promoteLoop(RewriterBase &rewriter, scf::ForOp loop,
                            const TargetModel &target, bool outputs) {
    SmallVector<TilePlan> plans = planTiles(loop, outputs);

    if (plans.empty())
      return success();

    auto requiredBytes = [&](bool doubled) {
      uint64_t bytes = 0;

      for (const TilePlan &plan : plans) {
        bool pipelined = doubled && plan.varies && plan.access == Access::Read;
        bytes += plan.bytes * (pipelined ? 2 : 1);
      }

      return bytes;
    };

    bool doubled = doubleBuffer && requiredBytes(true) <= target.localMemBytes;

    if (requiredBytes(false) > target.localMemBytes)
      return loop.emitError()
             << "tiles of this cache tile need " << requiredBytes(false)
             << " bytes of local memory, but target '" << target.name
             << "' has " << target.localMemBytes;

    Operation *nest = getOutermostLoop(loop);
    SmallVector<PromotedTile> promotedTiles;
    SmallVector<PromotedTile> pipelined;

    for (const TilePlan &plan : plans) {
      PromotedTile promoted = promoteTile(rewriter, plan, loop, nest, target);

      if (!plan.varies && promoted.store &&
          failed(moveInvariantDma(rewriter, loop, promoted.store,
                                  promoted.storeWait, true)))
        return loop.emitError("could not sink a loop-invariant tile store");

      if (!plan.varies && promoted.start &&
          failed(moveInvariantDma(rewriter, loop, promoted.start, promoted.wait,
                                  false)))
        return loop.emitError("could not hoist a loop-invariant tile DMA");

      if (plan.varies && plan.access == Access::Read && doubled) {
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
