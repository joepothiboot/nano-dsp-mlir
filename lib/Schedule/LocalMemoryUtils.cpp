#include "LocalMemoryUtils.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace mlir::nanodsp;
using namespace mlir::nanodsp::local;

namespace {

Access combine(Access a, Access b) {
  return static_cast<Access>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
}

std::optional<Access> classifyUse(Operation *user, Value view) {
  if (auto subview = dyn_cast<memref::SubViewOp>(user))
    return classifyView(subview.getResult());

  if (isa<vector::TransferReadOp, memref::LoadOp>(user))
    return Access::Read;

  if (isa<vector::TransferWriteOp, memref::StoreOp, linalg::FillOp>(user))
    return Access::Write;

  auto linalgOp = dyn_cast<linalg::LinalgOp>(user);

  if (!linalgOp)
    return std::nullopt;

  Access access = Access::None;

  for (OpOperand &operand : linalgOp->getOpOperands()) {
    if (operand.get() != view)
      continue;

    if (linalgOp.isDpsInput(&operand)) {
      access = combine(access, Access::Read);
      continue;
    }

    bool readsInit = linalgOp.payloadUsesValueFromOperand(&operand);
    access = combine(access, readsInit ? Access::ReadWrite : Access::Write);
  }

  return access;
}

}

std::optional<DmaShape> local::getDmaShape(ArrayRef<int64_t> tileShape,
                                           ArrayRef<int64_t> strides) {
  DmaShape dma;
  int64_t run = 1;
  int64_t rows = 0;
  int64_t rowStride = 0;

  for (auto d = static_cast<int64_t>(tileShape.size()) - 1; d >= 0; --d) {
    int64_t extent = tileShape[d];
    int64_t stride = strides[d];

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

std::optional<Access> local::classifyView(Value view, Operation *within) {
  Access access = Access::None;

  for (Operation *user : view.getUsers()) {
    if (within != nullptr && !within->isAncestor(user))
      continue;

    std::optional<Access> use = classifyUse(user, view);

    if (!use)
      return std::nullopt;

    access = combine(access, *use);
  }

  return access;
}

FailureOr<SmallVector<Operation *>> local::getLoopLocalSlice(Operation *op,
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

bool local::dependsOn(Operation *op, ArrayRef<Operation *> slice, Value value) {
  auto readsValue = [&](Operation *o) {
    return llvm::is_contained(o->getOperands(), value);
  };

  return readsValue(op) || llvm::any_of(slice, readsValue);
}

void local::replaceAndRetype(RewriterBase &rewriter, Value oldValue,
                             Value newValue, Operation *within) {
  for (OpOperand &use : llvm::make_early_inc_range(oldValue.getUses())) {
    Operation *user = use.getOwner();

    if (within != nullptr && !within->isAncestor(user))
      continue;

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

void local::eraseDeadDefs(RewriterBase &rewriter,
                          SmallVector<Operation *> worklist) {
  llvm::SmallPtrSet<Operation *, 8> erased;
  auto erase = [&](Operation *op) {
    erased.insert(op);
    rewriter.eraseOp(op);
  };

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();

    if ((op == nullptr) || erased.contains(op))
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

memref::DmaStartOp local::createDma(RewriterBase &rewriter, Location loc,
                                    Value src, ValueRange srcIndices, Value dst,
                                    ValueRange dstIndices,
                                    const DmaShape &shape, Value tag) {
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value numElements =
      arith::ConstantIndexOp::create(rewriter, loc, shape.numElements);
  Value stride;
  Value eltsPerStride;

  if (shape.stride) {
    stride = arith::ConstantIndexOp::create(rewriter, loc, *shape.stride);
    eltsPerStride =
        arith::ConstantIndexOp::create(rewriter, loc, shape.eltsPerStride);
  }

  return memref::DmaStartOp::create(rewriter, loc, src, srcIndices, dst,
                                    dstIndices, numElements, tag,
                                    ValueRange{zero}, stride, eltsPerStride);
}
