#include "Translator.h"

#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace mlir::nanodsp::mn1;

bool mlir::nanodsp::mn1::isContiguousRowMajor(MemRefType type) {
  SmallVector<int64_t> strides;
  int64_t offset = 0;

  if (!type.hasStaticShape() ||
      failed(type.getStridesAndOffset(strides, offset)) || offset != 0)
    return false;

  int64_t expected = 1;

  for (int64_t d = type.getRank() - 1; d >= 0; --d) {
    if (strides[d] != expected)
      return false;

    expected *= type.getDimSize(d);
  }

  return true;
}

LogicalResult Translator::translate(ModuleOp module) {
  SmallVector<func::FuncOp> funcs;

  for (auto func : module.getOps<func::FuncOp>())
    if (!func.isExternal())
      funcs.push_back(func);

  if (funcs.size() != 1)
    return module.emitError("MN1 translation expects exactly one function");

  return emitFunction(funcs.front());
}

LogicalResult Translator::emitFunction(func::FuncOp func) {
  out_.note("mn1 code for @" + func.getSymName());

  for (BlockArgument arg : func.getArguments()) {
    auto type = dyn_cast<MemRefType>(arg.getType());

    if (!type || !isContiguousRowMajor(type) ||
        type.getElementTypeBitWidth() != kElementBytes * 8)
      return func.emitError(
          "MN1 arguments must be contiguous memrefs of 32-bit elements");

    uint64_t bytes = type.getNumElements() * kElementBytes;

    if (bytes > kDramSlotBytes || dramTop_ + kDramSlotBytes > kDramBytes)
      return func.emitError("argument does not fit the MN1 DRAM layout");

    Buffer buffer;
    buffer.base = Scalar::of(static_cast<int64_t>(dramTop_));
    buffer.shape = llvm::to_vector(type.getShape());
    buffer.strides = type.getStridesAndOffset().first;
    buffers_[arg] = buffer;
    out_.note(llvm::formatv("arg{0} at {1:x} ({2} bytes)", arg.getArgNumber(),
                            dramTop_, bytes));
    dramTop_ += kDramSlotBytes;
  }

  planReleases(func);
  return emitBlock(func.getBody().front());
}

LogicalResult Translator::emitBlock(Block &block) {
  for (Operation &op : block) {
    if (failed(emitOp(&op)))
      return failure();

    releaseAfter(&op);
  }

  return success();
}

LogicalResult Translator::emitOp(Operation *op) {
  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case<arith::ConstantOp, ub::PoisonOp, memref::DeallocOp, arith::CmpIOp,
            scf::YieldOp>([](Operation *) { return success(); })
      .Case([&](arith::AddIOp o) { return emitAddI(o); })
      .Case([&](arith::MulIOp o) { return emitMulI(o); })
      .Case([&](affine::AffineApplyOp o) { return emitAffineApply(o); })
      .Case([&](memref::AllocOp o) { return emitAlloc(o); })
      .Case([&](memref::SubViewOp o) { return emitSubView(o); })
      .Case([&](memref::DmaStartOp o) { return emitDmaStart(o); })
      .Case([&](memref::DmaWaitOp o) { return emitDmaWait(o); })
      .Case([&](linalg::FillOp o) { return emitFill(o); })
      .Case([&](vector::TransferReadOp o) { return emitTransferRead(o); })
      .Case([&](vector::TransferWriteOp o) { return emitTransferWrite(o); })
      .Case<arith::AddFOp, arith::MulFOp>(
          [&](Operation *o) { return emitElementwise(o); })
      .Case([&](scf::ForOp o) { return emitFor(o); })
      .Case([&](scf::IfOp o) { return emitIf(o); })
      .Case([&](func::ReturnOp o) -> LogicalResult {
        for (Value result : o.getOperands()) {
          auto it = buffers_.find(result);

          if (it == buffers_.end() || it->second.local)
            return o.emitError("MN1 results must be buffers in DRAM");

          out_.note(llvm::formatv("result at {0:x}", it->second.base.constant));
        }

        out_.instr("halt");
        return success();
      })
      .Default([](Operation *o) {
        return o->emitError("the MN1 backend does not support this op");
      });
}

void Translator::planReleases(func::FuncOp func) {
  func.walk([&](arith::MulFOp mul) {
    if (!isa<VectorType>(mul.getType()) || !mul->hasOneUse())
      return;

    Operation *user = *mul->getUsers().begin();

    if (isa<arith::AddFOp>(user) && user->getBlock() == mul->getBlock())
      fusedInto_[mul] = user;
  });

  func.walk([&](arith::CmpIOp cmp) {
    if (!cmp->hasOneUse())
      return;

    Operation *user = *cmp->getUsers().begin();

    if (isa<scf::IfOp>(user) && user->getBlock() == cmp->getBlock())
      fusedInto_[cmp] = user;
  });

  auto plan = [&](Value value, Operation *def) {
    Block *home = value.getParentBlock();
    Operation *last = nullptr;

    for (Operation *user : value.getUsers()) {
      Operation *anchor = home->findAncestorOpInBlock(*user);

      if (anchor == nullptr)
        continue;

      if (auto it = fusedInto_.find(anchor); it != fusedInto_.end())
        anchor = it->second;

      if (last == nullptr || last->isBeforeInBlock(anchor))
        last = anchor;
    }

    releases_[last != nullptr ? last : def].push_back(value);
  };

  func.walk([&](Operation *op) {
    for (Value result : op->getResults())
      plan(result, op);
  });
}

void Translator::releaseAfter(Operation *op) {
  auto it = releases_.find(op);

  if (it == releases_.end())
    return;

  for (Value value : it->second)
    release(value);
}

void Translator::release(Value value) {
  if (auto it = scalars_.find(value); it != scalars_.end()) {
    if (it->second.ownsReg())
      scalarRegs_.release(it->second.reg);

    scalars_.erase(it);
  }

  if (auto it = buffers_.find(value); it != buffers_.end()) {
    if (it->second.base.ownsReg())
      scalarRegs_.release(it->second.base.reg);

    buffers_.erase(it);
  }

  if (auto it = vectors_.find(value); it != vectors_.end()) {
    llvm::SmallDenseSet<int, 16> seen;

    for (int reg : it->second.regs)
      if (seen.insert(reg).second)
        vectorRegs_.release(reg);

    vectors_.erase(it);
  }

  tags_.erase(value);
}
