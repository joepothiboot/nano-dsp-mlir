#include "Translator.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/FormatVariadic.h"

#include <utility>

using namespace mlir;
using namespace mlir::nanodsp::mn1;

namespace {

enum class ReadShape : uint8_t { Rows, SplatRows, SharedRow };

std::optional<ReadShape> classifyRead(AffineMap map, int64_t vectorRank,
                                      int64_t memrefRank) {
  if (map.isMinorIdentity() && vectorRank == memrefRank)
    return ReadShape::Rows;

  if (vectorRank != 2 || memrefRank != 1 || map.getNumResults() != 2)
    return std::nullopt;

  auto isDim = [](AffineExpr e) { return isa<AffineDimExpr>(e); };
  auto isZero = [](AffineExpr e) {
    auto c = dyn_cast<AffineConstantExpr>(e);
    return c && c.getValue() == 0;
  };

  if (isDim(map.getResult(0)) && isZero(map.getResult(1)))
    return ReadShape::SplatRows;

  if (isZero(map.getResult(0)) && isDim(map.getResult(1)))
    return ReadShape::SharedRow;

  return std::nullopt;
}

std::string memOperand(Scalar address, int64_t offset) {
  if (address.isConstant())
    return llvm::formatv("{0}(r0)", address.constant + offset).str();

  return llvm::formatv("{0}(r{1})", offset, address.reg).str();
}

bool hasNoSharedRegs(ArrayRef<int> regs) {
  llvm::SmallDenseSet<int, 16> seen(regs.begin(), regs.end());
  return seen.size() == regs.size();
}

}

FailureOr<VectorParts> Translator::allocParts(VectorType type, Operation *at) {
  if (!type.getElementType().isF32() || type.getRank() > 2)
    return at->emitError("MN1 vectors are rank 1 or 2 of f32");

  int64_t cols = type.getShape().back();

  if (cols % kLanes != 0)
    return at->emitError("MN1 vector rows must be multiples of ")
           << kLanes << " lanes";

  VectorParts parts;
  parts.rows = type.getRank() == 2 ? type.getDimSize(0) : 1;
  parts.partsPerRow = cols / kLanes;

  for (int64_t i = 0; i < parts.rows * parts.partsPerRow; ++i) {
    FailureOr<int> reg = takeVector(at);

    if (failed(reg))
      return failure();

    parts.regs.push_back(*reg);
  }

  return parts;
}

FailureOr<VectorParts> Translator::resultParts(Operation *op, Value reusable,
                                               VectorType type) {
  auto it = vectors_.find(reusable);
  bool dies = llvm::is_contained(releases_[op], reusable);

  if (dies && it != vectors_.end() && hasNoSharedRegs(it->second.regs) &&
      reusable.getType() == type) {
    VectorParts parts = it->second;
    vectors_.erase(it);
    return parts;
  }

  return allocParts(type, op);
}

LogicalResult Translator::emitTransferRead(vector::TransferReadOp op) {
  auto buffer = buffers_.find(op.getBase());

  if (buffer == buffers_.end() || !buffer->second.local)
    return op.emitError("the MN1 core can only read the scratchpad");

  if (llvm::any_of(op.getInBoundsValues(), [](bool b) { return !b; }))
    return op.emitError("MN1 needs in-bounds transfers");

  VectorType type = op.getVectorType();
  const Buffer &b = buffer->second;
  std::optional<ReadShape> shape =
      classifyRead(op.getPermutationMap(), type.getRank(),
                   static_cast<int64_t>(b.shape.size()));

  if (!shape)
    return op.emitError(
        "transfer_read layout the MN1 backend does not support");

  FailureOr<Scalar> address = elementAddress(b, op.getIndices(), op);

  if (failed(address))
    return failure();

  VectorParts parts;
  parts.rows = type.getRank() == 2 ? type.getDimSize(0) : 1;
  parts.partsPerRow = type.getShape().back() / kLanes;

  if (*shape == ReadShape::Rows) {
    FailureOr<VectorParts> fresh = allocParts(type, op);

    if (failed(fresh))
      return failure();

    parts = *fresh;
    int64_t rowStride = type.getRank() == 2 ? b.strides[0] : 0;

    for (int64_t r = 0; r < parts.rows; ++r)
      for (int64_t p = 0; p < parts.partsPerRow; ++p)
        out_.instr(
            llvm::formatv("vld v{0}, {1}", parts.at(r, p),
                          memOperand(*address, (r * rowStride * kElementBytes) +
                                                   (p * kVectorBytes))));
  } else if (*shape == ReadShape::SplatRows) {
    for (int64_t r = 0; r < parts.rows; ++r) {
      FailureOr<int> reg = takeVector(op);

      if (failed(reg))
        return failure();

      out_.instr(llvm::formatv(
          "vsplat v{0}, {1}", *reg,
          memOperand(*address, r * b.strides[0] * kElementBytes)));
      parts.regs.append(parts.partsPerRow, *reg);
    }
  } else {
    if (b.strides[0] != 1)
      return op.emitError("a broadcast row must be contiguous on MN1");

    SmallVector<int> row;

    for (int64_t p = 0; p < parts.partsPerRow; ++p) {
      FailureOr<int> reg = takeVector(op);

      if (failed(reg))
        return failure();

      out_.instr(llvm::formatv("vld v{0}, {1}", *reg,
                               memOperand(*address, p * kVectorBytes)));
      row.push_back(*reg);
    }

    for (int64_t r = 0; r < parts.rows; ++r)
      parts.regs.append(row.begin(), row.end());
  }

  if (address->ownsReg())
    scalarRegs_.release(address->reg);

  vectors_[op.getResult()] = std::move(parts);
  return success();
}

LogicalResult Translator::emitTransferWrite(vector::TransferWriteOp op) {
  auto buffer = buffers_.find(op.getBase());

  if (buffer == buffers_.end() || !buffer->second.local)
    return op.emitError("the MN1 core can only write the scratchpad");

  VectorType type = op.getVectorType();
  const Buffer &b = buffer->second;

  if (!op.getPermutationMap().isMinorIdentity() ||
      type.getRank() != static_cast<int64_t>(b.shape.size()))
    return op.emitError("MN1 writes whole rows only");

  auto value = vectors_.find(op.getValueToStore());

  if (value == vectors_.end())
    return op.emitError("stored vector has no MN1 registers");

  FailureOr<Scalar> address = elementAddress(b, op.getIndices(), op);

  if (failed(address))
    return failure();

  const VectorParts &parts = value->second;
  int64_t rowStride = type.getRank() == 2 ? b.strides[0] : 0;

  for (int64_t r = 0; r < parts.rows; ++r)
    for (int64_t p = 0; p < parts.partsPerRow; ++p)
      out_.instr(
          llvm::formatv("vst v{0}, {1}", parts.at(r, p),
                        memOperand(*address, (r * rowStride * kElementBytes) +
                                                 (p * kVectorBytes))));

  if (address->ownsReg())
    scalarRegs_.release(address->reg);

  return success();
}

LogicalResult Translator::emitElementwise(Operation *op) {
  if (fusedInto_.contains(op))
    return success();

  auto type = dyn_cast<VectorType>(op->getResult(0).getType());

  if (!type)
    return op->emitError("MN1 has no scalar floating-point unit");

  Value lhs = op->getOperand(0);
  Value rhs = op->getOperand(1);
  auto mulOf = [&](Value v) -> arith::MulFOp {
    auto mul = v.getDefiningOp<arith::MulFOp>();
    return mul && fusedInto_.lookup(mul) == op ? mul : nullptr;
  };

  arith::MulFOp mul = mulOf(rhs) ? mulOf(rhs) : mulOf(lhs);
  Value other = mul && mul.getResult() == rhs ? lhs : rhs;

  if (mul) {
    auto a = vectors_.find(mul.getLhs());
    auto b = vectors_.find(mul.getRhs());
    auto x = vectors_.find(other);

    if (a == vectors_.end() || b == vectors_.end() || x == vectors_.end())
      return op->emitError("operand has no MN1 registers");

    VectorParts av = a->second;
    VectorParts bv = b->second;
    VectorParts xv = x->second;
    FailureOr<VectorParts> result = resultParts(op, other, type);
    FailureOr<int> temp = takeVector(op);

    if (failed(result) || failed(temp))
      return failure();

    for (int64_t r = 0; r < result->rows; ++r) {
      for (int64_t p = 0; p < result->partsPerRow; ++p) {
        out_.instr(llvm::formatv("vmul v{0}, v{1}, v{2}", *temp, av.at(r, p),
                                 bv.at(r, p)));
        out_.instr(llvm::formatv("vadd v{0}, v{1}, v{2}", result->at(r, p),
                                 xv.at(r, p), *temp));
      }
    }

    vectorRegs_.release(*temp);
    vectors_[op->getResult(0)] = *result;
    return success();
  }

  auto a = vectors_.find(lhs);
  auto b = vectors_.find(rhs);

  if (a == vectors_.end() || b == vectors_.end())
    return op->emitError("operand has no MN1 registers");

  VectorParts av = a->second;
  VectorParts bv = b->second;
  FailureOr<VectorParts> result = resultParts(op, lhs, type);

  if (failed(result))
    return failure();

  llvm::StringRef mnemonic = isa<arith::AddFOp>(op) ? "vadd" : "vmul";

  for (int64_t r = 0; r < result->rows; ++r)
    for (int64_t p = 0; p < result->partsPerRow; ++p)
      out_.instr(llvm::formatv("{0} v{1}, v{2}, v{3}", mnemonic,
                               result->at(r, p), av.at(r, p), bv.at(r, p)));

  vectors_[op->getResult(0)] = *result;
  return success();
}
