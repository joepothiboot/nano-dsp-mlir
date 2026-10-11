#ifndef NANODSP_LIB_TARGET_MN1_TRANSLATOR_H
#define NANODSP_LIB_TARGET_MN1_TRANSLATOR_H

#include "Emitter.h"
#include "MN1Constants.h"
#include "RegisterPool.h"
#include "Values.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"

#include <functional>

namespace mlir::nanodsp::mn1 {

bool isContiguousRowMajor(MemRefType type);

bool needsParity(scf::ForOp loop);

class Translator {
public:
  explicit Translator(llvm::raw_ostream &os) : out_(os) {}

  LogicalResult translate(ModuleOp module);

private:
  LogicalResult emitFunction(func::FuncOp func);
  LogicalResult emitBlock(Block &block);
  LogicalResult emitOp(Operation *op);
  void planReleases(func::FuncOp func);
  void releaseAfter(Operation *op);
  void release(Value value);

  FailureOr<int> takeScalar(Operation *at);
  FailureOr<int> takeVector(Operation *at);
  FailureOr<Scalar> scalarOf(Value value, Operation *at);
  int materialize(Scalar value, int scratch);
  FailureOr<Scalar> linear(Scalar base, ArrayRef<Term> terms, Operation *at);
  LogicalResult emitAddI(arith::AddIOp op);
  LogicalResult emitMulI(arith::MulIOp op);
  LogicalResult emitAffineApply(affine::AffineApplyOp op);
  FailureOr<int> parityOf(Value iv);

  LogicalResult emitAlloc(memref::AllocOp op);
  LogicalResult emitSubView(memref::SubViewOp op);
  LogicalResult emitDmaStart(memref::DmaStartOp op);
  LogicalResult emitDmaWait(memref::DmaWaitOp op);
  LogicalResult emitFill(linalg::FillOp op);
  FailureOr<Scalar> elementAddress(const Buffer &buffer, ValueRange indices,
                                   Operation *at);
  FailureOr<TagRef> tagOf(Value memref, ValueRange indices, Operation *at);
  void tagged(const TagRef &tag, const std::function<void(int)> &emit);
  void emitStridedDma(llvm::StringRef mnemonic, const TagRef &tag, int spm,
                      int dram, int64_t rows, int64_t rowBytes,
                      int64_t strideBytes, int counter);
  FailureOr<int64_t> zeroBlock(Operation *at);

  LogicalResult emitTransferRead(vector::TransferReadOp op);
  LogicalResult emitTransferWrite(vector::TransferWriteOp op);
  LogicalResult emitElementwise(Operation *op);
  FailureOr<VectorParts> allocParts(VectorType type, Operation *at);
  FailureOr<VectorParts> resultParts(Operation *op, Value reusable,
                                     VectorType type);

  LogicalResult emitFor(scf::ForOp op);
  LogicalResult emitIf(scf::IfOp op);

  Emitter out_;
  RegisterPool scalarRegs_{kFirstScalarReg, kLastScalarReg};
  RegisterPool vectorRegs_{0, kNumVectorRegs - 1};
  llvm::DenseMap<Value, Scalar> scalars_;
  llvm::DenseMap<Value, Buffer> buffers_;
  llvm::DenseMap<Value, VectorParts> vectors_;
  llvm::DenseMap<Value, TagRef> tags_;
  llvm::DenseMap<Operation *, int> parity_;
  llvm::DenseMap<Operation *, Operation *> fusedInto_;
  llvm::DenseMap<Operation *, llvm::SmallVector<Value>> releases_;
  uint64_t spmTop_ = 0;
  uint64_t dramTop_ = 0;
  int nextTag_ = 0;
  std::optional<int64_t> zeroBlock_;
};

}

#endif
