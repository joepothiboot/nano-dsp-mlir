#ifndef NANODSP_LIB_SCHEDULE_LOCALMEMORYUTILS_H
#define NANODSP_LIB_SCHEDULE_LOCALMEMORYUTILS_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

#include <cstdint>
#include <optional>

namespace mlir::nanodsp::local {

struct DmaShape {
  int64_t numElements = 1;
  std::optional<int64_t> stride;
  int64_t eltsPerStride = 0;
};

enum class Access : uint8_t { None = 0, Read = 1, Write = 2, ReadWrite = 3 };

[[nodiscard]] inline bool reads(Access a) {
  return (static_cast<uint8_t>(a) & static_cast<uint8_t>(Access::Read)) != 0;
}

[[nodiscard]] inline bool writes(Access a) {
  return (static_cast<uint8_t>(a) & static_cast<uint8_t>(Access::Write)) != 0;
}

std::optional<DmaShape> getDmaShape(ArrayRef<int64_t> tileShape,
                                    ArrayRef<int64_t> strides);

std::optional<Access> classifyView(Value view, Operation *within = nullptr);

FailureOr<SmallVector<Operation *>> getLoopLocalSlice(Operation *op,
                                                      scf::ForOp loop);

bool dependsOn(Operation *op, ArrayRef<Operation *> slice, Value value);

void replaceAndRetype(RewriterBase &rewriter, Value oldValue, Value newValue,
                      Operation *within = nullptr);

void eraseDeadDefs(RewriterBase &rewriter, SmallVector<Operation *> worklist);

memref::DmaStartOp createDma(RewriterBase &rewriter, Location loc, Value src,
                             ValueRange srcIndices, Value dst,
                             ValueRange dstIndices, const DmaShape &shape,
                             Value tag);

}

#endif
