#ifndef NANODSP_SCHEDULE_TARGETMODEL_H
#define NANODSP_SCHEDULE_TARGETMODEL_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace mlir {
class Operation;

namespace nanodsp {

struct TargetModel {
  llvm::StringRef name;
  unsigned vectorBits = 0;
  unsigned numVectorRegs = 0;
  uint64_t cacheBytes = 0;
  double cacheFraction = 0.0;
  uint64_t localMemBytes = 0;

  [[nodiscard]] unsigned lanes(unsigned elemBits) const {
    return vectorBits / elemBits;
  }

  [[nodiscard]] uint64_t tileBudgetBytes() const {
    return static_cast<uint64_t>(static_cast<double>(cacheBytes) *
                                 cacheFraction);
  }

  static std::optional<TargetModel> lookup(llvm::StringRef name);
  static llvm::ArrayRef<TargetModel> all();
};

std::optional<TargetModel> lookupTarget(llvm::StringRef name, Operation *op);

}
}

#endif
