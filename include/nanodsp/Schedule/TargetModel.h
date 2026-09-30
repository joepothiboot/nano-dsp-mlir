//===- TargetModel.h - Compile-time machine model ---------------*- C++ -*-===//
//
// The handful of hardware numbers the tile-size model needs. These are
// compile-time constants picked by name (-nanodsp-optimize{target=...}), not
// queried from the machine running the compiler, so cross-compiling for a
// target you are not sitting on works and results are reproducible.
//
//===----------------------------------------------------------------------===//

#ifndef NANODSP_SCHEDULE_TARGETMODEL_H
#define NANODSP_SCHEDULE_TARGETMODEL_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace mlir {
namespace nanodsp {

struct TargetModel {
  llvm::StringRef name;
  /// Width of one SIMD register.
  unsigned vectorBits;
  /// Architectural SIMD registers the register tile may occupy.
  unsigned numVectorRegs;
  /// The cache level the vector unit loads from (L1D on CPUs).
  uint64_t cacheBytes;
  /// Share of `cacheBytes` one tile's working set may use. The rest is
  /// headroom for conflict misses and everything else live in the loop.
  double cacheFraction;
  /// Software-managed local memory (TCM/scratchpad); 0 if there is none.
  uint64_t localMemBytes;

  unsigned lanes(unsigned elemBits) const { return vectorBits / elemBits; }
  uint64_t tileBudgetBytes() const {
    return static_cast<uint64_t>(cacheBytes * cacheFraction);
  }

  /// Built-in models, looked up by name. Returns std::nullopt for an unknown
  /// name.
  static std::optional<TargetModel> lookup(llvm::StringRef name);
  static llvm::ArrayRef<TargetModel> all();
};

} // namespace nanodsp
} // namespace mlir

#endif // NANODSP_SCHEDULE_TARGETMODEL_H
