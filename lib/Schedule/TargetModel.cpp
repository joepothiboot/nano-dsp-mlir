//===- TargetModel.cpp - Built-in machine models --------------------------===//

#include "nanodsp/Schedule/TargetModel.h"

#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::nanodsp;

// cacheFraction = 0.5 is a starting estimate, not yet validated against
// measurements (see "Known limitations" in README.md).
static const TargetModel kTargets[] = {
    // Apple M-series performance core: 128-bit NEON, 32 V registers, 128 KiB
    // L1D.
    {"host-neon", /*vectorBits=*/128, /*numVectorRegs=*/32,
     /*cacheBytes=*/128 * 1024, /*cacheFraction=*/0.5, /*localMemBytes=*/0},
    // x86-64 with AVX2: 256-bit YMM, 16 registers, 32 KiB L1D.
    {"x86-avx2", /*vectorBits=*/256, /*numVectorRegs=*/16,
     /*cacheBytes=*/32 * 1024, /*cacheFraction=*/0.5, /*localMemBytes=*/0},
};

ArrayRef<TargetModel> TargetModel::all() { return kTargets; }

std::optional<TargetModel> TargetModel::lookup(StringRef name) {
  for (const TargetModel &t : kTargets)
    if (t.name == name)
      return t;
  return std::nullopt;
}
