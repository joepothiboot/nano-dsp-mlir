//===- TargetModel.cpp - Built-in machine models --------------------------===//

#include "nanodsp/Schedule/TargetModel.h"

#include "mlir/IR/Operation.h"
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
    // Hexagon V68 HVX, 128-byte mode (80-N2040-47, Hexagon V68 HVX
    // Programmer's Reference Manual): 1024-bit vectors (sec. 1.2.1), 32
    // vector registers V0-V31 (sec. 2.1). HVX loads and stores bypass the
    // scalar core's L1 D$ and go to L2, L2TCM or VTCM (sec. 1.2.3, 3.4), so
    // the cache the tiles are sized against is L2. The manual leaves L2 and
    // VTCM sizes implementation-defined (sec. 3.2); 512 KiB L2 and 256 KiB
    // VTCM are assumptions for a small part, see docs/hexagon-target.md.
    {"hexagon-hvx128", /*vectorBits=*/1024, /*numVectorRegs=*/32,
     /*cacheBytes=*/512 * 1024, /*cacheFraction=*/0.5,
     /*localMemBytes=*/256 * 1024},
};

ArrayRef<TargetModel> TargetModel::all() { return kTargets; }

std::optional<TargetModel> TargetModel::lookup(StringRef name) {
  for (const TargetModel &t : kTargets)
    if (t.name == name)
      return t;
  return std::nullopt;
}

std::optional<TargetModel> mlir::nanodsp::lookupTarget(StringRef name,
                                                       Operation *op) {
  if (std::optional<TargetModel> target = TargetModel::lookup(name))
    return target;
  std::string known;
  for (const TargetModel &t : TargetModel::all())
    known += (known.empty() ? "" : ", ") + t.name.str();
  op->emitError() << "unknown target '" << name << "' (known: " << known << ")";
  return std::nullopt;
}
