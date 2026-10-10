#include "nanodsp/Schedule/TargetModel.h"

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

#include <array>

using namespace mlir;
using namespace mlir::nanodsp;

constexpr uint64_t kKiB = 1024;

static const std::array<TargetModel, 3> kTargets = {{
    {"host-neon", 128, 32, 128 * kKiB, 0.5, 0},
    {"x86-avx2", 256, 16, 32 * kKiB, 0.5, 0},
    {"hexagon-hvx128", 1024, 32, 512 * kKiB, 0.5, 256 * kKiB},
}};

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
