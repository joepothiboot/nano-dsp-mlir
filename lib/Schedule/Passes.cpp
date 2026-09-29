//===- Passes.cpp - Stage 3 schedule passes and the lowering pipeline -----===//

#include "nanodsp/Schedule/Passes.h"
#include "nanodsp/Schedule/ScheduleGen.h"
#include "nanodsp/Schedule/TargetModel.h"

#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace nanodsp {
#define GEN_PASS_DEF_NANODSPOPTIMIZE
#define GEN_PASS_DEF_NANODSPEMITSCHEDULE
#include "nanodsp/Schedule/Passes.h.inc"
} // namespace nanodsp
} // namespace mlir

using namespace mlir;
using namespace mlir::nanodsp;

namespace {

/// The transform ops a schedule uses (structured.*, apply_patterns.vector.*)
/// are dialect extensions, which must be registered before the schedule is
/// parsed.
void registerScheduleExtensions(DialectRegistry &registry) {
  linalg::registerTransformDialectExtension(registry);
  vector::registerTransformDialectExtension(registry);
}

std::optional<TargetModel> lookupTarget(StringRef name, Operation *op) {
  if (std::optional<TargetModel> target = TargetModel::lookup(name))
    return target;
  std::string known;
  for (const TargetModel &t : TargetModel::all())
    known += (known.empty() ? "" : ", ") + t.name.str();
  op->emitError() << "unknown target '" << name << "' (known: " << known << ")";
  return std::nullopt;
}

struct NanoDSPOptimizePass
    : public mlir::nanodsp::impl::NanoDSPOptimizeBase<NanoDSPOptimizePass> {
  using mlir::nanodsp::impl::NanoDSPOptimizeBase<
      NanoDSPOptimizePass>::NanoDSPOptimizeBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    NanoDSPOptimizeBase::getDependentDialects(registry);
    registerScheduleExtensions(registry);
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    SmallVector<linalg::GenericOp> ops = tagScheduleTargets(module);

    OwningOpRef<ModuleOp> schedule;
    ParserConfig config(ctx);
    if (scheduleFile.empty()) {
      std::optional<TargetModel> target = lookupTarget(targetName, module);
      if (!target)
        return signalPassFailure();
      schedule = parseSourceString<ModuleOp>(buildDefaultSchedule(ops, *target),
                                             config, "nanodsp-schedule");
    } else {
      schedule = parseSourceFile<ModuleOp>(scheduleFile, config);
    }
    if (!schedule)
      return signalPassFailure();

    auto entry =
        schedule->lookupSymbol<transform::NamedSequenceOp>("__transform_main");
    if (!entry) {
      module.emitError() << "schedule has no @__transform_main";
      return signalPassFailure();
    }
    if (failed(transform::applyTransformNamedSequence(
            module, entry, *schedule, transform::TransformOptions())))
      return signalPassFailure();

    stripScheduleTags(module);
  }
};

struct NanoDSPEmitSchedulePass
    : public mlir::nanodsp::impl::NanoDSPEmitScheduleBase<
          NanoDSPEmitSchedulePass> {
  using mlir::nanodsp::impl::NanoDSPEmitScheduleBase<
      NanoDSPEmitSchedulePass>::NanoDSPEmitScheduleBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    NanoDSPEmitScheduleBase::getDependentDialects(registry);
    registerScheduleExtensions(registry);
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::optional<TargetModel> target = lookupTarget(targetName, module);
    if (!target)
      return signalPassFailure();

    SmallVector<linalg::GenericOp> ops = tagScheduleTargets(module);
    OwningOpRef<ModuleOp> schedule = parseSourceString<ModuleOp>(
        buildDefaultSchedule(ops, *target), ParserConfig(&getContext()),
        "nanodsp-schedule");
    if (!schedule)
      return signalPassFailure();
    module.getBody()->push_back(schedule.release());
  }
};

} // namespace

// Bufferize, then the upstream lowering to the LLVM dialect. The vector
// passes are no-ops on unscheduled (scalar-loop) IR, so this pipeline serves
// both the scheduled and the unscheduled path.
static constexpr char kLowerToLLVM[] =
    "one-shot-bufferize{bufferize-function-boundaries},"
    "buffer-deallocation-pipeline,"
    "convert-linalg-to-loops,"
    "func.func(lower-vector-multi-reduction),"
    "convert-vector-to-scf,"
    "lower-affine,"
    "convert-scf-to-cf,"
    "expand-strided-metadata,"
    "lower-affine,"
    "convert-vector-to-llvm,"
    "convert-arith-to-llvm,"
    "finalize-memref-to-llvm,"
    "convert-func-to-llvm,"
    "convert-cf-to-llvm,"
    "convert-ub-to-llvm,"
    "reconcile-unrealized-casts";

void mlir::nanodsp::registerNanoDSPPipelines() {
  PassPipelineRegistration<>(
      "nanodsp-lower-to-llvm",
      "Bufferize and lower linalg/scf/vector on tensors to the LLVM dialect.",
      [](OpPassManager &pm) {
        if (failed(parsePassPipeline(kLowerToLLVM, pm)))
          llvm::report_fatal_error("invalid nanodsp-lower-to-llvm pipeline");
      });
}
