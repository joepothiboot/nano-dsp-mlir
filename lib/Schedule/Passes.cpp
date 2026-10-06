#include "nanodsp/Schedule/Passes.h"
#include "nanodsp/Schedule/ScheduleGen.h"
#include "nanodsp/Schedule/TargetModel.h"
#include "nanodsp/Schedule/TileSizeModel.h"

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
}
}

using namespace mlir;
using namespace mlir::nanodsp;

namespace {

void registerScheduleExtensions(DialectRegistry &registry) {
  linalg::registerTransformDialectExtension(registry);
  vector::registerTransformDialectExtension(registry);
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

void annotateTileSizes(ArrayRef<linalg::GenericOp> ops,
                       const TargetModel &target) {
  for (linalg::GenericOp op : ops) {
    FailureOr<TileSizes> sizes = computeTileSizes(op, target);

    if (failed(sizes))
      continue;

    Builder b(op.getContext());
    op->setAttr("nanodsp.loop_ranges",
                b.getDenseI64ArrayAttr(sizes->loopRanges));
    op->setAttr("nanodsp.cache_tile", b.getDenseI64ArrayAttr(sizes->cache));
    op->setAttr("nanodsp.reg_tile", b.getDenseI64ArrayAttr(sizes->reg));
    op->setAttr("nanodsp.working_set_bytes",
                b.getI64IntegerAttr(sizes->cacheWorkingSetBytes));
  }
}

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
    annotateTileSizes(ops, *target);
    OwningOpRef<ModuleOp> schedule = parseSourceString<ModuleOp>(
        buildDefaultSchedule(ops, *target), ParserConfig(&getContext()),
        "nanodsp-schedule");
    if (!schedule)
      return signalPassFailure();

    module.getBody()->push_back(schedule.release());
  }
};

}

namespace {
struct LowerBufferizedOptions
    : public PassPipelineOptions<LowerBufferizedOptions> {
  Option<bool> genericAlloc{
      *this, "generic-alloc",
      llvm::cl::desc("See -nanodsp-lower-to-llvm=generic-alloc."),
      llvm::cl::init(false)};
};

struct LowerToLLVMOptions : public PassPipelineOptions<LowerToLLVMOptions> {
  Option<bool> genericAlloc{
      *this, "generic-alloc",
      llvm::cl::desc(
          "Allocate through _mlir_memref_to_llvm_alloc/_free, which the "
          "embedding program provides, instead of calling malloc directly. "
          "Needed on 32-bit targets such as Hexagon: index (and therefore "
          "the allocation size) stays 64-bit, which does not match a 32-bit "
          "libc's malloc(size_t)."),
      llvm::cl::init(false)};
  Option<std::string> localTarget{
      *this, "local-target",
      llvm::cl::desc(
          "Stage cache tiles through the local memory of this TargetModel "
          "(-nanodsp-promote-local), then lower the DMAs to copies "
          "(-nanodsp-lower-local). Empty: no promotion."),
      llvm::cl::init("")};
};
}

static constexpr llvm::StringLiteral kBufferizePipeline =
    "one-shot-bufferize{bufferize-function-boundaries "
    "function-boundary-type-conversion=identity-layout-map},"
    "buffer-deallocation-pipeline";

static std::string lowerBufferizedPipeline(bool genericAlloc) {
  return std::string("convert-linalg-to-loops,"
                     "func.func(lower-vector-multi-reduction),"
                     "convert-vector-to-scf{full-unroll=true},"
                     "lower-affine,"
                     "convert-scf-to-cf,"
                     "expand-strided-metadata,"
                     "lower-affine,"
                     "convert-vector-to-llvm,"
                     "convert-arith-to-llvm,") +
         (genericAlloc ? "finalize-memref-to-llvm{use-generic-functions},"
                       : "finalize-memref-to-llvm,") +
         "convert-func-to-llvm,"
         "convert-cf-to-llvm,"
         "convert-ub-to-llvm,"
         "reconcile-unrealized-casts";
}

static void addPipeline(OpPassManager &pm, StringRef pipeline) {
  if (failed(parsePassPipeline(pipeline, pm)))
    llvm::report_fatal_error("invalid nanodsp pipeline");
}

void mlir::nanodsp::registerNanoDSPPipelines() {
  PassPipelineRegistration<>(
      "nanodsp-bufferize",
      "Bufferize with identity-layout function boundaries and make "
      "deallocations explicit (the first half of -nanodsp-lower-to-llvm).",
      [](OpPassManager &pm) { addPipeline(pm, kBufferizePipeline); });

  PassPipelineRegistration<LowerBufferizedOptions>(
      "nanodsp-lower-bufferized-to-llvm",
      "Lower bufferized linalg/scf/vector IR to the LLVM dialect (the second "
      "half of -nanodsp-lower-to-llvm).",
      [](OpPassManager &pm, const LowerBufferizedOptions &options) {
        addPipeline(pm, lowerBufferizedPipeline(options.genericAlloc));
      });

  PassPipelineRegistration<LowerToLLVMOptions>(
      "nanodsp-lower-to-llvm",
      "Bufferize and lower linalg/scf/vector on tensors to the LLVM dialect.",
      [](OpPassManager &pm, const LowerToLLVMOptions &options) {
        addPipeline(pm, kBufferizePipeline);

        if (!options.localTarget.empty()) {
          NanoDSPPromoteLocalOptions promote;
          promote.targetName = options.localTarget;
          pm.addPass(createNanoDSPPromoteLocal(promote));
          pm.addPass(createNanoDSPLowerLocal());
        }

        addPipeline(pm, lowerBufferizedPipeline(options.genericAlloc));
      });
}
