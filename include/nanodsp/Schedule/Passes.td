#ifndef NANODSP_SCHEDULE_PASSES_TD
#define NANODSP_SCHEDULE_PASSES_TD

include "mlir/Pass/PassBase.td"

defvar ScheduleDependentDialects = [
  "::mlir::affine::AffineDialect",
  "::mlir::arith::ArithDialect",
  "::mlir::linalg::LinalgDialect",
  "::mlir::scf::SCFDialect",
  "::mlir::tensor::TensorDialect",
  "::mlir::transform::TransformDialect",
  "::mlir::ub::UBDialect",
  "::mlir::vector::VectorDialect"
];

def NanoDSPOptimize : Pass<"nanodsp-optimize", "::mlir::ModuleOp"> {
  let summary = "Tile and vectorize linalg.generic ops with a Transform schedule (L2 -> L3).";
  let description = [{
    Applies a Transform-dialect schedule to every `linalg.generic` in the
    module. By default the schedule is generated from the TargetModel named by
    `target` (see `-nanodsp-emit-schedule` to inspect it). With
    `schedule-file`, a hand-written schedule is applied instead; it addresses
    ops through the `nanodsp.tag` attribute ("op0", "op1", ... in walk order),
    which this pass assigns before applying and removes afterwards.

    The schedule is kept out of the payload module, so the output contains
    only the transformed IR.
  }];

  let options = [
    Option<"targetName", "target", "std::string", /*default=*/"\"host-neon\"",
           "TargetModel to derive tile sizes from">,
    Option<"scheduleFile", "schedule-file", "std::string", /*default=*/"\"\"",
           "Apply the __transform_main sequence from this file instead of "
           "the generated schedule">,
  ];

  let dependentDialects = ScheduleDependentDialects;
}

def NanoDSPEmitSchedule : Pass<"nanodsp-emit-schedule", "::mlir::ModuleOp"> {
  let summary = "Tag linalg.generic ops and append the generated schedule to the module.";
  let description = [{
    Emits the schedule `-nanodsp-optimize` would apply as a nested
    `module attributes {transform.with_named_sequence}`, so it can be read,
    checked in under `schedules/`, edited, and applied with either
    `-transform-interpreter` or `-nanodsp-optimize{schedule-file=...}`.

    Each op also records the model's decision as `nanodsp.loop_ranges`,
    `nanodsp.cache_tile`, `nanodsp.reg_tile` (absolute extents per loop) and
    `nanodsp.working_set_bytes`.
  }];

  let options = [
    Option<"targetName", "target", "std::string", /*default=*/"\"host-neon\"",
           "TargetModel to derive tile sizes from">,
  ];

  let dependentDialects = ScheduleDependentDialects;
}

#endif // NANODSP_SCHEDULE_PASSES_TD
