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

def NanoDSPPromoteLocal : Pass<"nanodsp-promote-local", "::mlir::ModuleOp"> {
  let summary = "Stage cache-tile input tiles through local memory with DMA (bufferized IR).";
  let description = [{
    Runs on bufferized IR produced from a `-nanodsp-optimize` schedule, and
    only for a target whose TargetModel has local memory (`localMemBytes >
    0`, e.g. `hexagon-hvx128`); for any other target it changes nothing.

    In every loop the schedule marked `nanodsp.cache_loop` (the innermost
    cache-tile loop), each read-only operand tile (a `memref.subview` of a
    buffer defined outside the loop) is copied into a buffer in
    `#dsp.local` with `memref.dma_start` / `memref.dma_wait`, and the
    compute reads the local copy. Buffers are allocated once around the
    whole cache-tile nest.

    Each tile is loaded synchronously: the DMA is issued and waited for at
    the top of the iteration that uses it. A tile that does not change with
    the loop is loaded once before it. If the tiles do not fit
    `localMemBytes`, the pass reports an error at the loop.

    The output tile is not promoted. DMAs move one contiguous run or one
    level of strided rows; a tile that needs more is left in place.
  }];

  let options = [
    Option<"targetName", "target", "std::string", /*default=*/"\"host-neon\"",
           "TargetModel whose localMemBytes bounds the promoted tiles">,
  ];

  let dependentDialects = [
    "::mlir::affine::AffineDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::nanodsp::DSPDialect"
  ];
}

def NanoDSPLowerLocal : Pass<"nanodsp-lower-local", "::mlir::ModuleOp"> {
  let summary = "Lower DMAs to synchronous copies and drop the #dsp.local memory space.";
  let description = [{
    Functional lowering for a host or an emulator, which have no DMA engine
    and no separate local memory: each `memref.dma_start` produced by
    `-nanodsp-promote-local` becomes a synchronous `linalg.copy` of the
    source tile into the local buffer, each `memref.dma_wait` is removed,
    and `#dsp.local` is replaced by the default memory space so the standard
    LLVM lowering applies. Tag buffers left without users are removed.

    `linalg.copy` rather than `memref.copy`: a strided `memref.copy` lowers
    to a call into the MLIR C runner library, which an embedded target such
    as the Hexagon harness does not link. `linalg.copy` lowers to plain loops.
  }];

  let dependentDialects = [
    "::mlir::linalg::LinalgDialect",
    "::mlir::memref::MemRefDialect"
  ];
}

#endif // NANODSP_SCHEDULE_PASSES_TD
