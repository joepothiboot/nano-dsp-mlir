// Hand-written schedule: the classic 8x12 AArch64 SGEMM register block
// (24 accumulators of 4 lanes) instead of the 4x16 the tile-size model picks
// for host-neon. Apply with
//
//   nanodsp-opt in.mlir -convert-dsp-to-linalg \
//     -nanodsp-optimize=schedule-file=schedules/matmul-8x12-neon.mlir
//
// Ops are addressed by the tag -nanodsp-optimize assigns in walk order, so
// "op0" is the first linalg.generic in the module: the matmul in
// test/Integration/Schedule/bit-exact.mlir, which runs this file. Every
// other op is left untouched (scalar loops).
//
// Loop dims are (m, n, k). The reduction tile stays 1 so accumulation order,
// and therefore every result bit, matches the unscheduled lowering.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.generic"]}
        attributes {nanodsp.tag = "op0"} in %root
        : (!transform.any_op) -> !transform.any_op

    // Cache tile: 32 rows x all 48 columns x 32 of k.
    %cache, %cache_loops:2 = transform.structured.tile_using_for %mm
        tile_sizes [32, 0, 32]
        : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Register tile: 8 rows x 12 columns (3 NEON vectors), one k at a time.
    %reg, %reg_loops:3 = transform.structured.tile_using_for %cache
        tile_sizes [8, 12, 1]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op,
                                  !transform.any_op, !transform.any_op)

    transform.structured.vectorize %reg : !transform.any_op

    %funcs = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %funcs {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.yield
  }
}
