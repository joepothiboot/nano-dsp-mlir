// Negative control for bit-exact.mlir: a schedule that really reassociates.
// split_reduction computes 4 partial sums over k and adds them at the end,
// which changes rounding. bit-exact.mlir expects this run to differ from the
// reference; if it didn't, the diff checks there would prove nothing.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.generic"]}
        attributes {nanodsp.tag = "op0"} in %root
        : (!transform.any_op) -> !transform.any_op
    %init, %fill, %split, %combine = transform.structured.split_reduction %mm
        {split_factor = 4, insert_split_dimension = 2}
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op,
                                  !transform.any_op, !transform.any_op)
    transform.yield
  }
}
