module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.generic"]}
        attributes {nanodsp.tag = "op0"} in %root
        : (!transform.any_op) -> !transform.any_op

    %cache, %cache_loops:2 = transform.structured.tile_using_for %mm
        tile_sizes [32, 0, 32]
        : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op)

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
