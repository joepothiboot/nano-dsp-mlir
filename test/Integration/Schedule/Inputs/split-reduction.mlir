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
