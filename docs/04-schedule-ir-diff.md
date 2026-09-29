# Schedule IR, before and after 🔬

One matmul, `16×96 · 96×32`, scheduled for `x86-avx2`. All IR below is real
`nanodsp-opt` output, lightly trimmed.

## Before: L2, one `linalg.generic`

```bash
nanodsp-opt matmul.mlir -convert-dsp-to-linalg
```

```mlir
%1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x32xf32>) -> tensor<16x32xf32>
%2 = linalg.generic {
       indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                        affine_map<(m, n, k) -> (k, n)>,
                        affine_map<(m, n, k) -> (m, n)>],
       iterator_types = ["parallel", "parallel", "reduction"]}
     ins(%arg0, %arg1 : tensor<16x96xf32>, tensor<96x32xf32>)
     outs(%1 : tensor<16x32xf32>) {
^bb0(%in: f32, %in_0: f32, %out: f32):
  %3 = arith.mulf %in, %in_0 : f32
  %4 = arith.addf %out, %3 : f32
  linalg.yield %4 : f32
} -> tensor<16x32xf32>
```

## The schedule

```bash
nanodsp-opt matmul.mlir -convert-dsp-to-linalg -nanodsp-emit-schedule=target=x86-avx2
```

```mlir
%0 = transform.structured.match ops{["linalg.generic"]}
       attributes {nanodsp.tag = "op0"} in %arg0
// Cache tile: k in blocks of 48. Rows (16) and columns (32) already fit.
%tiled, %loops = transform.structured.tile_using_for %0 tile_sizes [0, 0, 48]
// Register tile: 4 rows x 16 columns (2 AVX2 vectors), one k at a time.
%tiled_0, %loops_1:3 = transform.structured.tile_using_for %tiled tile_sizes [4, 16, 1]
// Drop the tile's unit k dim, find the 2-D op again, vectorize it.
transform.apply_patterns to %funcs {
  transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
  transform.apply_patterns.canonicalization
}
%5 = transform.structured.match ops{["linalg.generic"]} in %loops_1#2
transform.structured.vectorize %5
```

The model wanted a 24-wide register tile (3 vectors). 32 has no divisor that
is a multiple of 8 and ≤ 24, so it snapped to 16. Growing `k` from 48 to 96
would take the working set from 11 KiB to 20 KiB, over the 16 KiB budget.

## After: L3, loops around a vector body

```bash
nanodsp-opt matmul.mlir -convert-dsp-to-linalg -nanodsp-optimize=target=x86-avx2
```

```mlir
%3 = scf.for %k0 = %c0 to %c96 step %c48 iter_args(%acc0 = %2) {        // cache: k
  %4 = scf.for %m = %c0 to %c16 step %c4 iter_args(%acc1 = %acc0) {      // reg: rows
    %5 = scf.for %n = %c0 to %c32 step %c16 iter_args(%acc2 = %acc1) {   // reg: cols
      %6 = scf.for %k = %c0 to %c48 step %c1 iter_args(%acc3 = %acc2) {  // reg: k
        // A column (4) broadcast across n, B row (16) broadcast across m.
        %7  = vector.transfer_read %a_col[%c0] : tensor<4xf32>, vector<4x16xf32>
        %8  = vector.transfer_read %b_row[%c0] : tensor<16xf32>, vector<4x16xf32>
        %9  = vector.transfer_read %c_tile[%c0, %c0] : tensor<4x16xf32>, vector<4x16xf32>
        %10 = arith.mulf %7, %8 : vector<4x16xf32>
        %11 = arith.addf %9, %10 : vector<4x16xf32>
        %12 = vector.transfer_write %11, %c_tile[%c0, %c0] : vector<4x16xf32>, tensor<4x16xf32>
        ...
```

Things to notice:

- **No `vector.contract`.** With a reduction tile of 1 the vectorizer emits a
  plain `mulf` and `addf`. A contraction would lower to FMA and change
  rounding.
- **No unit dims.** Vectorizing the `4×16×1` tile directly gives
  `vector<4x16x1xf32>`. LLVM lowers that trailing 1 to scalar multiplies
  (`vmulss` on x86), so the schedule folds unit dims away first.
  `test/Schedule/codegen.mlir` checks the machine code: 16 `fmul.4s` on NEON,
  12 `vmulps` on AVX2, no scalar multiplies, no FMA.
- **Still on tensors.** L3 has no memory yet. `-nanodsp-lower-to-llvm`
  bufferizes and lowers the rest of the way.
- **The accumulator round-trips every `k`.** The C tile is read and written
  inside the innermost loop. Hoisting it into registers across `k`
  (`transform.structured.hoist_redundant_vector_transfers`, after
  bufferization) is the obvious next optimization. It isn't done yet.

## Conv needs the fold too

A conv's input map `(n, oh + kh, ow + kw, c)` isn't a projected permutation,
so the vectorizer rejects the op even when `oh`, `kh` and `kw` all have
extent 1 in the register tile. The same unit-dim fold leaves a 2-D `(ow, f)`
generic, which vectorizes into a `4×8` multiply-add. The fold replaces the op
and drops its `nanodsp.tag`, so every tile is found again through its
innermost loop handle. See `test/Schedule/optimize.mlir`.
