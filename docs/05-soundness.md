# Why the schedule can't change results 🧪

Every schedule `-nanodsp-optimize` generates produces output that is
**bit-identical** to the unscheduled lowering, for matmul, conv2d (including
strided and dilated) and the elementwise ops. That's a stronger claim than
"within tolerance", and here is why it holds.

## The argument

Floating-point addition isn't associative, so a schedule can only change a
result by changing **which operations** produce an output element or **in
what order** they happen. Each schedule step leaves both alone:

1. **Tiling parallel dims** changes which output elements are computed next
   to each other, not how any single one is computed.
2. **Tiling the reduction dim into cache blocks** splits `for k in 0..K` into
   `for k0 in 0..K step kc: for k in k0..k0+kc`. Same sequence of `k`, same
   order.
3. **Register tiles have reduction size 1.** Each vector step adds exactly one
   product per output element, so accumulation stays
   `((0 + a₀b₀) + a₁b₁) + …` in increasing `k`, just like the naive loop nest
   (and the C++ reference in `reference/`).
4. **Vectorization is elementwise.** The body becomes `arith.mulf` then
   `arith.addf` on vectors. No `vector.contract` or `vector.fma` is formed,
   and no fast-math flags are added.
5. **LLVM won't fuse them.** Without the `contract` fast-math flag, LLVM's
   default `-fp-contract` behavior never turns a separate `fmul` and `fadd`
   into an FMA. The C++ reference is built with `-ffp-contract=off` for the
   same reason.
6. **Folding unit dims** (the conv path) only reshapes. Extent-1 dims carry
   no arithmetic.

## The evidence

`test/Integration/Schedule/bit-exact.mlir` runs matmul (several cache tiles
on both targets), three conv variants and add+relu on fractional inputs chosen
so partial sums round. It prints every output as its raw 32-bit pattern and
`diff`s four runs against each other: unscheduled, the `host-neon` schedule,
the `x86-avx2` schedule, and the hand-written `schedules/matmul-8x12-neon.mlir`.

A test like this only means something if it can fail. Swapping in a schedule
that really reassociates (`transform.structured.split_reduction` with 4
partial sums) changes 2,503 of the printed values.

## What isn't covered

- **Hand-written schedules can break this.** Nothing stops a schedule from
  using `split_reduction`, a contraction, or a register tile over `k` lowered
  with a tree reduction. Run such a schedule through the bit-exact test
  before trusting it.
- **Different bits from other implementations are expected in general**
  (Mojo, CUDA, anything that uses FMA). The repo keeps them bit-exact only
  by applying the same no-FMA, naive-order discipline everywhere.
