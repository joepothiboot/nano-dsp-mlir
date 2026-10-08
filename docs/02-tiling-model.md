# Tiling model 📐

`-nanodsp-optimize` tiles every `linalg.generic` twice, a cache tile and a
register tile inside it, and then vectorizes the register tile. The sizes are
not hardcoded. They come from a small analytical model
(`lib/Schedule/TileSizeModel.cpp`) fed by a compile-time `TargetModel`
(`lib/Schedule/TargetModel.cpp`).

## 🎯 Target models

| Target      | Vector width | Vector regs | Cache the vector unit reads  | Budget (50%) |
| ----------- | ------------ | ----------- | ---------------------------- | ------------ |
| `host-neon` | 128 bit      | 32          | 128 KiB L1D (Apple M-series) | 64 KiB       |
| `x86-avx2`  | 256 bit      | 16          | 32 KiB L1D                   | 16 KiB       |

`cacheFraction = 0.5` is a starting estimate. The matmul sweep below tests it
on `host-neon` only.

## 🧮 Register tile

The register tile is what one vectorized step computes. For an op with a
reduction, it holds `mr` rows of the output by `nv` vectors of it:

- **Reduction dims are always 1.** Each output element is then accumulated in
  the same order as the naive loop nest, which is what keeps scheduled code
  bit-exact (see [`05-soundness.md`](05-soundness.md)).
- **The vector dim** is the output's innermost (contiguous) dim. It gets
  `nv × lanes` elements.
- **The row dim** is the output's next dim out. It gets `mr` elements.
- `(mr, nv)` maximizes arithmetic intensity `mr·nv / (mr + nv)` subject to
  `mr·nv + nv + 1 ≤ numVectorRegs`: accumulators, plus one row of the other
  operand, plus one broadcast value. `nv ≤ 4` bounds unrolling.

| Target      | Best `(mr, nv)` | Register tile before snapping |
| ----------- | --------------- | ----------------------------- |
| `host-neon` | (6, 4)          | 6 × 16                        |
| `x86-avx2`  | (4, 3)          | 4 × 24                        |

A dim that an input reads with a stride (a conv with stride 2 reads `ow * 2`)
can't be loaded as one contiguous vector, so it gets a register tile of 1.

Ops without a reduction (add, relu) use one row of 4 vectors.

## 📦 Cache tile

Only reductions get a cache tile. Elementwise ops reuse nothing, so there is
nothing to block for. The model starts from the register tile and grows it
round-robin: reduction dims first (a longer `k` amortizes accumulator traffic),
then rows, then columns. Each step moves to the next extent that is a multiple
of the register tile. Growth stops when the next step would exceed the budget.

The working set is computed from the indexing maps rather than assumed to be
`mc·kc + kc·nc + mc·nc`. Each operand touches `Π (expr(tile − 1) − expr(0) + 1)`
elements, so a conv's input window (`oh + kh`, `ow + kw`, with strides and
dilations) is counted exactly.

Worked example, matmul `128×256 · 256×96`, from `test/Schedule/emit-schedule.mlir`:

| Target      | Cache tile `(m, n, k)` | Working set                            | Register tile |
| ----------- | ---------------------- | -------------------------------------- | ------------- |
| `host-neon` | 64 × 96 × 64           | (64·64 + 64·96 + 64·96) · 4 B = 64 KiB | 4 × 16        |
| `x86-avx2`  | 32 × 96 × 8            | (32·8 + 8·96 + 32·96) · 4 B = 16 KiB   | 4 × 24        |

Both land exactly on their budget.

## ✂️ Divisor snapping

Every tile size divides its loop extent. Every tile then has a static shape,
and the vectorizer never needs masking or a remainder loop. The cost is that
awkward extents lose blocking:

- The 128-row matmul above wanted `mr = 6` and got 4, the largest divisor of
  128 that is ≤ 6.
- A prime extent degrades to a tile of 1.

Masked vectorization (`vector_sizes` on `transform.structured.vectorize`) or
peeling would remove this restriction. It's the first thing to try when a
benchmark shows a shape that suffers from it.

## 🔬 Sweep: cache tile of matmul 512³ on Apple M2

`scripts/sweep.sh` compiles `matmul` 512³ with 48 hand-set cache tiles
(`m ∈ {16, 64, 256}`, `n ∈ {16, 64, 128, 512}`, `k ∈ {8, 32, 128, 512}`) and
the model's register tile 4×16×1, then times each one with
`benchmarks/sweep.cpp`. Every tile is checked bit for bit against a scalar
loop nest. Raw numbers: `benchmarks/results/sweep-m2-matmul512.csv`
(one run, 10 samples per tile, best time).

- **The surface is flat.** All 48 tiles land between 29.2 and 36.3 GFLOP/s.
  The model's pick (64, 128, 32; 56 KiB, 0.44 of L1D) reaches 34.5, within 5%
  of the best tile (36.3).
- **`k` matters most.** `k = 8` and `k = 32` reach 31–36 GFLOP/s, `k ≥ 128`
  only 29–31. Long `k` blocks lose about 12%, not gain: the 16 MiB L2 holds a
  whole operand, so blocking for L1 saves little traffic.
- **The working-set budget is not what separates tiles.** Tiles at 7× L1D
  (`256, 512, 8`) run as fast as ones at 0.05× (`16, 64, 8`). Raising or
  lowering `cacheFraction` would barely change this shape.
- **Limits.** One shape, one machine, single runs of about 1% noise. This says
  the model is not wrong for matmul 512³ on an M2; it does not show the budget
  is right, and `x86-avx2` and the conv2d tiles are untested.

## 🔗 Where the numbers go

The model's output is a Transform-dialect schedule, not a C++ transformation.
Run `-nanodsp-emit-schedule=target=<name>` to see it, or check a hand-edited
copy into `schedules/` and apply it with
`-nanodsp-optimize=schedule-file=<path>`. Both go through the same code path.
[`04-schedule-ir-diff.md`](04-schedule-ir-diff.md) shows the IR before and
after.
