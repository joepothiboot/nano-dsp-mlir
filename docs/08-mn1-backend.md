# MN1 backend: bringing up a scratchpad accelerator

MN1 is the fictional accelerator of [`mock-npu-bench`](https://github.com/joepothiboot/mock-npu-bench):
an in-order core with an 8-lane f32 vector unit, a 64 KiB scratchpad and an
asynchronous DMA engine, defined only by its manual (`docs/manual.md`,
rev 0.3) and checked by a cycle-level simulator. This page is the bring-up
log for compiling the `dsp` dialect to it: what the manual forced the
compiler to do, what broke on the way, and how fast the result is.

All cycle counts are **simulated** by `mnpu-sim`; every run below was
checked bit for bit against `reference/nanodsp_ref.h` before its cycles
were recorded.

## Pipeline

```bash
build/bin/nanodsp-opt kernel.mlir -convert-dsp-to-linalg \
    -nanodsp-optimize=target=mn1 -nanodsp-bufferize \
    -nanodsp-promote-local=target=mn1 \
  | build/bin/nanodsp-translate --mlir-to-mn1 > kernel.s
../mock-npu-bench/build/mnpu-sim kernel.s --load 0 a.f32 --load 0x40000 b.f32 \
    --expect-bits 0x80000 c_ref.f32
```

`nanodsp-golden matmul M K N DIR` writes the inputs and the reference
output. Arguments live in 256 KiB DRAM slots (0x0, 0x40000, ...), the
result in the next slot; the assembly starts with that layout as comments.

| stage                             | what it does for MN1                                                                                                  |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `TargetModel` `mn1`               | 256-bit vectors, 16 registers, no cache, 64 KiB local memory; tiles are budgeted against half the scratchpad          |
| `-nanodsp-promote-local`          | stages input and output tiles through the scratchpad; stages whole operands where no cache tile exists                |
| `nanodsp-translate --mlir-to-mn1` | instruction selection, register assignment and DMA lowering, straight from the promoted vector IR (`lib/Target/MN1/`) |

## What the manual forced

| MN1 fact (manual)                                                            | Consequence                                                                   | What the compiler does                                                                                                                                                                                                                                                                                                                        |
| ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| No cache; the core can only address the scratchpad                           | Every operand the compute touches must be staged, outputs included            | `-nanodsp-promote-local` promotes output tiles (load before, store after a loop they do not vary in; load and store per iteration otherwise) and, on a target without a cache, stages whole buffers around loop nests that still touch DRAM. Output promotion is opt-in (`promote-outputs`) for targets with a cache, so Hexagon is unchanged |
| No cache size to budget tiles against                                        | The tile-size model had nothing to fit                                        | Budget against `localMemBytes × cacheFraction` when `cacheBytes` is 0                                                                                                                                                                                                                                                                         |
| Elementwise ops got no cache tile at all                                     | A 128×128 add does not fit 64 KiB whole                                       | Cacheless targets tile elementwise ops too                                                                                                                                                                                                                                                                                                    |
| DMA moves one contiguous run; 8 tags; a tag cannot be reused before its wait | A strided tile (rows of a wider matrix) is many transfers, each needing a tag | One row per transfer on the tile's tag, waiting each row before the next; the last row is waited by the IR's `dma_wait`. The core blocks while the earlier rows move                                                                                                                                                                          |
| DMA tags are immediates in the encoding; no divide                           | `memref::multiBuffer` selects the slot with `(iv floordiv step) mod 2`        | A parity register per double-buffered loop, toggled at the latch; the slot is that register (or 1 − it for the prefetch), and an instruction that needs a tag branches on it                                                                                                                                                                  |
| Only `bne` (no set-less-than)                                                | The prefetch guard is `iv + step < ub`                                        | Lowered to `!=` after proving equivalence (static bounds, whole number of iterations)                                                                                                                                                                                                                                                         |
| The core cannot store to DRAM                                                | `linalg.fill` of the result buffer                                            | Memset by DMA: zero a 512-byte scratchpad block, store it over the buffer in chunks                                                                                                                                                                                                                                                           |
| No vector max                                                                | `dsp.relu` lowers to `arith.maximumf`                                         | Rejected with an error (`test/MN1/unsupported.mlir`); needs ISA support                                                                                                                                                                                                                                                                       |
| 16 scalar registers, 16 vector registers                                     | No spill code                                                                 | Values own registers and free them after their last use in their block; borrowed values (loop variables, parity, zero-offset views) never free; a fused `vmul`+`vadd` per 8-lane part needs one temporary                                                                                                                                     |

Bugs found during bring-up, each fixed and covered by a test:

- Nest staging checked its own bytes but not the tile buffers already
  allocated, so some tile shapes passed the pass and failed in the backend's
  allocator. The pass now counts local memory already in use.
- In the backend, an unqualified `unique(regs)` resolved to `llvm::unique`
  through argument-dependent lookup, reordering the register list and always
  returning true. A compiler warning ("unused function") gave it away.
- clang-tidy's `modernize-return-braced-init-list` rewrote
  `SmallVector<T>(n, value)` to `{n, value}`, a two-element list; it only
  failed because the narrowing made it a hard error. The check is disabled.

## Results

Matmul 128×128×128, f32, multiply and add unfused (8 flop/cycle is the
unfused peak at one vector instruction per cycle; 16 with `vfma`):

| kernel                                              |    cycles | instructions |  IPC | raw stalls | DMA-wait stalls | flop/cycle | of unfused peak |
| --------------------------------------------------- | --------: | -----------: | ---: | ---------: | --------------: | ---------: | --------------: |
| compiled (model tile 32×128×16, register tile 4×16) | 3,362,848 |    2,051,872 | 0.61 |      35.1% |            1.8% |       1.25 |           15.6% |
| hand-tiled (`tools/gen_matmul_tiled.py`)            | 1,149,328 |      844,625 | 0.73 |      20.0% |            0.9% |       3.65 |           45.6% |

Other compiled kernels, all bit-exact: matmul 32³ (staged whole) 53,140
cycles; matmul 64×128·128×32 421,273; add 128×128 (tiled, 16-row tiles,
double-buffered inputs) 34,216. The hand-written resident 32³ matmul
(`gen_matmul.py --unroll 4 --unfused`) takes 40,388.

### Why the compiled kernel is 2.9× slower

- **The accumulator is reloaded every `k` step.** The register tile keeps
  `k = 1`, so each step loads and stores eight accumulator vectors for 16
  multiply-adds (the README's "known limitation"). The hand kernel keeps a
  4×16 block in registers for a whole 32-row chunk.
- **Address arithmetic is recomputed in the inner loops**, with `mul` (three
  cycles) for every subview offset; the hand kernel walks pointers with
  `addi`.
- **One product temporary.** Each `vadd` waits for its `vmul` (four-cycle
  latency); the hand kernel alternates two.
- **Tags by branch.** The hand kernel unrolls the outer loops in the
  generator (modulo variable expansion), so every tag is static.

None of these is a tile-size problem, which the sweep confirms.

## Tile sweep

`scripts/sweep-mn1.sh` compiles matmul 128³ with 60 hand-set cache tiles
(`tm ∈ {8, 16, 32, 64}`, `tn ∈ {32, 64, 128}`, `tk ∈ {8, 16, 32, 64, 128}`,
register tile 4×16) plus the model's own pick, and runs each on the
simulator. Raw data: `benchmarks/results/sweep-mn1-matmul128.csv`.

- 54 of 60 tiles fit and all are bit-exact; the 6 that need more than
  64 KiB (double-buffered) are rejected by the pass.
- **The model's pick (32, 128, 16) is within 1% of the best tile**
  (32, 128, 64: 3,330,700 cycles). The worst, (8, 32, 8), takes 4,262,548.
- **Raw stalls barely move** (1.180 M to 1.198 M across all tiles): the
  register-level loop is identical. The 28% spread is DMA-wait stalls (43 K to 659 K):
  narrow tiles (`tn = 32`) fetch many short rows, and each transfer pays
  MN1's 40-cycle setup.
- So on MN1, as on the M2 (`02-tiling-model.md`), cache-tile choice moves
  little; what separates the compiled kernel from the hand one is the
  register schedule.

## Next steps

1. Hoist the accumulator out of the `k` loop (planned in the compiler work
   after this phase); on MN1 it removes most of the 2.9× gap's first cause.
2. Strength-reduce subview addresses in the backend.
3. Unroll double-buffered loops by two so tags become static.

## Reproduce

```bash
./test.sh                                   # includes test/MN1/, end-to-end when mnpu-sim is found
NANODSP_BIN=build/bin scripts/sweep-mn1.sh  # needs ../mock-npu-bench/build/mnpu-sim
```

The end-to-end lit tests (`test/MN1/e2e-*.mlir`) find the simulator through
`MNPU_SIM` or a sibling `mock-npu-bench` checkout, and are reported as
unsupported otherwise.

## Limits

- Simulated cycles from one model of a fictional machine; MN1 timing is what
  its manual says, nothing more.
- `dsp.add` and `dsp.matmul` on f32; `relu` (no vector max) and `qmatmul`
  (no integer vector unit) are rejected.
- Static shapes whose tiles divide the loops; buffers up to 256 KiB each.
- No spilling: a schedule that needs more than 12 scalar or 16 vector
  registers at once is an error.
