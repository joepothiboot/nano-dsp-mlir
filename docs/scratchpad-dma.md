# 🧱 Scratchpad and DMA lowering

Accelerators such as Hexagon have a software-managed local memory next to
the vector unit (VTCM on Hexagon) instead of, or besides, a cache the
compiler cannot steer. Using it means the compiler has to decide what lives
there, allocate it, and move data in with DMA, ideally while the previous
tile is still being computed on. This document describes how nano-dsp-mlir
does that for the cache tiles its schedules already produce.

Two passes do the work, both in `lib/Schedule/LocalMemory.cpp`:

- `-nanodsp-promote-local=target=<name>` stages the input tiles of every
  cache tile through local memory, with double-buffered DMAs.
- `-nanodsp-lower-local` lowers those DMAs to synchronous copies for a
  machine without a DMA engine (the host, `qemu-hexagon`).

## 🗺️ Where it sits in the pipeline

```
-convert-dsp-to-linalg
-nanodsp-optimize=target=hexagon-hvx128     tile + vectorize, mark the cache loop
-nanodsp-bufferize                          one-shot-bufferize + deallocation
-nanodsp-promote-local=target=...           #dsp.local buffers, dma_start/dma_wait
-nanodsp-lower-local                        DMAs -> linalg.copy, #dsp.local -> default
-nanodsp-lower-bufferized-to-llvm           the rest of the LLVM lowering
```

Promotion needs memrefs (DMAs move buffers, not values), so it runs after
bufferization. It also needs the loop structure the schedule created, so it
runs before `convert-linalg-to-loops` and the vector lowering. That is why
`-nanodsp-lower-to-llvm` is now two registered halves,
`-nanodsp-bufferize` and `-nanodsp-lower-bufferized-to-llvm`. Without the
new option it runs exactly the passes it ran before.

The whole pipeline is one `nanodsp-opt` invocation, which is also the form
to use with `-mlir-print-ir-after-all`:

```bash
build/bin/nanodsp-opt test/Hexagon/kernels-local-f32.mlir -convert-dsp-to-linalg -nanodsp-optimize=target=hexagon-hvx128 -nanodsp-lower-to-llvm=local-target=hexagon-hvx128
```

`local-target=` inserts `nanodsp-promote-local` and `nanodsp-lower-local`
after the bufferization half. With `-mlir-print-ir-after-all` the trace
shows `NanoDSPPromoteLocal` right after the deallocation passes and
`NanoDSPLowerLocal` right before `ConvertLinalgToLoopsPass`. To stop at the
DMA form, run the halves by hand:

```bash
build/bin/nanodsp-opt test/Hexagon/kernels-local-f32.mlir -convert-dsp-to-linalg \
  -nanodsp-optimize=target=hexagon-hvx128 -nanodsp-bufferize \
  -nanodsp-promote-local=target=hexagon-hvx128
```

## 🏷️ Pieces

**`#dsp.local`** is a memref memory space attribute of the `dsp` dialect
(`include/nanodsp/Dialect/DSP/IR/DSPAttrs.td`). It is a dialect attribute
rather than an integer address space, so a promoted buffer cannot be
mistaken for GPU address space 1.

**`nanodsp.cache_loop`** marks the innermost cache-tile loop. For a target
with `localMemBytes > 0` the generated schedule adds
`transform.annotate %<loop> "nanodsp.cache_loop"` after the cache-level
`tile_using_for`; other targets' schedules are unchanged. The attribute
survives bufferization. A hand-written schedule opts in with the same
annotate line. Ops without a cache tile (elementwise ops, or any op whose
working set fits the budget whole) have no marker and are not promoted.

**The target model.** `hexagon-hvx128` has `localMemBytes = 256 KiB`, an
assumption for a small part (see [`hexagon-target.md`](hexagon-target.md)).
`host-neon` and `x86-avx2` have none, so `-nanodsp-promote-local` leaves
their IR unchanged.

## ⚙️ What `-nanodsp-promote-local` does

In each marked loop, every `memref.subview` that

- takes a tile of a buffer defined outside the loop,
- is only read (by `vector.transfer_read`, `memref.load`, or as a linalg
  input, possibly through further subviews), and
- can be moved by one `memref.dma_start` (one contiguous run, or rows of
  equal length a fixed stride apart)

gets a buffer in `#dsp.local`, allocated once in front of the whole
cache-tile nest and freed after it. The compute is redirected to the local
buffer, with the subviews in between recreated for the new layout.

- **A tile that changes with the loop** is double-buffered. The buffer and
  its DMA tag become two slots each (upstream `memref::multiBuffer`), the
  first tile is loaded before the loop, and iteration `i` issues the DMA
  for tile `i + step` into the other slot before it waits for its own.
- **A tile that does not change with the loop** (the A tile when the
  innermost cache loop is `n`) gets one buffer and is loaded once in front
  of the loop.
- **Budget.** If the double-buffered tiles do not fit `localMemBytes`, the
  pass falls back to single buffering (issue and wait at the top of each
  iteration). If even that does not fit, it is an error at the loop.
  `double-buffer=false` forces single buffering.

The output tile is not promoted: it stays in main memory and is written in
place, as before.

### Before

The `hexagon-hvx128` schedule for a 128x256 by 256x128 matmul
(`test/Hexagon/kernels-local-f32.mlir`) cuts `k` into two 128-wide cache
tiles. After `-nanodsp-bufferize`, the cache loop takes its two input tiles
as subviews of the arguments (register-tile body elided):

```mlir
    scf.for %arg2 = %c0 to %c256 step %c128 {
      %subview = memref.subview %arg0[0, %arg2] [128, 128] [1, 1] : memref<128x256xf32> to memref<128x128xf32, strided<[256, 1], offset: ?>>
      %subview_0 = memref.subview %arg1[%arg2, 0] [128, 128] [1, 1] : memref<256x128xf32> to memref<128x128xf32, strided<[128, 1], offset: ?>>
      scf.for %arg3 = %c0 to %c128 step %c4 {
        scf.for %arg4 = %c0 to %c128 step %c1 {
          %subview_1 = memref.subview %subview[%arg3, %arg4] [4, 1] [1, 1] : memref<128x128xf32, strided<[256, 1], offset: ?>> to memref<4x1xf32, strided<[256, 1], offset: ?>>
          ...
        }
      }
    } {nanodsp.cache_loop}
```

### After `-nanodsp-promote-local=target=hexagon-hvx128`

Two slots per tile, the first tiles loaded before the loop, the next tiles
prefetched under `scf.if` before the waits. The A tile is 128 rows of 128
elements, 256 apart in the source (`%c256, %c128` are the stride and the
elements per stride); the B tile is one contiguous run of 16384 elements:

```mlir
    %alloc_0 = memref.alloc() {alignment = 128 : i64} : memref<2x128x128xf32, #dsp.local>
    %alloc_1 = memref.alloc() : memref<2x1xi32>
    %alloc_2 = memref.alloc() {alignment = 128 : i64} : memref<2x128x128xf32, #dsp.local>
    %alloc_3 = memref.alloc() : memref<2x1xi32>
    %subview = memref.subview %alloc_1[%c0, 0] [1, 1] [1, 1] : memref<2x1xi32> to memref<1xi32, strided<[1], offset: ?>>
    %subview_4 = memref.subview %alloc_0[%c0, 0, 0] [1, 128, 128] [1, 1, 1] : memref<2x128x128xf32, #dsp.local> to memref<128x128xf32, strided<[128, 1], offset: ?>, #dsp.local>
    %c16384 = arith.constant 16384 : index
    memref.dma_start %arg0[%c0, %c0], %subview_4[%c0, %c0], %c16384, %subview[%c0], %c256, %c128 : memref<128x256xf32>, memref<128x128xf32, strided<[128, 1], offset: ?>, #dsp.local>, memref<1xi32, strided<[1], offset: ?>>
    %subview_5 = memref.subview %alloc_3[%c0, 0] [1, 1] [1, 1] : memref<2x1xi32> to memref<1xi32, strided<[1], offset: ?>>
    %subview_6 = memref.subview %alloc_2[%c0, 0, 0] [1, 128, 128] [1, 1, 1] : memref<2x128x128xf32, #dsp.local> to memref<128x128xf32, strided<[128, 1], offset: ?>, #dsp.local>
    memref.dma_start %arg1[%c0, %c0], %subview_6[%c0, %c0], %c16384, %subview_5[%c0] : memref<256x128xf32>, memref<128x128xf32, strided<[128, 1], offset: ?>, #dsp.local>, memref<1xi32, strided<[1], offset: ?>>
    scf.for %arg2 = %c0 to %c256 step %c128 {
      %1 = affine.apply #map(%arg2)
      %subview_7 = memref.subview %alloc_3[%1, 0] [1, 1] [1, 1] : memref<2x1xi32> to memref<1xi32, strided<[1], offset: ?>>
      %subview_8 = memref.subview %alloc_2[%1, 0, 0] [1, 128, 128] [1, 1, 1] : memref<2x128x128xf32, #dsp.local> to memref<128x128xf32, strided<[128, 1], offset: ?>, #dsp.local>
      %subview_9 = memref.subview %alloc_1[%1, 0] [1, 1] [1, 1] : memref<2x1xi32> to memref<1xi32, strided<[1], offset: ?>>
      %subview_10 = memref.subview %alloc_0[%1, 0, 0] [1, 128, 128] [1, 1, 1] : memref<2x128x128xf32, #dsp.local> to memref<128x128xf32, strided<[128, 1], offset: ?>, #dsp.local>
      %2 = arith.addi %arg2, %c128 : index
      %3 = arith.cmpi slt, %2, %c256 : index
      scf.if %3 {
        %4 = affine.apply #map(%2)
        %subview_11 = memref.subview %alloc_1[%4, 0] [1, 1] [1, 1] : memref<2x1xi32> to memref<1xi32, strided<[1], offset: ?>>
        %subview_12 = memref.subview %alloc_0[%4, 0, 0] [1, 128, 128] [1, 1, 1] : memref<2x128x128xf32, #dsp.local> to memref<128x128xf32, strided<[128, 1], offset: ?>, #dsp.local>
        memref.dma_start %arg0[%c0, %2], %subview_12[%c0, %c0], %c16384, %subview_11[%c0], %c256, %c128 : memref<128x256xf32>, memref<128x128xf32, strided<[128, 1], offset: ?>, #dsp.local>, memref<1xi32, strided<[1], offset: ?>>
        %subview_13 = memref.subview %alloc_3[%4, 0] [1, 1] [1, 1] : memref<2x1xi32> to memref<1xi32, strided<[1], offset: ?>>
        %subview_14 = memref.subview %alloc_2[%4, 0, 0] [1, 128, 128] [1, 1, 1] : memref<2x128x128xf32, #dsp.local> to memref<128x128xf32, strided<[128, 1], offset: ?>, #dsp.local>
        memref.dma_start %arg1[%2, %c0], %subview_14[%c0, %c0], %c16384, %subview_13[%c0] : memref<256x128xf32>, memref<128x128xf32, strided<[128, 1], offset: ?>, #dsp.local>, memref<1xi32, strided<[1], offset: ?>>
      }
      memref.dma_wait %subview_9[%c0], %c16384 : memref<1xi32, strided<[1], offset: ?>>
      memref.dma_wait %subview_7[%c0], %c16384 : memref<1xi32, strided<[1], offset: ?>>
      scf.for %arg3 = %c0 to %c128 step %c4 {
        scf.for %arg4 = %c0 to %c128 step %c1 {
          %subview_11 = memref.subview %subview_10[%arg3, %arg4] [4, 1] [1, 1] : memref<128x128xf32, strided<[128, 1], offset: ?>, #dsp.local> to memref<4x1xf32, strided<[128, 1], offset: ?>, #dsp.local>
          ...
        }
      }
    } {nanodsp.cache_loop}
    memref.dealloc %alloc_0 : memref<2x128x128xf32, #dsp.local>
    memref.dealloc %alloc_1 : memref<2x1xi32>
    memref.dealloc %alloc_2 : memref<2x128x128xf32, #dsp.local>
    memref.dealloc %alloc_3 : memref<2x1xi32>
```

`#map` is `(d0) -> ((d0 floordiv 128) mod 2)`, the slot of an iteration.
The prefetch writes the slot that iteration `i - step` read; that compute
has finished, so a tile in use is never overwritten.

### After `-nanodsp-lower-local`

Each DMA becomes a `linalg.copy` of the same tile into its slot, waits and
tags are gone, and the buffers are in the default memory space (register
loop bodies elided):

```mlir
    %alloc_0 = memref.alloc() {alignment = 128 : i64} : memref<2x128x128xf32>
    %alloc_1 = memref.alloc() {alignment = 128 : i64} : memref<2x128x128xf32>
    %subview = memref.subview %alloc_0[%c0, 0, 0] [1, 128, 128] [1, 1, 1] : memref<2x128x128xf32> to memref<128x128xf32, strided<[128, 1], offset: ?>>
    %subview_2 = memref.subview %arg0[0, 0] [128, 128] [1, 1] : memref<128x256xf32> to memref<128x128xf32, strided<[256, 1]>>
    linalg.copy ins(%subview_2 : memref<128x128xf32, strided<[256, 1]>>) outs(%subview : memref<128x128xf32, strided<[128, 1], offset: ?>>)
    ...
    scf.for %arg2 = %c0 to %c256 step %c128 {
      ...
      scf.if %3 {
        %4 = affine.apply #map(%2)
        %subview_7 = memref.subview %alloc_0[%4, 0, 0] [1, 128, 128] [1, 1, 1] : memref<2x128x128xf32> to memref<128x128xf32, strided<[128, 1], offset: ?>>
        %subview_8 = memref.subview %arg0[0, %2] [128, 128] [1, 1] : memref<128x256xf32> to memref<128x128xf32, strided<[256, 1], offset: ?>>
        linalg.copy ins(%subview_8 : memref<128x128xf32, strided<[256, 1], offset: ?>>) outs(%subview_7 : memref<128x128xf32, strided<[128, 1], offset: ?>>)
        ...
      }
      scf.for %arg3 = %c0 to %c128 step %c4 {
        ...
      }
    } {nanodsp.cache_loop}
```

The copies are `linalg.copy`, not `memref.copy`: a strided `memref.copy`
lowers to a call into the MLIR C runner library, which the Hexagon harness
does not link; `linalg.copy` becomes plain loops.

### The budget error

Tiles that do not fit even single-buffered (`test/Schedule/promote-local.mlir`,
512x128 and 128x64 f32 tiles):

```
error: input tiles of this cache tile need 294912 bytes of local memory, but target 'hexagon-hvx128' has 262144
  scf.for %k = %c0 to %c256 step %c128 {
  ^
```

## ✅ What is verified

- `test/Schedule/promote-local.mlir`: buffer placement, two slots, DMA
  order (prologue, guarded prefetch, waits before compute), the hoisted
  invariant tile, the single-buffer fallback, the budget error, and no
  change for `host-neon`.
- `test/Schedule/lower-local.mlir`: DMA to `linalg.copy`, multi-buffer
  slots, removal of tags and `#dsp.local`, and the error for a DMA that is
  not in the promoted form.
- `test/Integration/Schedule/local-bit-exact.mlir` (runs in CI): a
  256x256x128 matmul and a 1x34x34x16 conv run on the host with the
  `hexagon-hvx128` schedule, unpromoted, double-buffered and
  single-buffered. All three print the same bits as the unscheduled
  reference. Temporarily making the prefetch load the current tile instead
  of the next one makes this test fail, so it does see the buffer rotation.
- `scripts/run-hexagon.sh`, build `local`: a 128x256x128 f32 matmul and a
  128x1024x128 int8 `qmatmul`, both promoted and double-buffered, under
  `qemu-hexagon` against the C++ reference. Real output:

```
== local
PASS matmul           3072/3072 values bit-identical
PASS conv2d           800/800 values bit-identical
PASS add+relu         240/240 values bit-identical
PASS qmatmul golden   8/8 values bit-identical
PASS qmatmul          2048/2048 values bit-identical
PASS matmul local     16384/16384 values bit-identical
PASS qmatmul local    16384/16384 values bit-identical
```

The int8 kernel runs on HVX. The f32 kernel keeps the HVX schedule but is
compiled without HVX features: the image's QEMU 8.2 has no HVX
floating-point instructions, so its 1024-bit vectors are legalized to
scalar IEEE code.

## 🚧 What is modeled, and what isn't

Modeled:

- Which tiles live in local memory, its capacity, and the allocation of
  each buffer once per cache-tile nest.
- The order of transfers: the next tile's DMA is issued before the wait for,
  and the compute on, the current one.
- DMA shapes: one contiguous run, or one level of stride (the form
  `memref.dma_start` expresses).

Not modeled:

- **Timing.** On the host and in `qemu-hexagon` the DMAs are synchronous
  copies, and QEMU models neither VTCM nor a DMA engine or their latency.
  Nothing here shows the overlap paying off, and no speedup is claimed.
- **A real DMA lowering.** Nothing maps `memref.dma_start` to Hexagon's DMA
  engine or to a VTCM allocator; on the emulator `#dsp.local` buffers are
  ordinary heap memory.
- **The output tile.** It is not promoted, so each `k` step still reads and
  writes the accumulator in main memory.
- **Tiles needing more than one level of stride** are left in place, as are
  whole operands used without a subview (the conv filter above).
- **Budget interplay.** Cache tiles are still sized against L2
  (`cacheFraction` of 512 KiB); the promotion checks VTCM afterwards and
  falls back to single buffering instead of resizing tiles.
- The 256 KiB VTCM size is an assumption, as in
  [`hexagon-target.md`](hexagon-target.md).
