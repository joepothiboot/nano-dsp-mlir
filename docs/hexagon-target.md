# 🔶 Hexagon target

`-nanodsp-optimize=target=hexagon-hvx128` tiles and vectorizes for a Hexagon V68
core with 128-byte HVX vectors, and `scripts/run-hexagon.sh` runs the compiled
kernels on an emulated core and compares every result bit for bit with the
scalar C++ reference.

## 📐 The target model

The numbers live in `lib/Schedule/TargetModel.cpp`, taken from the Hexagon V68
HVX Programmer's Reference Manual (80-N2040-47):

| Field           | Value   | Source                                      |
| --------------- | ------- | ------------------------------------------- |
| `vectorBits`    | 1024    | 128-byte mode (sec. 1.2.1)                  |
| `numVectorRegs` | 32      | V0-V31 (sec. 2.1)                           |
| `cacheBytes`    | 512 KiB | L2, an **assumption** (see below)           |
| `localMemBytes` | 256 KiB | VTCM, an **assumption** (see below)         |
| `cacheFraction` | 0.5     | same starting estimate as the other targets |

HVX loads and stores bypass the scalar core's L1 data cache and go to L2, L2TCM
or VTCM, so the working-set model sizes cache tiles against **L2**, not L1.
The manual leaves the L2 and VTCM sizes implementation-defined (sec. 3.2). The
512 KiB and 256 KiB above are assumptions for a small part. They have not been
checked against a specific chip, which makes the tile sizes for this target a
model, not a measurement.

## 🚀 Running it

```bash
./test.sh                                                  # builds nanodsp-opt
docker build --platform linux/amd64 -t nanodsp-hexagon docker/hexagon
scripts/run-hexagon.sh
```

Everything is compiled on the host (`nanodsp-opt`, then `mlir-translate` and
`llc` for the kernels, Homebrew `clang++` for the harness). The Docker image
holds the CodeLinaro Hexagon toolchain's musl, libc++ and compiler-rt, links the
objects, and runs the result under `qemu-hexagon`. The image is `linux/amd64`
and is emulated on Apple silicon, so it is slow but needs no hardware.

`test/Hexagon/harness.cpp` calls the MLIR-compiled kernels
(`kernels-f32.mlir`, `kernels-i8.mlir`) and the reference on the same inputs, on
the same emulated core, and compares every value bit for bit.

## 🧱 What the lowering changes

Two options on `-nanodsp-lower-to-llvm` exist for this target:

- Function arguments and results become identity-layout memrefs, so a kernel
  marked `llvm.emit_c_interface` is callable from C with a plain descriptor
  struct.
- `generic-alloc` allocates through `_mlir_memref_to_llvm_alloc` and
  `_mlir_memref_to_llvm_free`, which the embedding program provides. Hexagon is a
  32-bit target but `index`, and so the allocation size, stays 64-bit, which does
  not match a 32-bit libc's `malloc(size_t)`.

## ✅ What is verified

| Build      | Kernels                                   | Result                                   |
| ---------- | ----------------------------------------- | ---------------------------------------- |
| `scalar`   | unscheduled f32 + unscheduled int8        | ✅ all five checks bit-identical         |
| `hvx-int`  | unscheduled f32 + HVX-scheduled int8      | ✅ all five checks bit-identical         |
| `hvx-ieee` | HVX f32 (IEEE, `+hvx-ieee-fp`) + HVX int8 | ⏳ compiles; needs `--hvx-fp`, see below |
| `hvx-qf32` | HVX f32 (QFloat) + HVX int8               | ⏳ compiles; needs `--hvx-fp`, see below |
| `local`    | `hvx-int` + VTCM-promoted f32 and int8    | ✅ all seven checks bit-identical        |

The two f32-on-HVX builds need a `qemu-hexagon` with the HVX floating-point
instructions. The image's QEMU 8.2 lacks them, so `scripts/run-hexagon.sh
--hvx-fp` (which uses the binary named by `$QEMU_HEXAGON`) has not been run
here. Until it is, only the integer HVX path is checked on the emulator.

The `local` build adds a 128x256x128 f32 matmul and a 128x1024x128 int8
`qmatmul` whose cache tiles are double-buffered through VTCM
(`-nanodsp-lower-to-llvm=local-target=hexagon-hvx128`). The DMAs run as
synchronous copies, and the f32 kernel is compiled without HVX features for
the same QEMU reason. See [`scratchpad-dma.md`](scratchpad-dma.md).

## 🧱 Local memory (VTCM)

`localMemBytes` is used by `-nanodsp-promote-local`, which stages the input
tiles of each cache tile through `#dsp.local` buffers with double-buffered
`memref.dma_start`/`dma_wait`, and `-nanodsp-lower-local`, which lowers the
DMAs to copies for the host and the emulator. The design, IR before and
after, and what is and isn't modeled are in
[`scratchpad-dma.md`](scratchpad-dma.md).

## 🚧 Limits

- Emulated, not measured: no timing or throughput is claimed for Hexagon.
  QEMU models neither VTCM nor DMA timing, so the local-memory build checks
  correctness only.
- The L2 and VTCM sizes are assumptions, as above.
- Tile sizes must still divide loop extents, as on the other targets.
