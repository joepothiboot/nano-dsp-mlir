// f32 kernel for the local-memory variant of the on-target test
// (scripts/run-hexagon.sh, build "local"): the hexagon-hvx128 schedule cuts
// k into 128-wide cache tiles, -nanodsp-promote-local double-buffers the A
// and B tiles in VTCM (#dsp.local), and -nanodsp-lower-local turns the DMAs
// into copies, since qemu-hexagon models neither VTCM nor a DMA engine.
//
// llc gets no HVX features here, so the schedule's 1024-bit vectors are
// legalized to scalar IEEE code: the image's QEMU 8.2 lacks the HVX
// floating-point instructions (see kernels-f32.mlir). What runs is the
// promoted loop structure, not HVX float arithmetic.
//
// The RUN lines only check codegen (no emulator needed).
//
// RUN: nanodsp-opt %s -convert-dsp-to-linalg -nanodsp-optimize=target=hexagon-hvx128 \
// RUN:     -nanodsp-bufferize -nanodsp-promote-local=target=hexagon-hvx128 \
// RUN: | FileCheck %s --check-prefix=PROMOTED
// RUN: nanodsp-opt %s -convert-dsp-to-linalg -nanodsp-optimize=target=hexagon-hvx128 \
// RUN:     -nanodsp-lower-to-llvm="generic-alloc local-target=hexagon-hvx128" \
// RUN: | mlir-translate --mlir-to-llvmir \
// RUN: | llc -O2 -mtriple=hexagon-unknown-linux-musl -mcpu=hexagonv68 \
// RUN:     -hexagon-small-data-threshold=0 \
// RUN: | FileCheck %s
//
// PROMOTED-LABEL: func.func @matmul_local
// PROMOTED:       memref.alloc() {alignment = 128 : i64} : memref<2x128x128xf32, #dsp.local>
// PROMOTED:       memref.alloc() {alignment = 128 : i64} : memref<2x128x128xf32, #dsp.local>
// PROMOTED:       scf.if
// PROMOTED-COUNT-2: memref.dma_start
//
// CHECK-LABEL: matmul_local:
// CHECK:       sfmpy
// CHECK:       sfadd
// CHECK-NOT:   .sf = vmpy
module {
  func.func @matmul_local(%a: tensor<128x256xf32>, %b: tensor<256x128xf32>) -> tensor<128x128xf32>
      attributes {llvm.emit_c_interface} {
    %r = dsp.matmul %a, %b : (tensor<128x256xf32>, tensor<256x128xf32>) -> tensor<128x128xf32>
    return %r : tensor<128x128xf32>
  }
}
