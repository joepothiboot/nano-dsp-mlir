// RUN: nanodsp-opt %s -convert-dsp-to-linalg -nanodsp-optimize=target=hexagon-hvx128 \
// RUN:     -nanodsp-bufferize -nanodsp-promote-local=target=hexagon-hvx128 \
// RUN: | FileCheck %s --check-prefix=PROMOTED
// RUN: nanodsp-opt %s -convert-dsp-to-linalg -nanodsp-optimize=target=hexagon-hvx128 \
// RUN:     -nanodsp-lower-to-llvm="generic-alloc local-target=hexagon-hvx128" \
// RUN: | mlir-translate --mlir-to-llvmir \
// RUN: | llc -O2 -mtriple=hexagon-unknown-linux-musl -mcpu=hexagonv68 \
// RUN:     -mattr=+hvxv68,+hvx-length128b -hexagon-small-data-threshold=0 \
// RUN: | FileCheck %s
// PROMOTED-LABEL: func.func @qmatmul_local
// PROMOTED:       memref.alloc() {alignment = 128 : i64} : memref<2x128x512xi8, #dsp.local>
// PROMOTED:       memref.alloc() {alignment = 128 : i64} : memref<2x512x128xi8, #dsp.local>
// PROMOTED:       scf.if
// PROMOTED-COUNT-2: memref.dma_start
// CHECK-LABEL: qmatmul_local:
// CHECK:       .h = vunpack(v{{[0-9]+}}.b)
module {
  func.func @qmatmul_local(%a: tensor<128x1024xi8>, %b: tensor<1024x128xi8>) -> tensor<128x128xi8>
      attributes {llvm.emit_c_interface} {
    %r = dsp.qmatmul %a, %b {lhs_zp = -7 : i32, rhs_zp = 12 : i32,
                             multiplier = 1276901417 : i32, shift = 9 : i32,
                             out_zp = 4 : i32}
       : (tensor<128x1024xi8>, tensor<1024x128xi8>) -> tensor<128x128xi8>
    return %r : tensor<128x128xi8>
  }
}
