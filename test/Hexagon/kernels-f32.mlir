// RUN: nanodsp-opt %s -convert-dsp-to-linalg -nanodsp-optimize=target=hexagon-hvx128 \
// RUN:     -nanodsp-lower-to-llvm=generic-alloc \
// RUN: | mlir-translate --mlir-to-llvmir \
// RUN: | llc -O2 -mtriple=hexagon-unknown-linux-musl -mcpu=hexagonv68 \
// RUN:     -mattr=+hvxv68,+hvx-length128b,+hvx-ieee-fp -hexagon-small-data-threshold=0 \
// RUN: | FileCheck %s --check-prefixes=CHECK,IEEE
// RUN: nanodsp-opt %s -convert-dsp-to-linalg -nanodsp-optimize=target=hexagon-hvx128 \
// RUN:     -nanodsp-lower-to-llvm=generic-alloc \
// RUN: | mlir-translate --mlir-to-llvmir \
// RUN: | llc -O2 -mtriple=hexagon-unknown-linux-musl -mcpu=hexagonv68 \
// RUN:     -mattr=+hvxv68,+hvx-length128b -hexagon-small-data-threshold=0 \
// RUN: | FileCheck %s --check-prefixes=CHECK,QF32
// CHECK-LABEL: matmul:
// IEEE:        v{{[0-9]+}}.sf = vmpy(v{{[0-9]+}}.sf,v{{[0-9]+}}.sf)
// IEEE:        v{{[0-9]+}}.sf = vadd(v{{[0-9]+}}.sf,v{{[0-9]+}}.sf)
// IEEE-NOT:    qf32
// QF32:        v{{[0-9:]+}}.qf32 = vmpy(v{{[0-9]+}}.sf,v{{[0-9]+}}.sf)
// QF32:        .qf32 = vadd(v{{[0-9]+}}.qf32,v{{[0-9]+}}.sf)
module {
  func.func @matmul(%a: tensor<64x96xf32>, %b: tensor<96x48xf32>) -> tensor<64x48xf32>
      attributes {llvm.emit_c_interface} {
    %r = dsp.matmul %a, %b : (tensor<64x96xf32>, tensor<96x48xf32>) -> tensor<64x48xf32>
    return %r : tensor<64x48xf32>
  }
  func.func @conv2d(%i: tensor<1x12x12x3xf32>, %f: tensor<3x3x3x8xf32>) -> tensor<1x10x10x8xf32>
      attributes {llvm.emit_c_interface} {
    %r = dsp.conv2d %i, %f : (tensor<1x12x12x3xf32>, tensor<3x3x3x8xf32>) -> tensor<1x10x10x8xf32>
    return %r : tensor<1x10x10x8xf32>
  }
  func.func @add_relu(%a: tensor<6x40xf32>, %b: tensor<6x40xf32>) -> tensor<6x40xf32>
      attributes {llvm.emit_c_interface} {
    %s = dsp.add %a, %b : tensor<6x40xf32>
    %r = dsp.relu %s : tensor<6x40xf32>
    return %r : tensor<6x40xf32>
  }
}
