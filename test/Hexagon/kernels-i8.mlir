// int8 kernels for the on-target test (scripts/run-hexagon.sh); see
// kernels-f32.mlir.
//
// Hexagon is a 32-bit target, but index stays 64-bit: not every MLIR->LLVM
// conversion honors a narrower index width. So descriptors hold 64-bit
// offsets/sizes/strides, and allocation goes through
// _mlir_memref_to_llvm_alloc(uint64_t), defined in harness.cpp
// (-nanodsp-lower-to-llvm=generic-alloc), not through the 32-bit malloc.
//
// RUN: nanodsp-opt %s -convert-dsp-to-linalg -nanodsp-optimize=target=hexagon-hvx128 \
// RUN:     -nanodsp-lower-to-llvm=generic-alloc \
// RUN: | mlir-translate --mlir-to-llvmir \
// RUN: | llc -O2 -mtriple=hexagon-unknown-linux-musl -mcpu=hexagonv68 \
// RUN:     -mattr=+hvxv68,+hvx-length128b -hexagon-small-data-threshold=0 \
// RUN: | FileCheck %s --check-prefixes=CHECK
//
// int8 lanes are widened b -> h -> w, then multiplied as 32-bit words. LLVM
// does not form the 4-wide int8 dot product vrmpy (HVX PRM sec. 6.9) from
// this IR; that needs a dedicated lowering.
// CHECK-LABEL: qmatmul_golden:
// CHECK:       .h = vunpack(v{{[0-9]+}}.b)
// CHECK:       .w = vunpack(v{{[0-9]+}}.h)
// CHECK-NOT:   vrmpy
module {
  func.func @qmatmul_golden(%a: tensor<2x3xi8>, %b: tensor<3x4xi8>) -> tensor<2x4xi8>
      attributes {llvm.emit_c_interface} {
    %r = dsp.qmatmul %a, %b {lhs_zp = 3 : i32, rhs_zp = -2 : i32,
                             multiplier = 1073741824 : i32, shift = 3 : i32,
                             out_zp = -5 : i32}
       : (tensor<2x3xi8>, tensor<3x4xi8>) -> tensor<2x4xi8>
    return %r : tensor<2x4xi8>
  }
  func.func @qmatmul(%a: tensor<32x128xi8>, %b: tensor<128x64xi8>) -> tensor<32x64xi8>
      attributes {llvm.emit_c_interface} {
    %r = dsp.qmatmul %a, %b {lhs_zp = -7 : i32, rhs_zp = 12 : i32,
                             multiplier = 1276901417 : i32, shift = 9 : i32,
                             out_zp = 4 : i32}
       : (tensor<32x128xi8>, tensor<128x64xi8>) -> tensor<32x64xi8>
    return %r : tensor<32x64xi8>
  }
}
