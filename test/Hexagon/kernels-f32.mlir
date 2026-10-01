// f32 kernels for the on-target test (scripts/run-hexagon.sh): compiled on
// the host to a Hexagon object, linked with harness.cpp, run under
// qemu-hexagon. The harness computes the same results with
// reference/nanodsp_ref.h on the emulated core and compares bit for bit.
//
// Hexagon is a 32-bit target, but index stays 64-bit: not every MLIR->LLVM
// conversion honors a narrower index width. So descriptors hold 64-bit
// offsets/sizes/strides, and allocation goes through
// _mlir_memref_to_llvm_alloc(uint64_t), defined in harness.cpp
// (-nanodsp-lower-to-llvm=generic-alloc), not through the 32-bit malloc.
//
// The RUN lines only check codegen (no emulator needed).
//
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
//
// With +hvx-ieee-fp the f32 math is IEEE single precision on HVX. Without
// it, llc multiplies into QFloat (qf32, HVX PRM sec. 5.6) and accumulates in
// qf32, which is not IEEE-754 (no implied significand bit, Von Neumann
// rounding, no Inf/NaN).
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
