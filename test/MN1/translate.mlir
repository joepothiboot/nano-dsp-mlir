// RUN: nanodsp-opt %s -convert-dsp-to-linalg -nanodsp-optimize=target=mn1 \
// RUN:     -nanodsp-bufferize -nanodsp-promote-local=target=mn1 \
// RUN: | nanodsp-translate --mlir-to-mn1 | FileCheck %s

// CHECK:       ; arg0 at 0x0 (4096 bytes)
// CHECK-NEXT:  ; arg1 at 0x40000 (4096 bytes)
// CHECK:       ; buffer at 0x80000 (4096 bytes)
// CHECK:       vzero
// CHECK:       dma.st 7,
// CHECK-NEXT:  dma.wait 7
// CHECK:       dma.ld 0,
// CHECK-NEXT:  dma.wait 0
// CHECK:       vsplat
// CHECK:       vmul [[T:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
// CHECK-NEXT:  vadd v{{[0-9]+}}, v{{[0-9]+}}, [[T]]
// CHECK-NOT:   vfma
// CHECK:       dma.st
// CHECK:       ; result at 0x80000
// CHECK-NEXT:  halt
func.func @mm(%a: tensor<32x32xf32>, %b: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %c = dsp.matmul %a, %b : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %c : tensor<32x32xf32>
}
