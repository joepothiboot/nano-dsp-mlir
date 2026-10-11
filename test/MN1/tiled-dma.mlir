// RUN: nanodsp-opt %s -convert-dsp-to-linalg -nanodsp-optimize=target=mn1 \
// RUN:     -nanodsp-bufferize -nanodsp-promote-local=target=mn1 \
// RUN: | nanodsp-translate --mlir-to-mn1 | FileCheck %s

// CHECK:       dma_rows{{[0-9]+}}:
// CHECK:       dma.ld [[TAG:[0-9]]], r14, r15, r13
// CHECK:       dma.wait [[TAG]]
// CHECK:       addi r14, r14, 64
// CHECK-NEXT:  addi r15, r15, 512
// CHECK:       bne r{{[0-9]+}}, r0, slot1_
// CHECK:       dma.wait
// CHECK:       vsplat
// CHECK:       vmul
// CHECK:       mul r{{[0-9]+}}, r{{[0-9]+}}, r13
// CHECK:       halt
func.func @mm(%a: tensor<128x128xf32>, %b: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %c = dsp.matmul %a, %b : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
  return %c : tensor<128x128xf32>
}
