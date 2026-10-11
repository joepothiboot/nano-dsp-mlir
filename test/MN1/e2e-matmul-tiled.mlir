// REQUIRES: mnpu-sim
// RUN: nanodsp-opt %s -convert-dsp-to-linalg -nanodsp-optimize=target=mn1 \
// RUN:     -nanodsp-bufferize -nanodsp-promote-local=target=mn1 \
// RUN: | nanodsp-translate --mlir-to-mn1 > %t.s
// RUN: rm -rf %t.d && mkdir -p %t.d && nanodsp-golden matmul 128 128 128 %t.d
// RUN: %mnpu-sim %t.s --load 0 %t.d/a.f32 --load 0x40000 %t.d/b.f32 \
// RUN:     --expect-bits 0x80000 %t.d/c_ref.f32 | FileCheck %s

// CHECK: OK (16384 values, bit-exact)
func.func @f(%a: tensor<128x128xf32>, %b: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %c = dsp.matmul %a, %b : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
  return %c : tensor<128x128xf32>
}
