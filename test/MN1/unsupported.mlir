// RUN: nanodsp-opt %s -convert-dsp-to-linalg -nanodsp-optimize=target=mn1 \
// RUN:     -nanodsp-bufferize -nanodsp-promote-local=target=mn1 \
// RUN: | not nanodsp-translate --mlir-to-mn1 2>&1 | FileCheck %s

// CHECK: error: the MN1 backend does not support this op
// CHECK: arith.maximumf
func.func @relu(%a: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %c = dsp.relu %a : tensor<64x64xf32>
  return %c : tensor<64x64xf32>
}
