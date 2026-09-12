// RUN: nanodsp-opt %s -canonicalize -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @relu_idempotent
//       CHECK:   %[[R:.*]] = dsp.relu %arg0 : tensor<4xf32>
//   CHECK-NOT:   dsp.relu %[[R]]
//       CHECK:   return %[[R]]
func.func @relu_idempotent(%a: tensor<4xf32>) -> tensor<4xf32> {
  %0 = dsp.relu %a : tensor<4xf32>
  %1 = dsp.relu %0 : tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// Three deep collapses to one.
// CHECK-LABEL: func.func @relu_triple
//       CHECK:   dsp.relu
//   CHECK-NOT:   dsp.relu
func.func @relu_triple(%a: tensor<4xf32>) -> tensor<4xf32> {
  %0 = dsp.relu %a : tensor<4xf32>
  %1 = dsp.relu %0 : tensor<4xf32>
  %2 = dsp.relu %1 : tensor<4xf32>
  return %2 : tensor<4xf32>
}

// -----

// An intervening add must block the fold.
// CHECK-LABEL: func.func @relu_not_adjacent
//       CHECK:   dsp.relu
//       CHECK:   dsp.add
//       CHECK:   dsp.relu
func.func @relu_not_adjacent(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %0 = dsp.relu %a : tensor<4xf32>
  %1 = dsp.add %0, %b : tensor<4xf32>
  %2 = dsp.relu %1 : tensor<4xf32>
  return %2 : tensor<4xf32>
}