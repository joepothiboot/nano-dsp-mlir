// RUN: nanodsp-opt %s -convert-dsp-to-linalg -split-input-file | FileCheck %s

// CHECK: #[[ID2:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @add_2d
//  CHECK-SAME:   %[[A:.*]]: tensor<2x3xf32>, %[[B:.*]]: tensor<2x3xf32>
//       CHECK:   %[[E:.*]] = tensor.empty() : tensor<2x3xf32>
//       CHECK:   %[[G:.*]] = linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[ID2]], #[[ID2]], #[[ID2]]]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel"]
//  CHECK-SAME:     ins(%[[A]], %[[B]] : tensor<2x3xf32>, tensor<2x3xf32>)
//  CHECK-SAME:     outs(%[[E]] : tensor<2x3xf32>)
//       CHECK:   ^bb0(%[[IN0:.*]]: f32, %[[IN1:.*]]: f32, %{{.*}}: f32):
//       CHECK:     %[[S:.*]] = arith.addf %[[IN0]], %[[IN1]] : f32
//       CHECK:     linalg.yield %[[S]] : f32
//   CHECK-NOT:   dsp.add
//       CHECK:   return %[[G]]
func.func @add_2d(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = dsp.add %a, %b : tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// -----

// Rank generality: identity maps must follow the operand rank, and the
// destination must NOT be zero-filled (every element is written).
// CHECK: #[[ID4:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func.func @add_4d
//       CHECK:   tensor.empty() : tensor<1x4x4x2xf32>
//   CHECK-NOT:   linalg.fill
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[ID4]], #[[ID4]], #[[ID4]]]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel"]
func.func @add_4d(%a: tensor<1x4x4x2xf32>, %b: tensor<1x4x4x2xf32>) -> tensor<1x4x4x2xf32> {
  %0 = dsp.add %a, %b : tensor<1x4x4x2xf32>
  return %0 : tensor<1x4x4x2xf32>
}