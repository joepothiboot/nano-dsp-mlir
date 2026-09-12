// RUN: nanodsp-opt %s -convert-dsp-to-linalg | FileCheck %s

// CHECK: #[[ID1:.*]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func.func @relu_1d
//  CHECK-SAME:   %[[A:.*]]: tensor<8xf32>
//       CHECK:   %[[E:.*]] = tensor.empty() : tensor<8xf32>
//       CHECK:   %[[Z:.*]] = arith.constant 0.000000e+00 : f32
//       CHECK:   %[[G:.*]] = linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[ID1]], #[[ID1]]]
//  CHECK-SAME:     iterator_types = ["parallel"]
//  CHECK-SAME:     ins(%[[A]] : tensor<8xf32>)
//  CHECK-SAME:     outs(%[[E]] : tensor<8xf32>)
//       CHECK:   ^bb0(%[[IN:.*]]: f32, %{{.*}}: f32):
// NaN-propagating max is required; maxnumf would be wrong.
//       CHECK:     %[[R:.*]] = arith.maximumf %[[IN]], %[[Z]] : f32
//   CHECK-NOT:     arith.maxnumf
//       CHECK:     linalg.yield %[[R]] : f32
//       CHECK:   return %[[G]]
func.func @relu_1d(%a: tensor<8xf32>) -> tensor<8xf32> {
  %0 = dsp.relu %a : tensor<8xf32>
  return %0 : tensor<8xf32>
}