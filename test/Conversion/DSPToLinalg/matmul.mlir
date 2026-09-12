// RUN: nanodsp-opt %s -convert-dsp-to-linalg | FileCheck %s

// CHECK-DAG: #[[LHS:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[RHS:.*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[OUT:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func.func @matmul
//  CHECK-SAME:   %[[A:.*]]: tensor<4x8xf32>, %[[B:.*]]: tensor<8x16xf32>
//       CHECK:   %[[E:.*]] = tensor.empty() : tensor<4x16xf32>
//       CHECK:   %[[Z:.*]] = arith.constant 0.000000e+00 : f32
// A reduction destination MUST be zero-initialized -- this fill is load-bearing.
//       CHECK:   %[[F:.*]] = linalg.fill ins(%[[Z]] : f32) outs(%[[E]] : tensor<4x16xf32>)
//       CHECK:   %[[G:.*]] = linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[LHS]], #[[RHS]], #[[OUT]]]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "reduction"]
//  CHECK-SAME:     ins(%[[A]], %[[B]] : tensor<4x8xf32>, tensor<8x16xf32>)
//  CHECK-SAME:     outs(%[[F]] : tensor<4x16xf32>)
//       CHECK:   ^bb0(%[[X:.*]]: f32, %[[Y:.*]]: f32, %[[ACC:.*]]: f32):
//       CHECK:     %[[P:.*]] = arith.mulf %[[X]], %[[Y]] : f32
//       CHECK:     %[[S:.*]] = arith.addf %[[ACC]], %[[P]] : f32
//       CHECK:     linalg.yield %[[S]] : f32
func.func @matmul(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>) -> tensor<4x16xf32> {
  %0 = dsp.matmul %a, %b : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}