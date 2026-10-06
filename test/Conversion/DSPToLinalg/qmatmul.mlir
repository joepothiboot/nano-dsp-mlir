// RUN: nanodsp-opt %s -convert-dsp-to-linalg | FileCheck %s

// CHECK-DAG: #[[LHS:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[RHS:.*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[OUT:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-DAG: #[[ID:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @qmatmul
//  CHECK-SAME:   %[[A:.*]]: tensor<2x3xi8>, %[[B:.*]]: tensor<3x4xi8>
//   CHECK-DAG:   %[[LZP:.*]] = arith.constant 3 : i32
//   CHECK-DAG:   %[[RZP:.*]] = arith.constant -2 : i32
//       CHECK:   %[[E:.*]] = tensor.empty() : tensor<2x4xi32>
//       CHECK:   %[[F:.*]] = linalg.fill ins(%{{.*}} : i32) outs(%[[E]] : tensor<2x4xi32>)
//       CHECK:   %[[ACC:.*]] = linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[LHS]], #[[RHS]], #[[OUT]]]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "reduction"]
//  CHECK-SAME:     ins(%[[A]], %[[B]] : tensor<2x3xi8>, tensor<3x4xi8>)
//  CHECK-SAME:     outs(%[[F]] : tensor<2x4xi32>)
//       CHECK:   ^bb0(%[[X:.*]]: i8, %[[Y:.*]]: i8, %[[S:.*]]: i32):
//       CHECK:     %[[XW:.*]] = arith.extsi %[[X]] : i8 to i32
//       CHECK:     %[[XS:.*]] = arith.subi %[[XW]], %[[LZP]] : i32
//       CHECK:     %[[YW:.*]] = arith.extsi %[[Y]] : i8 to i32
//       CHECK:     %[[YS:.*]] = arith.subi %[[YW]], %[[RZP]] : i32
//       CHECK:     %[[P:.*]] = arith.muli %[[XS]], %[[YS]] : i32
//       CHECK:     %[[SUM:.*]] = arith.addi %[[S]], %[[P]] : i32
//       CHECK:     linalg.yield %[[SUM]] : i32

//   CHECK-DAG:   %[[MUL:.*]] = arith.constant 1073741824 : i64
//   CHECK-DAG:   %[[RND:.*]] = arith.constant 8589934592 : i64
//   CHECK-DAG:   %[[SH:.*]] = arith.constant 34 : i64
//   CHECK-DAG:   %[[OZP:.*]] = arith.constant -5 : i64
//   CHECK-DAG:   %[[LO:.*]] = arith.constant -128 : i64
//   CHECK-DAG:   %[[HI:.*]] = arith.constant 127 : i64
//       CHECK:   %[[E8:.*]] = tensor.empty() : tensor<2x4xi8>
//       CHECK:   %[[R:.*]] = linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[ID]], #[[ID]]]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel"]
//  CHECK-SAME:     ins(%[[ACC]] : tensor<2x4xi32>) outs(%[[E8]] : tensor<2x4xi8>)
//       CHECK:     %[[W:.*]] = arith.extsi %{{.*}} : i32 to i64
//       CHECK:     %[[M:.*]] = arith.muli %[[W]], %[[MUL]] : i64
//       CHECK:     %[[RD:.*]] = arith.addi %[[M]], %[[RND]] : i64
//       CHECK:     %[[SR:.*]] = arith.shrsi %[[RD]], %[[SH]] : i64
//       CHECK:     %[[Z:.*]] = arith.addi %[[SR]], %[[OZP]] : i64
//       CHECK:     %[[C1:.*]] = arith.maxsi %[[Z]], %[[LO]] : i64
//       CHECK:     %[[C2:.*]] = arith.minsi %[[C1]], %[[HI]] : i64
//       CHECK:     %[[T:.*]] = arith.trunci %[[C2]] : i64 to i8
//       CHECK:     linalg.yield %[[T]] : i8
//       CHECK:   return %[[R]] : tensor<2x4xi8>
func.func @qmatmul(%a: tensor<2x3xi8>, %b: tensor<3x4xi8>) -> tensor<2x4xi8> {
  %0 = dsp.qmatmul %a, %b {lhs_zp = 3 : i32, rhs_zp = -2 : i32,
                           multiplier = 1073741824 : i32, shift = 3 : i32,
                           out_zp = -5 : i32}
     : (tensor<2x3xi8>, tensor<3x4xi8>) -> tensor<2x4xi8>
  return %0 : tensor<2x4xi8>
}
