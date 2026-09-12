// RUN: nanodsp-opt %s -convert-dsp-to-linalg -split-input-file | FileCheck %s

// Unit stride / unit dilation: the input map must degenerate to plain adds,
// matching linalg.conv_2d_nhwc_hwcf's own indexing maps.
// CHECK-DAG: #[[IN:.*]]  = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
// CHECK-DAG: #[[FLT:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
// CHECK-DAG: #[[OUT:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @conv2d_unit
//  CHECK-SAME:   %[[I:.*]]: tensor<1x8x8x3xf32>, %[[F:.*]]: tensor<3x3x3x4xf32>
//       CHECK:   %[[E:.*]] = tensor.empty() : tensor<1x6x6x4xf32>
//       CHECK:   %[[Z:.*]] = arith.constant 0.000000e+00 : f32
//       CHECK:   %[[FI:.*]] = linalg.fill ins(%[[Z]] : f32) outs(%[[E]] : tensor<1x6x6x4xf32>)
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[IN]], #[[FLT]], #[[OUT]]]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
//  CHECK-SAME:     ins(%[[I]], %[[F]] : tensor<1x8x8x3xf32>, tensor<3x3x3x4xf32>)
//  CHECK-SAME:     outs(%[[FI]] : tensor<1x6x6x4xf32>)
//       CHECK:   ^bb0(%[[X:.*]]: f32, %[[W:.*]]: f32, %[[ACC:.*]]: f32):
//       CHECK:     %[[P:.*]] = arith.mulf %[[X]], %[[W]] : f32
//       CHECK:     %[[S:.*]] = arith.addf %[[ACC]], %[[P]] : f32
//       CHECK:     linalg.yield %[[S]] : f32
func.func @conv2d_unit(%in: tensor<1x8x8x3xf32>, %f: tensor<3x3x3x4xf32>) -> tensor<1x6x6x4xf32> {
  %0 = dsp.conv2d %in, %f : (tensor<1x8x8x3xf32>, tensor<3x3x3x4xf32>) -> tensor<1x6x6x4xf32>
  return %0 : tensor<1x6x6x4xf32>
}

// -----

// Stride 2, dilation 3 must be folded into the affine expression, not emitted
// as arithmetic in the body.
// CHECK-DAG: #[[IN:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4 * 3, d2 * 2 + d5 * 3, d6)>

// CHECK-LABEL: func.func @conv2d_strided_dilated
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[IN]],
//       CHECK:   ^bb0(
//   CHECK-NOT:     arith.muli
//   CHECK-NOT:     arith.addi
//       CHECK:     linalg.yield
func.func @conv2d_strided_dilated(%in: tensor<1x13x13x1xf32>, %f: tensor<3x3x1x1xf32>) -> tensor<1x4x4x1xf32> {
  // OH = (13 - (3-1)*3 - 1)/2 + 1 = 4
  %0 = dsp.conv2d %in, %f {strides = array<i64: 2, 2>, dilations = array<i64: 3, 3>}
     : (tensor<1x13x13x1xf32>, tensor<3x3x1x1xf32>) -> tensor<1x4x4x1xf32>
  return %0 : tensor<1x4x4x1xf32>
}

// -----

// Full conversion: nothing from 'dsp' may survive a composed pipeline.
// CHECK-LABEL: func.func @pipeline
//   CHECK-NOT:   dsp.
//       CHECK:   linalg.generic
//       CHECK:   linalg.generic
//       CHECK:   linalg.generic
//   CHECK-NOT:   dsp.
func.func @pipeline(%img: tensor<1x8x8x1xf32>, %k: tensor<3x3x1x1xf32>,
                    %bias: tensor<1x6x6x1xf32>) -> tensor<1x6x6x1xf32> {
  %0 = dsp.conv2d %img, %k : (tensor<1x8x8x1xf32>, tensor<3x3x1x1xf32>) -> tensor<1x6x6x1xf32>
  %1 = dsp.add %0, %bias : tensor<1x6x6x1xf32>
  %2 = dsp.relu %1 : tensor<1x6x6x1xf32>
  return %2 : tensor<1x6x6x1xf32>
}