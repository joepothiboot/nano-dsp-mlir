// RUN: nanodsp-opt %s | nanodsp-opt | FileCheck %s

// CHECK-LABEL: func.func @add_2d
func.func @add_2d(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: dsp.add %{{.*}}, %{{.*}} : tensor<2x3xf32>
  %0 = dsp.add %a, %b : tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @add_4d
func.func @add_4d(%a: tensor<1x4x4x2xf32>, %b: tensor<1x4x4x2xf32>) -> tensor<1x4x4x2xf32> {
  // CHECK: dsp.add %{{.*}}, %{{.*}} : tensor<1x4x4x2xf32>
  %0 = dsp.add %a, %b : tensor<1x4x4x2xf32>
  return %0 : tensor<1x4x4x2xf32>
}

// CHECK-LABEL: func.func @relu_1d
func.func @relu_1d(%a: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK: dsp.relu %{{.*}} : tensor<8xf32>
  %0 = dsp.relu %a : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func.func @matmul
func.func @matmul(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>) -> tensor<4x16xf32> {
  // CHECK: dsp.matmul %{{.*}}, %{{.*}} : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  %0 = dsp.matmul %a, %b : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func.func @conv2d_default
func.func @conv2d_default(%in: tensor<1x8x8x3xf32>, %f: tensor<3x3x3x4xf32>) -> tensor<1x6x6x4xf32> {
  // CHECK: dsp.conv2d %{{.*}}, %{{.*}} :
  %0 = dsp.conv2d %in, %f : (tensor<1x8x8x3xf32>, tensor<3x3x3x4xf32>) -> tensor<1x6x6x4xf32>
  return %0 : tensor<1x6x6x4xf32>
}

// CHECK-LABEL: func.func @conv2d_strided_dilated
func.func @conv2d_strided_dilated(%in: tensor<1x9x9x1xf32>, %f: tensor<3x3x1x1xf32>) -> tensor<1x3x3x1xf32> {
  // OH = (9 - (3-1)*2 - 1)/2 + 1 = 3
  // CHECK: dsp.conv2d %{{.*}}, %{{.*}} {dilations = array<i64: 2, 2>, strides = array<i64: 2, 2>}
  %0 = dsp.conv2d %in, %f {strides = array<i64: 2, 2>, dilations = array<i64: 2, 2>}
     : (tensor<1x9x9x1xf32>, tensor<3x3x1x1xf32>) -> tensor<1x3x3x1xf32>
  return %0 : tensor<1x3x3x1xf32>
}

// CHECK-LABEL: func.func @pipeline
func.func @pipeline(%img: tensor<1x8x8x1xf32>, %k: tensor<3x3x1x1xf32>,
                    %bias: tensor<1x6x6x1xf32>) -> tensor<1x6x6x1xf32> {
  // CHECK: dsp.conv2d
  // CHECK: dsp.add
  // CHECK: dsp.relu
  %0 = dsp.conv2d %img, %k : (tensor<1x8x8x1xf32>, tensor<3x3x1x1xf32>) -> tensor<1x6x6x1xf32>
  %1 = dsp.add %0, %bias : tensor<1x6x6x1xf32>
  %2 = dsp.relu %1 : tensor<1x6x6x1xf32>
  return %2 : tensor<1x6x6x1xf32>
}