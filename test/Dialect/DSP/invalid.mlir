// RUN: nanodsp-opt %s -split-input-file -verify-diagnostics

func.func @add_shape_mismatch(%a: tensor<2x3xf32>, %b: tensor<3x2xf32>) -> tensor<2x3xf32> {
  // expected-error @+1 {{requires the same type for all operands and results}}
  %0 = "dsp.add"(%a, %b) : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// -----

func.func @add_dynamic_rejected(%a: tensor<?x3xf32>, %b: tensor<?x3xf32>) -> tensor<?x3xf32> {
  // expected-error @+1 {{op operand #0 must be statically shaped tensor of f32}}
  %0 = "dsp.add"(%a, %b) : (tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<?x3xf32>
  return %0 : tensor<?x3xf32>
}

// -----

func.func @add_wrong_elem_type(%a: tensor<2x3xi32>, %b: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // expected-error @+1 {{op operand #0 must be statically shaped tensor of f32}}
  %0 = "dsp.add"(%a, %b) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}

// -----

func.func @matmul_k_mismatch(%a: tensor<4x8xf32>, %b: tensor<7x16xf32>) -> tensor<4x16xf32> {
  // expected-error @+1 {{contraction dimension mismatch: lhs has K=8 but rhs has K=7}}
  %0 = dsp.matmul %a, %b : (tensor<4x8xf32>, tensor<7x16xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// -----

func.func @matmul_bad_result(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>) -> tensor<4x15xf32> {
  // expected-error @+1 {{result shape must be 4x16, got 4x15}}
  %0 = dsp.matmul %a, %b : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x15xf32>
  return %0 : tensor<4x15xf32>
}

// -----

func.func @matmul_wrong_rank(%a: tensor<4x8x2xf32>, %b: tensor<8x16xf32>) -> tensor<4x16xf32> {
  // expected-error @+1 {{op operand #0 must be statically shaped rank-2 tensor of f32}}
  %0 = "dsp.matmul"(%a, %b) : (tensor<4x8x2xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// -----

func.func @conv2d_channel_mismatch(%in: tensor<1x8x8x3xf32>, %f: tensor<3x3x5x4xf32>) -> tensor<1x6x6x4xf32> {
  // expected-error @+1 {{channel mismatch: input has C=3 but filter has C=5}}
  %0 = dsp.conv2d %in, %f : (tensor<1x8x8x3xf32>, tensor<3x3x5x4xf32>) -> tensor<1x6x6x4xf32>
  return %0 : tensor<1x6x6x4xf32>
}

// -----

func.func @conv2d_bad_output_extent(%in: tensor<1x8x8x1xf32>, %f: tensor<3x3x1x1xf32>) -> tensor<1x7x6x1xf32> {
  // expected-error @+1 {{result shape must be 1x6x6x1}}
  %0 = dsp.conv2d %in, %f : (tensor<1x8x8x1xf32>, tensor<3x3x1x1xf32>) -> tensor<1x7x6x1xf32>
  return %0 : tensor<1x7x6x1xf32>
}

// -----

func.func @conv2d_filter_too_big(%in: tensor<1x2x2x1xf32>, %f: tensor<3x3x1x1xf32>) -> tensor<1x1x1x1xf32> {
  // expected-error @+1 {{dilated filter 3x3 does not fit in input spatial extent 2x2}}
  %0 = dsp.conv2d %in, %f : (tensor<1x2x2x1xf32>, tensor<3x3x1x1xf32>) -> tensor<1x1x1x1xf32>
  return %0 : tensor<1x1x1x1xf32>
}

// -----

func.func @conv2d_bad_stride_count(%in: tensor<1x8x8x1xf32>, %f: tensor<3x3x1x1xf32>) -> tensor<1x6x6x1xf32> {
  // expected-error @+1 {{expected 2 strides, got 3}}
  %0 = dsp.conv2d %in, %f {strides = array<i64: 1, 1, 1>, dilations = array<i64: 1, 1>}
     : (tensor<1x8x8x1xf32>, tensor<3x3x1x1xf32>) -> tensor<1x6x6x1xf32>
  return %0 : tensor<1x6x6x1xf32>
}

// -----

func.func @conv2d_zero_stride(%in: tensor<1x8x8x1xf32>, %f: tensor<3x3x1x1xf32>) -> tensor<1x6x6x1xf32> {
  // expected-error @+1 {{strides must be >= 1, got 0}}
  %0 = dsp.conv2d %in, %f {strides = array<i64: 0, 1>, dilations = array<i64: 1, 1>}
     : (tensor<1x8x8x1xf32>, tensor<3x3x1x1xf32>) -> tensor<1x6x6x1xf32>
  return %0 : tensor<1x6x6x1xf32>
}