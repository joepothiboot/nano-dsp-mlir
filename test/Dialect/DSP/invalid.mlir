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
// -----

func.func @qmatmul_f32_rejected(%a: tensor<2x3xf32>, %b: tensor<3x4xf32>) -> tensor<2x4xf32> {
  // expected-error @+1 {{op operand #0 must be statically shaped rank-2 tensor of i8}}
  %0 = "dsp.qmatmul"(%a, %b) {lhs_zp = 0 : i32, rhs_zp = 0 : i32, multiplier = 1073741824 : i32, shift = 0 : i32, out_zp = 0 : i32}
     : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func @qmatmul_k_mismatch(%a: tensor<2x3xi8>, %b: tensor<4x4xi8>) -> tensor<2x4xi8> {
  // expected-error @+1 {{contraction dimension mismatch: lhs has K=3 but rhs has K=4}}
  %0 = dsp.qmatmul %a, %b {lhs_zp = 0 : i32, rhs_zp = 0 : i32, multiplier = 1073741824 : i32, shift = 0 : i32, out_zp = 0 : i32}
     : (tensor<2x3xi8>, tensor<4x4xi8>) -> tensor<2x4xi8>
  return %0 : tensor<2x4xi8>
}

// -----

func.func @qmatmul_bad_result(%a: tensor<2x3xi8>, %b: tensor<3x4xi8>) -> tensor<4x2xi8> {
  // expected-error @+1 {{result shape must be 2x4, got 4x2}}
  %0 = dsp.qmatmul %a, %b {lhs_zp = 0 : i32, rhs_zp = 0 : i32, multiplier = 1073741824 : i32, shift = 0 : i32, out_zp = 0 : i32}
     : (tensor<2x3xi8>, tensor<3x4xi8>) -> tensor<4x2xi8>
  return %0 : tensor<4x2xi8>
}

// -----

func.func @qmatmul_k_overflow(%a: tensor<1x33026xi8>, %b: tensor<33026x1xi8>) -> tensor<1x1xi8> {
  // expected-error @+1 {{K=33026 can overflow the i32 accumulator (at most 33025)}}
  %0 = dsp.qmatmul %a, %b {lhs_zp = 0 : i32, rhs_zp = 0 : i32, multiplier = 1073741824 : i32, shift = 0 : i32, out_zp = 0 : i32}
     : (tensor<1x33026xi8>, tensor<33026x1xi8>) -> tensor<1x1xi8>
  return %0 : tensor<1x1xi8>
}

// -----

func.func @qmatmul_zp_range(%a: tensor<2x3xi8>, %b: tensor<3x4xi8>) -> tensor<2x4xi8> {
  // expected-error @+1 {{rhs_zp must be in [-128, 127], got -129}}
  %0 = dsp.qmatmul %a, %b {lhs_zp = 0 : i32, rhs_zp = -129 : i32, multiplier = 1073741824 : i32, shift = 0 : i32, out_zp = 0 : i32}
     : (tensor<2x3xi8>, tensor<3x4xi8>) -> tensor<2x4xi8>
  return %0 : tensor<2x4xi8>
}

// -----

func.func @qmatmul_unnormalized_multiplier(%a: tensor<2x3xi8>, %b: tensor<3x4xi8>) -> tensor<2x4xi8> {
  // expected-error @+1 {{multiplier must be normalized to [2^30, 2^31), got 536870912}}
  %0 = dsp.qmatmul %a, %b {lhs_zp = 0 : i32, rhs_zp = 0 : i32, multiplier = 536870912 : i32, shift = 0 : i32, out_zp = 0 : i32}
     : (tensor<2x3xi8>, tensor<3x4xi8>) -> tensor<2x4xi8>
  return %0 : tensor<2x4xi8>
}

// -----

func.func @qmatmul_negative_multiplier(%a: tensor<2x3xi8>, %b: tensor<3x4xi8>) -> tensor<2x4xi8> {
  // expected-error @+1 {{multiplier must be normalized to [2^30, 2^31), got -1073741824}}
  %0 = dsp.qmatmul %a, %b {lhs_zp = 0 : i32, rhs_zp = 0 : i32, multiplier = -1073741824 : i32, shift = 0 : i32, out_zp = 0 : i32}
     : (tensor<2x3xi8>, tensor<3x4xi8>) -> tensor<2x4xi8>
  return %0 : tensor<2x4xi8>
}

// -----

func.func @qmatmul_negative_shift(%a: tensor<2x3xi8>, %b: tensor<3x4xi8>) -> tensor<2x4xi8> {
  // expected-error @+1 {{shift must be in [0, 31], got -1}}
  %0 = dsp.qmatmul %a, %b {lhs_zp = 0 : i32, rhs_zp = 0 : i32, multiplier = 1073741824 : i32, shift = -1 : i32, out_zp = 0 : i32}
     : (tensor<2x3xi8>, tensor<3x4xi8>) -> tensor<2x4xi8>
  return %0 : tensor<2x4xi8>
}
