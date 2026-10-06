module {
  func.func @matmul_64(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>) -> tensor<64x64xf32>
      attributes {llvm.emit_c_interface} {
    %r = dsp.matmul %a, %b : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %r : tensor<64x64xf32>
  }
  func.func @matmul_128(%a: tensor<128x128xf32>, %b: tensor<128x128xf32>) -> tensor<128x128xf32>
      attributes {llvm.emit_c_interface} {
    %r = dsp.matmul %a, %b : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    return %r : tensor<128x128xf32>
  }
  func.func @matmul_256(%a: tensor<256x256xf32>, %b: tensor<256x256xf32>) -> tensor<256x256xf32>
      attributes {llvm.emit_c_interface} {
    %r = dsp.matmul %a, %b : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    return %r : tensor<256x256xf32>
  }
  func.func @matmul_512(%a: tensor<512x512xf32>, %b: tensor<512x512xf32>) -> tensor<512x512xf32>
      attributes {llvm.emit_c_interface} {
    %r = dsp.matmul %a, %b : (tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
    return %r : tensor<512x512xf32>
  }
  func.func @conv2d_56_64_64(%i: tensor<1x56x56x64xf32>, %f: tensor<3x3x64x64xf32>) -> tensor<1x54x54x64xf32>
      attributes {llvm.emit_c_interface} {
    %r = dsp.conv2d %i, %f : (tensor<1x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<1x54x54x64xf32>
    return %r : tensor<1x54x54x64xf32>
  }
  func.func @conv2d_28_128_128(%i: tensor<1x28x28x128xf32>, %f: tensor<3x3x128x128xf32>) -> tensor<1x26x26x128xf32>
      attributes {llvm.emit_c_interface} {
    %r = dsp.conv2d %i, %f : (tensor<1x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<1x26x26x128xf32>
    return %r : tensor<1x26x26x128xf32>
  }
  func.func @qmatmul_256(%a: tensor<256x256xi8>, %b: tensor<256x256xi8>) -> tensor<256x256xi8>
      attributes {llvm.emit_c_interface} {
    %r = dsp.qmatmul %a, %b {lhs_zp = -7 : i32, rhs_zp = 12 : i32,
                             multiplier = 1276901417 : i32, shift = 9 : i32,
                             out_zp = 4 : i32}
       : (tensor<256x256xi8>, tensor<256x256xi8>) -> tensor<256x256xi8>
    return %r : tensor<256x256xi8>
  }
}
