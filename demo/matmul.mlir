func.func @matmul(%a: tensor<128x256xf32>, %b: tensor<256x96xf32>) -> tensor<128x96xf32> {
  %r = dsp.matmul %a, %b : (tensor<128x256xf32>, tensor<256x96xf32>) -> tensor<128x96xf32>
  return %r : tensor<128x96xf32>
}
