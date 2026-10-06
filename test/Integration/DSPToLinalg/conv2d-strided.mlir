// RUN: nanodsp-opt %s -convert-dsp-to-linalg \
// RUN: | mlir-opt %stock_lower_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_runner_utils --shared-libs=%mlir_c_runner_utils \
// RUN: | FileCheck %s

func.func private @printMemrefF32(%ptr : tensor<*xf32>)

func.func @main() {
  %in = arith.constant dense<[[[[ 1.0],[ 2.0],[ 3.0],[ 4.0],[ 5.0]],
                               [[ 6.0],[ 7.0],[ 8.0],[ 9.0],[10.0]],
                               [[11.0],[12.0],[13.0],[14.0],[15.0]],
                               [[16.0],[17.0],[18.0],[19.0],[20.0]],
                               [[21.0],[22.0],[23.0],[24.0],[25.0]]]]> : tensor<1x5x5x1xf32>
  %f = arith.constant dense<1.0> : tensor<3x3x1x1xf32>

  %r = dsp.conv2d %in, %f {strides = array<i64: 2, 2>, dilations = array<i64: 1, 1>}
     : (tensor<1x5x5x1xf32>, tensor<3x3x1x1xf32>) -> tensor<1x2x2x1xf32>

  %u = tensor.cast %r : tensor<1x2x2x1xf32> to tensor<*xf32>
  call @printMemrefF32(%u) : (tensor<*xf32>) -> ()
  return
}

// CHECK: rank = 4 offset = 0 sizes = [1, 2, 2, 1]
// CHECK: 63
// CHECK: 81
// CHECK: 153
// CHECK: 171