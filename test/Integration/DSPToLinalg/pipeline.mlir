// RUN: nanodsp-opt %s -convert-dsp-to-linalg \
// RUN: | mlir-opt %stock_lower_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_runner_utils --shared-libs=%mlir_c_runner_utils \
// RUN: | FileCheck %s

func.func private @printMemrefF32(%ptr : tensor<*xf32>)

// relu(blur(image) + bias)
func.func @main() {
  %in = arith.constant dense<[[[[ 1.0],[ 2.0],[ 3.0],[ 4.0]],
                               [[ 5.0],[ 6.0],[ 7.0],[ 8.0]],
                               [[ 9.0],[10.0],[11.0],[12.0]],
                               [[13.0],[14.0],[15.0],[16.0]]]]> : tensor<1x4x4x1xf32>
  %k = arith.constant dense<1.0> : tensor<3x3x1x1xf32>

  // conv result is [[54, 63],[90, 99]]; bias drives two lanes negative.
  %bias = arith.constant dense<[[[[-60.0],[-60.0]],
                                 [[-60.0],[-60.0]]]]> : tensor<1x2x2x1xf32>

  %c = dsp.conv2d %in, %k : (tensor<1x4x4x1xf32>, tensor<3x3x1x1xf32>) -> tensor<1x2x2x1xf32>
  %s = dsp.add %c, %bias : tensor<1x2x2x1xf32>
  %r = dsp.relu %s : tensor<1x2x2x1xf32>

  // 54-60 = -6 -> 0 ; 63-60 =  3 ->  3
  // 90-60 = 30 -> 30 ; 99-60 = 39 -> 39
  %u = tensor.cast %r : tensor<1x2x2x1xf32> to tensor<*xf32>
  call @printMemrefF32(%u) : (tensor<*xf32>) -> ()
  return
}

// CHECK: rank = 4 offset = 0 sizes = [1, 2, 2, 1]
// CHECK: 0
// CHECK: 3
// CHECK: 30
// CHECK: 39