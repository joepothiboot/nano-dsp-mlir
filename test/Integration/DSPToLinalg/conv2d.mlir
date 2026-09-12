// RUN: nanodsp-opt %s -convert-dsp-to-linalg \
// RUN: | mlir-opt %stock_lower_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_runner_utils --shared-libs=%mlir_c_runner_utils \
// RUN: | FileCheck %s

func.func private @printMemrefF32(%ptr : tensor<*xf32>)

func.func @main() {
  // 4x4 input with values 1..16, 3x3 all-ones filter, stride 1 -> 2x2 output.
  // Distinct values catch an off-by-one or swapped H/W in the input map.
  %in = arith.constant dense<[[[[ 1.0],[ 2.0],[ 3.0],[ 4.0]],
                               [[ 5.0],[ 6.0],[ 7.0],[ 8.0]],
                               [[ 9.0],[10.0],[11.0],[12.0]],
                               [[13.0],[14.0],[15.0],[16.0]]]]> : tensor<1x4x4x1xf32>
  %f = arith.constant dense<1.0> : tensor<3x3x1x1xf32>

  // Expected window sums:
  //   (0,0): 1+2+3+5+6+7+9+10+11    = 54
  //   (0,1): 2+3+4+6+7+8+10+11+12   = 63
  //   (1,0): 5+6+7+9+10+11+13+14+15 = 90
  //   (1,1): 6+7+8+10+11+12+14+15+16 = 99
  %r = dsp.conv2d %in, %f : (tensor<1x4x4x1xf32>, tensor<3x3x1x1xf32>) -> tensor<1x2x2x1xf32>

  %u = tensor.cast %r : tensor<1x2x2x1xf32> to tensor<*xf32>
  call @printMemrefF32(%u) : (tensor<*xf32>) -> ()
  return
}

// CHECK: rank = 4 offset = 0 sizes = [1, 2, 2, 1] strides = [4, 2, 1, 1]
// CHECK: 54
// CHECK: 63
// CHECK: 90
// CHECK: 99