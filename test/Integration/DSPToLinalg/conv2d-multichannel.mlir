// RUN: nanodsp-opt %s -convert-dsp-to-linalg \
// RUN: | mlir-opt %stock_lower_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_runner_utils --shared-libs=%mlir_c_runner_utils \
// RUN: | FileCheck %s

func.func private @printMemrefF32(%ptr : tensor<*xf32>)

func.func @main() {
  // 1x3x3x2 input: channel 0 is all 1.0, channel 1 is all 10.0.
  %in = arith.constant dense<[[[[1.0, 10.0],[1.0, 10.0],[1.0, 10.0]],
                               [[1.0, 10.0],[1.0, 10.0],[1.0, 10.0]],
                               [[1.0, 10.0],[1.0, 10.0],[1.0, 10.0]]]]> : tensor<1x3x3x2xf32>

  // 3x3x2x2 filter (HWCF).
  //   output feature 0: weight 1.0 on channel 0, 0.0 on channel 1
  //   output feature 1: weight 0.0 on channel 0, 1.0 on channel 1
  // -> feature 0 = 9 * 1.0 = 9 ; feature 1 = 9 * 10.0 = 90
  // Swapping the C and F positions in the filter map gives 0/99 instead.
  %f = arith.constant dense<[
    [[[1.0, 0.0],[0.0, 1.0]], [[1.0, 0.0],[0.0, 1.0]], [[1.0, 0.0],[0.0, 1.0]]],
    [[[1.0, 0.0],[0.0, 1.0]], [[1.0, 0.0],[0.0, 1.0]], [[1.0, 0.0],[0.0, 1.0]]],
    [[[1.0, 0.0],[0.0, 1.0]], [[1.0, 0.0],[0.0, 1.0]], [[1.0, 0.0],[0.0, 1.0]]]
  ]> : tensor<3x3x2x2xf32>

  %r = dsp.conv2d %in, %f : (tensor<1x3x3x2xf32>, tensor<3x3x2x2xf32>) -> tensor<1x1x1x2xf32>

  %u = tensor.cast %r : tensor<1x1x1x2xf32> to tensor<*xf32>
  call @printMemrefF32(%u) : (tensor<*xf32>) -> ()
  return
}

// CHECK: rank = 4 offset = 0 sizes = [1, 1, 1, 2]
// CHECK: 9
// CHECK: 90