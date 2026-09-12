// RUN: nanodsp-opt %s -convert-dsp-to-linalg \
// RUN: | mlir-opt %stock_lower_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_runner_utils --shared-libs=%mlir_c_runner_utils \
// RUN: | FileCheck %s

func.func private @printMemrefF32(%ptr : tensor<*xf32>)

func.func @main() {
  // A is 2x3, B is 3x2. Non-square on purpose: a transposed indexing map
  // would still typecheck on square inputs but produces wrong numbers here.
  %a = arith.constant dense<[[1.0, 2.0, 3.0],
                             [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %b = arith.constant dense<[[1.0, 0.0],
                             [0.0, 1.0],
                             [1.0, 1.0]]> : tensor<3x2xf32>

  // Expected:
  //   [1*1 + 2*0 + 3*1, 1*0 + 2*1 + 3*1] = [4, 5]
  //   [4*1 + 5*0 + 6*1, 4*0 + 5*1 + 6*1] = [10, 11]
  %r = dsp.matmul %a, %b : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>

  %u = tensor.cast %r : tensor<2x2xf32> to tensor<*xf32>
  call @printMemrefF32(%u) : (tensor<*xf32>) -> ()
  return
}

// CHECK: rank = 2 offset = 0 sizes = [2, 2] strides = [2, 1]
// CHECK-NEXT: [4,   5]
// CHECK-NEXT: [10,   11]