// RUN: nanodsp-opt %s -convert-dsp-to-linalg \
// RUN: | mlir-opt %stock_lower_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_runner_utils --shared-libs=%mlir_c_runner_utils \
// RUN: | FileCheck %s

func.func private @printMemrefF32(%ptr : tensor<*xf32>)

func.func @main() {
  // Covers: negative, negative-zero-adjacent, exact zero, positive, and a
  // large-magnitude negative.
  %a = arith.constant dense<[[-2.0, -1.0, 0.0, 1.0, 2.0, -3.0]]> : tensor<1x6xf32>

  %r = dsp.relu %a : tensor<1x6xf32>

  %u = tensor.cast %r : tensor<1x6xf32> to tensor<*xf32>
  call @printMemrefF32(%u) : (tensor<*xf32>) -> ()
  return
}

// CHECK: rank = 2 offset = 0 sizes = [1, 6] strides = [6, 1]
// CHECK-NEXT: [0,   0,   0,   1,   2,   0]