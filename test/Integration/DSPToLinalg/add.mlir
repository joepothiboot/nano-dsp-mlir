// RUN: nanodsp-opt %s -convert-dsp-to-linalg \
// RUN: | mlir-opt %stock_lower_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_runner_utils --shared-libs=%mlir_c_runner_utils \
// RUN: | FileCheck %s

func.func private @printMemrefF32(%ptr : tensor<*xf32>)

func.func @main() {
  %a = arith.constant dense<[[1.0, 2.0, 3.0],
                             [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %b = arith.constant dense<[[10.0, 20.0, 30.0],
                             [40.0, 50.0, 60.0]]> : tensor<2x3xf32>

  %r = dsp.add %a, %b : tensor<2x3xf32>

  %u = tensor.cast %r : tensor<2x3xf32> to tensor<*xf32>
  call @printMemrefF32(%u) : (tensor<*xf32>) -> ()
  return
}

// CHECK: rank = 2 offset = 0 sizes = [2, 3] strides = [3, 1]
// CHECK-NEXT: [11,   22,   33]
// CHECK-NEXT: [44,   55,   66]