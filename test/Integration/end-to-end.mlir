// RUN: nanodsp-opt %s -convert-dsp-to-linalg -one-shot-bufferize -convert-linalg-to-loops -convert-scf-to-cf -convert-cf-to-llvm \
// RUN: -convert-vector-to-llvm -convert-arith-to-llvm -finalize-memref-to-llvm \
// RUN: -convert-func-to-llvm -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main --entry-point-result=i32 \
// RUN:     --shared-libs=%mlir_runner_utils --shared-libs=%mlir_c_runner_utils \
// RUN: | FileCheck %s

func.func @main() -> i32 {
  %c1 = arith.constant 1.0 : f32
  %c2 = arith.constant 2.0 : f32
  %zero = arith.constant 0 : index
  %a = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %b = arith.constant dense<[1.0, 1.0, 1.0, 1.0]> : tensor<4xf32>
  
  %0 = dsp.add %a, %b : tensor<4xf32>

  %val = tensor.extract %0[%zero] : tensor<4xf32>
  %expected = arith.constant 2.0 : f32
  %cmp = arith.cmpf oeq, %val, %expected : f32
  %res = arith.extui %cmp : i1 to i32
  return %res : i32
}
// CHECK: 1