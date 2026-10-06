// RUN: nanodsp-opt %s -convert-dsp-to-linalg \
// RUN: | mlir-opt %stock_lower_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_runner_utils --shared-libs=%mlir_c_runner_utils \
// RUN: | FileCheck %s
// RUN: nanodsp-opt %s -convert-dsp-to-linalg -nanodsp-optimize=target=host-neon \
// RUN:     -nanodsp-lower-to-llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_runner_utils --shared-libs=%mlir_c_runner_utils \
// RUN: | FileCheck %s

func.func private @printMemrefI32(%ptr : tensor<*xi32>)

func.func @print(%t: tensor<?x?xi8>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %t, %c0 : tensor<?x?xi8>
  %d1 = tensor.dim %t, %c1 : tensor<?x?xi8>
  %w = tensor.generate %d0, %d1 {
  ^bb0(%i: index, %j: index):
    %x = tensor.extract %t[%i, %j] : tensor<?x?xi8>
    %y = arith.extsi %x : i8 to i32
    tensor.yield %y : i32
  } : tensor<?x?xi32>
  %u = tensor.cast %w : tensor<?x?xi32> to tensor<*xi32>
  call @printMemrefI32(%u) : (tensor<*xi32>) -> ()
  return
}

func.func @main() {
  %a = arith.constant dense<[[-128, 0, 127],
                             [  10, -7, 50]]> : tensor<2x3xi8>
  %b = arith.constant dense<[[ 1, -3,  127, -128],
                             [ 4,  2,    0,    9],
                             [-1,  5,  -55,   30]]> : tensor<3x4xi8>
  %r = dsp.qmatmul %a, %b {lhs_zp = 3 : i32, rhs_zp = -2 : i32,
                           multiplier = 1073741824 : i32, shift = 3 : i32,
                           out_zp = -5 : i32}
     : (tensor<2x3xi8>, tensor<3x4xi8>) -> tensor<2x4xi8>
  %rd = tensor.cast %r : tensor<2x4xi8> to tensor<?x?xi8>
  call @print(%rd) : (tensor<?x?xi8>) -> ()

  %c = arith.constant dense<[[127, -128, 64, -1],
                             [-50,   33, -2, 90]]> : tensor<2x4xi8>
  %d = arith.constant dense<[[  3,   -7],
                             [ -2,   11],
                             [100, -100],
                             [ -9,    4]]> : tensor<4x2xi8>
  %s = dsp.qmatmul %c, %d {lhs_zp = 0 : i32, rhs_zp = 0 : i32,
                           multiplier = 1518500250 : i32, shift = 5 : i32,
                           out_zp = 1 : i32}
     : (tensor<2x4xi8>, tensor<4x2xi8>) -> tensor<2x2xi8>
  %sd = tensor.cast %s : tensor<2x2xi8> to tensor<?x?xi8>
  call @print(%sd) : (tensor<?x?xi8>) -> ()
  return
}

// CHECK:      sizes = [2, 4]
// CHECK-NEXT: [-23,   57,   -128,   127]
// CHECK-NEXT: [-4,   13,   -105,   27]
// CHECK:      sizes = [2, 2]
// CHECK-NEXT: [127,   -128]
// CHECK-NEXT: [-26,   29]
