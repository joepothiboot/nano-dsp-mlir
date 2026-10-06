// RUN: nanodsp-opt %s -convert-dsp-to-linalg \
// RUN: | mlir-opt %stock_lower_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_runner_utils --shared-libs=%mlir_c_runner_utils \
// RUN: | sed -e "s/base@ = 0x[0-9a-f]*//" > %t.ref
// RUN: nanodsp-opt %s -convert-dsp-to-linalg -nanodsp-optimize=target=host-neon \
// RUN:     -nanodsp-lower-to-llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_runner_utils --shared-libs=%mlir_c_runner_utils \
// RUN: | sed -e "s/base@ = 0x[0-9a-f]*//" > %t.neon
// RUN: diff %t.ref %t.neon
// RUN: FileCheck %s < %t.ref

func.func private @printMemrefI32(%ptr : tensor<*xi32>)

func.func @fill2(%i: index, %j: index) -> f32 {
  %z = arith.constant 0 : index
  %r = func.call @fill4(%i, %j, %z, %z) : (index, index, index, index) -> f32
  return %r : f32
}
func.func @fill4(%i: index, %j: index, %k: index, %l: index) -> f32 {
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %c7 = arith.constant 7 : index
  %c11 = arith.constant 11 : index
  %c13 = arith.constant 13 : index
  %a = arith.muli %i, %c7 : index
  %b = arith.muli %j, %c3 : index
  %c = arith.muli %k, %c5 : index
  %d = arith.muli %l, %c11 : index
  %s0 = arith.addi %a, %b : index
  %s1 = arith.addi %s0, %c : index
  %s2 = arith.addi %s1, %d : index
  %m = arith.remui %s2, %c13 : index
  %mi = arith.index_cast %m : index to i32
  %mf = arith.sitofp %mi : i32 to f32
  %scale = arith.constant 0.37 : f32
  %bias = arith.constant 1.9 : f32
  %p = arith.mulf %mf, %scale : f32
  %v = arith.subf %p, %bias : f32
  return %v : f32
}

func.func @print2(%t: tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %t, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %t, %c1 : tensor<?x?xf32>
  %bits = tensor.generate %d0, %d1 {
  ^bb0(%i: index, %j: index):
    %x = tensor.extract %t[%i, %j] : tensor<?x?xf32>
    %b = arith.bitcast %x : f32 to i32
    tensor.yield %b : i32
  } : tensor<?x?xi32>
  %u = tensor.cast %bits : tensor<?x?xi32> to tensor<*xi32>
  call @printMemrefI32(%u) : (tensor<*xi32>) -> ()
  return
}
func.func @main() {
  %a = tensor.generate {
  ^bb0(%i: index, %j: index):
    %v = func.call @fill2(%i, %j) : (index, index) -> f32
    tensor.yield %v : f32
  } : tensor<256x256xf32>
  %b = tensor.generate {
  ^bb0(%i: index, %j: index):
    %v = func.call @fill2(%j, %i) : (index, index) -> f32
    tensor.yield %v : f32
  } : tensor<256x128xf32>
  %mm = dsp.matmul %a, %b : (tensor<256x256xf32>, tensor<256x128xf32>) -> tensor<256x128xf32>
  %mmd = tensor.cast %mm : tensor<256x128xf32> to tensor<?x?xf32>
  call @print2(%mmd) : (tensor<?x?xf32>) -> ()
  return
}

// CHECK: sizes = [256, 128]
