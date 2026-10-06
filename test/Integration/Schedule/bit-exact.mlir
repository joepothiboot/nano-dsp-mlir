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
// RUN: nanodsp-opt %s -convert-dsp-to-linalg -nanodsp-optimize=target=x86-avx2 \
// RUN:     -nanodsp-lower-to-llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_runner_utils --shared-libs=%mlir_c_runner_utils \
// RUN: | sed -e "s/base@ = 0x[0-9a-f]*//" > %t.avx2
// RUN: nanodsp-opt %s -convert-dsp-to-linalg \
// RUN:     -nanodsp-optimize=schedule-file=%S/../../../schedules/matmul-8x12-neon.mlir \
// RUN:     -nanodsp-lower-to-llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_runner_utils --shared-libs=%mlir_c_runner_utils \
// RUN: | sed -e "s/base@ = 0x[0-9a-f]*//" > %t.hand
// RUN: nanodsp-opt %s -convert-dsp-to-linalg \
// RUN:     -nanodsp-optimize=schedule-file=%S/Inputs/split-reduction.mlir \
// RUN:     -nanodsp-lower-to-llvm \
// RUN: | mlir-runner -e main --entry-point-result=void \
// RUN:     --shared-libs=%mlir_runner_utils --shared-libs=%mlir_c_runner_utils \
// RUN: | sed -e "s/base@ = 0x[0-9a-f]*//" > %t.split
// RUN: diff %t.ref %t.neon
// RUN: diff %t.ref %t.avx2
// RUN: diff %t.ref %t.hand
// RUN: not diff %t.ref %t.split > /dev/null
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
func.func @print4(%t: tensor<?x?x?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %d0 = tensor.dim %t, %c0 : tensor<?x?x?x?xf32>
  %d1 = tensor.dim %t, %c1 : tensor<?x?x?x?xf32>
  %d2 = tensor.dim %t, %c2 : tensor<?x?x?x?xf32>
  %d3 = tensor.dim %t, %c3 : tensor<?x?x?x?xf32>
  %bits = tensor.generate %d0, %d1, %d2, %d3 {
  ^bb0(%i: index, %j: index, %k: index, %l: index):
    %x = tensor.extract %t[%i, %j, %k, %l] : tensor<?x?x?x?xf32>
    %b = arith.bitcast %x : f32 to i32
    tensor.yield %b : i32
  } : tensor<?x?x?x?xi32>
  %u = tensor.cast %bits : tensor<?x?x?x?xi32> to tensor<*xi32>
  call @printMemrefI32(%u) : (tensor<*xi32>) -> ()
  return
}

func.func @main() {
  %a = tensor.generate {
  ^bb0(%i: index, %j: index):
    %v = func.call @fill2(%i, %j) : (index, index) -> f32
    tensor.yield %v : f32
  } : tensor<64x96xf32>
  %b = tensor.generate {
  ^bb0(%i: index, %j: index):
    %v = func.call @fill2(%j, %i) : (index, index) -> f32
    tensor.yield %v : f32
  } : tensor<96x48xf32>
  %mm = dsp.matmul %a, %b : (tensor<64x96xf32>, tensor<96x48xf32>) -> tensor<64x48xf32>
  %mmd = tensor.cast %mm : tensor<64x48xf32> to tensor<?x?xf32>
  call @print2(%mmd) : (tensor<?x?xf32>) -> ()

  %in = tensor.generate {
  ^bb0(%n: index, %h: index, %w: index, %c: index):
    %v = func.call @fill4(%n, %h, %w, %c) : (index, index, index, index) -> f32
    tensor.yield %v : f32
  } : tensor<1x12x12x3xf32>
  %f = tensor.generate {
  ^bb0(%h: index, %w: index, %c: index, %o: index):
    %v = func.call @fill4(%o, %c, %h, %w) : (index, index, index, index) -> f32
    tensor.yield %v : f32
  } : tensor<3x3x3x8xf32>
  %c1 = dsp.conv2d %in, %f : (tensor<1x12x12x3xf32>, tensor<3x3x3x8xf32>) -> tensor<1x10x10x8xf32>
  %c2 = dsp.conv2d %in, %f {strides = array<i64: 2, 2>}
      : (tensor<1x12x12x3xf32>, tensor<3x3x3x8xf32>) -> tensor<1x5x5x8xf32>
  %c3 = dsp.conv2d %in, %f {dilations = array<i64: 2, 2>}
      : (tensor<1x12x12x3xf32>, tensor<3x3x3x8xf32>) -> tensor<1x8x8x8xf32>
  %c1d = tensor.cast %c1 : tensor<1x10x10x8xf32> to tensor<?x?x?x?xf32>
  %c2d = tensor.cast %c2 : tensor<1x5x5x8xf32> to tensor<?x?x?x?xf32>
  %c3d = tensor.cast %c3 : tensor<1x8x8x8xf32> to tensor<?x?x?x?xf32>
  call @print4(%c1d) : (tensor<?x?x?x?xf32>) -> ()
  call @print4(%c2d) : (tensor<?x?x?x?xf32>) -> ()
  call @print4(%c3d) : (tensor<?x?x?x?xf32>) -> ()

  %s = tensor.extract_slice %a[0, 0] [6, 20] [1, 1] : tensor<64x96xf32> to tensor<6x20xf32>
  %t = tensor.extract_slice %a[6, 3] [6, 20] [1, 1] : tensor<64x96xf32> to tensor<6x20xf32>
  %sum = dsp.add %s, %t : tensor<6x20xf32>
  %r = dsp.relu %sum : tensor<6x20xf32>
  %rd = tensor.cast %r : tensor<6x20xf32> to tensor<?x?xf32>
  call @print2(%rd) : (tensor<?x?xf32>) -> ()
  return
}

// CHECK: sizes = [64, 48]
// CHECK: sizes = [1, 10, 10, 8]
// CHECK: sizes = [1, 5, 5, 8]
// CHECK: sizes = [1, 8, 8, 8]
// CHECK: sizes = [6, 20]
