// RUN: nanodsp-opt %s -split-input-file -convert-dsp-to-linalg -nanodsp-optimize \
// RUN: | FileCheck %s
// RUN: not nanodsp-opt %s -split-input-file -convert-dsp-to-linalg \
// RUN:     -nanodsp-optimize=target=no-such-target 2>&1 \
// RUN: | FileCheck %s --check-prefix=BAD-TARGET
// RUN: not nanodsp-opt %s -split-input-file -convert-dsp-to-linalg \
// RUN:     -nanodsp-optimize=schedule-file=%S/no-such-schedule.mlir 2>&1 \
// RUN: | FileCheck %s --check-prefix=BAD-FILE

// BAD-TARGET: error: unknown target 'no-such-target' (known: host-neon, x86-avx2, hexagon-hvx128)
// BAD-FILE: no-such-schedule.mlir

// CHECK-LABEL: func.func @matmul
// CHECK-NOT:   linalg.generic
// CHECK-NOT:   vector.contract
// CHECK:       scf.for
// CHECK:         scf.for
// CHECK:           arith.mulf {{.*}} : vector<4x16xf32>
// CHECK:           arith.addf {{.*}} : vector<4x16xf32>
// CHECK-NOT:   nanodsp.tag
// CHECK-NOT:   transform.
func.func @matmul(%a: tensor<128x256xf32>, %b: tensor<256x96xf32>) -> tensor<128x96xf32> {
  %r = dsp.matmul %a, %b : (tensor<128x256xf32>, tensor<256x96xf32>) -> tensor<128x96xf32>
  return %r : tensor<128x96xf32>
}

// -----

// CHECK-LABEL: func.func @conv
// CHECK-NOT:   linalg.generic
// CHECK:       vector.transfer_read {{.*}} : tensor<4xf32>, vector<4x8xf32>
// CHECK:       vector.transfer_read {{.*}} : tensor<8xf32>, vector<4x8xf32>
// CHECK:       arith.mulf {{.*}} : vector<4x8xf32>
// CHECK:       arith.addf {{.*}} : vector<4x8xf32>
func.func @conv(%i: tensor<1x10x10x3xf32>, %f: tensor<3x3x3x8xf32>) -> tensor<1x8x8x8xf32> {
  %r = dsp.conv2d %i, %f : (tensor<1x10x10x3xf32>, tensor<3x3x3x8xf32>) -> tensor<1x8x8x8xf32>
  return %r : tensor<1x8x8x8xf32>
}

// -----

// CHECK-LABEL: func.func @relu
// CHECK-NOT:   linalg.generic
// CHECK:       arith.maximumf {{.*}} : vector<4xf32>
func.func @relu(%a: tensor<6x20xf32>) -> tensor<6x20xf32> {
  %r = dsp.relu %a : tensor<6x20xf32>
  return %r : tensor<6x20xf32>
}
