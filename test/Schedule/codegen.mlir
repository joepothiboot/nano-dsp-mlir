// The scheduled matmul must reach machine code as full-width vector
// multiplies and adds, never scalar multiplies and never FMA (which would
// break bit-exactness).
//
// RUN: nanodsp-opt %s -convert-dsp-to-linalg -nanodsp-optimize=target=host-neon \
// RUN:     -nanodsp-lower-to-llvm \
// RUN: | mlir-translate --mlir-to-llvmir \
// RUN: | llc -O2 -mtriple=aarch64-apple-darwin \
// RUN: | FileCheck %s --check-prefix=NEON
// RUN: nanodsp-opt %s -convert-dsp-to-linalg -nanodsp-optimize=target=x86-avx2 \
// RUN:     -nanodsp-lower-to-llvm \
// RUN: | mlir-translate --mlir-to-llvmir \
// RUN: | llc -O2 -mtriple=x86_64-unknown-linux-gnu -mattr=+avx2 \
// RUN: | FileCheck %s --check-prefix=AVX2

// host-neon: 4x16 register tile = 4 rows x 4 q-registers.
// NEON-COUNT-16: fmul.4s
// NEON-NOT:      fmla
// NEON-NOT:      fmul s{{[0-9]+}}

// x86-avx2: 4x24 register tile = 4 rows x 3 ymm registers.
// AVX2-COUNT-12: vmulps {{.*}}%ymm
// AVX2-NOT:      vfmadd
// AVX2-NOT:      vmulss

func.func @matmul(%a: tensor<128x256xf32>, %b: tensor<256x96xf32>) -> tensor<128x96xf32> {
  %r = dsp.matmul %a, %b : (tensor<128x256xf32>, tensor<256x96xf32>) -> tensor<128x96xf32>
  return %r : tensor<128x96xf32>
}
