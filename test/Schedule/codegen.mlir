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
// RUN: nanodsp-opt %s -convert-dsp-to-linalg -nanodsp-optimize=target=host-neon \
// RUN:     -nanodsp-lower-to-llvm \
// RUN: | FileCheck %s --check-prefix=STACK

// NEON-COUNT-16: fmul.4s
// NEON-NOT:      fmla
// NEON-NOT:      fmul s{{[0-9]+}}

// AVX2-COUNT-12: vmulps {{.*}}%ymm
// AVX2-NOT:      vfmadd
// AVX2-NOT:      vmulss

// STACK-LABEL:   llvm.func @matmul(
// STACK-NOT:     llvm.alloca

func.func @matmul(%a: tensor<128x256xf32>, %b: tensor<256x96xf32>) -> tensor<128x96xf32> {
  %r = dsp.matmul %a, %b : (tensor<128x256xf32>, tensor<256x96xf32>) -> tensor<128x96xf32>
  return %r : tensor<128x96xf32>
}
