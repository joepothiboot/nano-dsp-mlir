// RUN: nanodsp-opt %s -split-input-file -convert-dsp-to-linalg \
// RUN:     -nanodsp-optimize=target=mn1 -nanodsp-bufferize \
// RUN:     -nanodsp-promote-local=target=mn1 | FileCheck %s

// CHECK-LABEL: func.func @matmul_tiled
// CHECK:         %[[C:.*]] = memref.alloc() {alignment = 64 : i64} : memref<128x128xf32>
// CHECK:         %[[CL:.*]] = memref.alloc() {alignment = 32 : i64} : memref<32x128xf32, #dsp.local>
// CHECK:         scf.for %[[M:.*]] =
// CHECK:           memref.dma_start %[[C]][%[[M]], %{{.*}}], %[[CL]][
// CHECK-NEXT:      memref.dma_wait
// CHECK:           scf.for
// CHECK:             vector.transfer_write {{.*}} #dsp.local>
// CHECK:           } {nanodsp.cache_loop}
// CHECK-NEXT:      memref.dma_start %[[CL]][%{{.*}}, %{{.*}}], %[[C]][%[[M]], %{{.*}}]
// CHECK-NEXT:      memref.dma_wait
func.func @matmul_tiled(%a: tensor<128x128xf32>, %b: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %c = dsp.matmul %a, %b : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
  return %c : tensor<128x128xf32>
}

// -----

// CHECK-LABEL: func.func @matmul_staged
// CHECK-SAME:    (%[[A:.*]]: memref<32x32xf32>, %[[B:.*]]: memref<32x32xf32>)
// CHECK:         %[[C:.*]] = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
// CHECK:         linalg.fill {{.*}} outs(%[[C]]
// CHECK:         memref.dma_start %[[A]][{{.*}}], %{{.*}}[{{.*}}], %c1024
// CHECK:         memref.dma_start %[[B]][{{.*}}], %{{.*}}[{{.*}}], %c1024
// CHECK:         memref.dma_start %[[C]][{{.*}}], %[[CL:.*]][{{.*}}], %c1024
// CHECK:         scf.for
// CHECK-NOT:       memref<32x32xf32>{{$}}
// CHECK:         memref.dma_start %[[CL]][{{.*}}], %[[C]][{{.*}}], %c1024
// CHECK-NEXT:    memref.dma_wait
func.func @matmul_staged(%a: tensor<32x32xf32>, %b: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %c = dsp.matmul %a, %b : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %c : tensor<32x32xf32>
}

// -----

// CHECK-LABEL: func.func @add_tiled
// CHECK:         %[[C:.*]] = memref.alloc() {alignment = 64 : i64} : memref<128x128xf32>
// CHECK:         %[[CL:.*]] = memref.alloc() {alignment = 32 : i64} : memref<16x128xf32, #dsp.local>
// CHECK:         scf.for %[[R:.*]] = %{{.*}} to %{{.*}} step %c16
// CHECK-NOT:       memref.dma_start %[[C]]
// CHECK:           vector.transfer_write
// CHECK:           memref.dma_start %[[CL]][%{{.*}}, %{{.*}}], %[[C]][%[[R]], %{{.*}}], %c2048
// CHECK-NEXT:      memref.dma_wait
func.func @add_tiled(%a: tensor<128x128xf32>, %b: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %c = dsp.add %a, %b : tensor<128x128xf32>
  return %c : tensor<128x128xf32>
}
