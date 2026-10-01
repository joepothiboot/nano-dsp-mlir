// RUN: nanodsp-opt %s -split-input-file -nanodsp-promote-local=target=hexagon-hvx128 \
// RUN:     -verify-diagnostics \
// RUN: | FileCheck %s
// RUN: nanodsp-opt %s -split-input-file -nanodsp-promote-local=target=host-neon \
// RUN: | FileCheck %s --check-prefix=NOLOCAL

// Bufferized cache-tile loops, as -nanodsp-optimize + -nanodsp-bufferize
// leave them; nanodsp.cache_loop marks the innermost cache-tile loop.
// host-neon has no local memory, so nothing changes there.
// NOLOCAL-NOT: #dsp.local
// NOLOCAL-NOT: memref.dma_start

// A k loop over matmul tiles: both input tiles change with k. The A tile is
// rows of 64 elements, 256 apart (one level of stride); the B tile is one
// contiguous run. The output tile is written, so it is not promoted.
// CHECK-LABEL: func.func @k_tiles
// CHECK-SAME:    (%[[A:.*]]: memref<128x256xf32>, %[[B:.*]]: memref<256x64xf32>, %[[C:.*]]: memref<128x64xf32>)
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C64:.*]] = arith.constant 64 : index
// CHECK-DAG:   %[[C256:.*]] = arith.constant 256 : index
// CHECK:       %[[LA:.*]] = memref.alloc() {alignment = 128 : i64} : memref<128x64xf32, #dsp.local>
// CHECK:       %[[TA:.*]] = memref.alloc() : memref<1xi32>
// CHECK:       %[[LB:.*]] = memref.alloc() {alignment = 128 : i64} : memref<64x64xf32, #dsp.local>
// CHECK:       %[[TB:.*]] = memref.alloc() : memref<1xi32>
// CHECK:       scf.for %[[K:.*]] = %[[C0]] to %[[C256]] step %[[C64]] {
// CHECK:         %[[NA:.*]] = arith.constant 8192 : index
// CHECK:         memref.dma_start %[[A]][%[[C0]], %[[K]]], %[[LA]][%[[C0]], %[[C0]]], %[[NA]], %[[TA]][%[[C0]]], %[[C256]], %[[C64]]
// CHECK-NEXT:    memref.dma_wait %[[TA]][%[[C0]]], %[[NA]]
// CHECK:         %[[NB:.*]] = arith.constant 4096 : index
// CHECK:         memref.dma_start %[[B]][%[[K]], %[[C0]]], %[[LB]][%[[C0]], %[[C0]]], %[[NB]], %[[TB]][%[[C0]]] :
// CHECK-NEXT:    memref.dma_wait %[[TB]][%[[C0]]], %[[NB]]
// CHECK-NEXT:    linalg.matmul ins(%[[LA]], %[[LB]] : memref<128x64xf32, #dsp.local>, memref<64x64xf32, #dsp.local>) outs(%[[C]] : memref<128x64xf32>)
// CHECK:       } {nanodsp.cache_loop}
// CHECK-DAG:   memref.dealloc %[[LA]] :
// CHECK-DAG:   memref.dealloc %[[TA]] :
// CHECK-DAG:   memref.dealloc %[[LB]] :
// CHECK-DAG:   memref.dealloc %[[TB]] :
func.func @k_tiles(%a: memref<128x256xf32>, %b: memref<256x64xf32>, %c: memref<128x64xf32>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c256 = arith.constant 256 : index
  scf.for %k = %c0 to %c256 step %c64 {
    %sa = memref.subview %a[0, %k] [128, 64] [1, 1] : memref<128x256xf32> to memref<128x64xf32, strided<[256, 1], offset: ?>>
    %sb = memref.subview %b[%k, 0] [64, 64] [1, 1] : memref<256x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
    linalg.matmul ins(%sa, %sb : memref<128x64xf32, strided<[256, 1], offset: ?>>, memref<64x64xf32, strided<[64, 1], offset: ?>>)
                  outs(%c : memref<128x64xf32>)
  } {nanodsp.cache_loop}
  return
}

// -----

// The innermost cache loop is n: the A tile only depends on m, so it is
// loaded once per m iteration, before the n loop. Buffers are allocated
// around the whole nest.
// CHECK-LABEL: func.func @invariant_tile
// CHECK:       %[[LA:.*]] = memref.alloc() {alignment = 128 : i64} : memref<64x256xf32, #dsp.local>
// CHECK:       %[[LB:.*]] = memref.alloc() {alignment = 128 : i64} : memref<256x64xf32, #dsp.local>
// CHECK:       scf.for %[[M:.*]] =
// CHECK:         memref.dma_start %{{.*}}[%[[M]], %{{.*}}], %[[LA]]
// CHECK-NEXT:    memref.dma_wait
// CHECK-NEXT:    scf.for %[[N:.*]] =
// CHECK:           memref.dma_start %{{.*}}[%{{.*}}, %[[N]]], %[[LB]]
// CHECK-NEXT:      memref.dma_wait
// CHECK:           linalg.matmul ins(%[[LA]], %[[LB]] :
// CHECK:         } {nanodsp.cache_loop}
// CHECK:       }
// CHECK:       memref.dealloc %[[LA]] :
func.func @invariant_tile(%a: memref<256x256xf32>, %b: memref<256x128xf32>, %c: memref<256x128xf32>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  scf.for %m = %c0 to %c256 step %c64 {
    scf.for %n = %c0 to %c128 step %c64 {
      %sa = memref.subview %a[%m, 0] [64, 256] [1, 1] : memref<256x256xf32> to memref<64x256xf32, strided<[256, 1], offset: ?>>
      %sb = memref.subview %b[0, %n] [256, 64] [1, 1] : memref<256x128xf32> to memref<256x64xf32, strided<[128, 1], offset: ?>>
      %sc = memref.subview %c[%m, %n] [64, 64] [1, 1] : memref<256x128xf32> to memref<64x64xf32, strided<[128, 1], offset: ?>>
      linalg.matmul ins(%sa, %sb : memref<64x256xf32, strided<[256, 1], offset: ?>>, memref<256x64xf32, strided<[128, 1], offset: ?>>)
                    outs(%sc : memref<64x64xf32, strided<[128, 1], offset: ?>>)
    } {nanodsp.cache_loop}
  }
  return
}

// -----

// Left alone: a loop without the marker, and a tile whose rows are not
// evenly strided (two levels of stride), which one DMA cannot describe.
// CHECK-LABEL: func.func @not_promoted
// CHECK-NOT:   #dsp.local
// CHECK-NOT:   memref.dma_start
// CHECK:       return
func.func @not_promoted(%a: memref<8x8x16xf32>, %out: memref<2x4x8xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c2 {
    %s = memref.subview %a[%i, 0, 0] [2, 4, 8] [1, 1, 1] : memref<8x8x16xf32> to memref<2x4x8xf32, strided<[128, 16, 1], offset: ?>>
    memref.copy %s, %out : memref<2x4x8xf32, strided<[128, 16, 1], offset: ?>> to memref<2x4x8xf32>
  }
  scf.for %i = %c0 to %c8 step %c2 {
    %s = memref.subview %a[%i, 0, 0] [2, 4, 8] [1, 1, 1] : memref<8x8x16xf32> to memref<2x4x8xf32, strided<[128, 16, 1], offset: ?>>
    linalg.copy ins(%s : memref<2x4x8xf32, strided<[128, 16, 1], offset: ?>>) outs(%out : memref<2x4x8xf32>)
  } {nanodsp.cache_loop}
  return
}

// -----

// 512x128 + 128x64 f32 tiles need 288 KiB; hexagon-hvx128 has 256 KiB of
// VTCM.
func.func @over_budget(%a: memref<512x256xf32>, %b: memref<256x64xf32>, %c: memref<512x64xf32>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  // expected-error @+1 {{input tiles of this cache tile need 294912 bytes of local memory, but target 'hexagon-hvx128' has 262144}}
  scf.for %k = %c0 to %c256 step %c128 {
    %sa = memref.subview %a[0, %k] [512, 128] [1, 1] : memref<512x256xf32> to memref<512x128xf32, strided<[256, 1], offset: ?>>
    %sb = memref.subview %b[%k, 0] [128, 64] [1, 1] : memref<256x64xf32> to memref<128x64xf32, strided<[64, 1], offset: ?>>
    linalg.matmul ins(%sa, %sb : memref<512x128xf32, strided<[256, 1], offset: ?>>, memref<128x64xf32, strided<[64, 1], offset: ?>>)
                  outs(%c : memref<512x64xf32>)
  } {nanodsp.cache_loop}
  return
}
