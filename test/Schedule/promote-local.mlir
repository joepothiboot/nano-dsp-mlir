// RUN: nanodsp-opt %s -split-input-file -nanodsp-promote-local=target=hexagon-hvx128 \
// RUN:     -verify-diagnostics \
// RUN: | FileCheck %s --check-prefixes=CHECK,DB
// RUN: nanodsp-opt %s -split-input-file \
// RUN:     -nanodsp-promote-local="target=hexagon-hvx128 double-buffer=false" \
// RUN:     -verify-diagnostics \
// RUN: | FileCheck %s --check-prefixes=CHECK,SINGLE
// RUN: nanodsp-opt %s -split-input-file -nanodsp-promote-local=target=host-neon \
// RUN: | FileCheck %s --check-prefix=NOLOCAL

// NOLOCAL-NOT: #dsp.local
// NOLOCAL-NOT: memref.dma_start

// CHECK-LABEL:  func.func @k_tiles
// CHECK-SAME:     (%[[A:.*]]: memref<128x256xf32>, %[[B:.*]]: memref<256x64xf32>, %[[C:.*]]: memref<128x64xf32>)
// CHECK-DAG:    %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:    %[[C64:.*]] = arith.constant 64 : index
// CHECK-DAG:    %[[C256:.*]] = arith.constant 256 : index
// DB:           %[[LA:.*]] = memref.alloc() {alignment = 128 : i64} : memref<2x128x64xf32, #dsp.local>
// DB-NEXT:      %[[TA:.*]] = memref.alloc() : memref<2x1xi32>
// DB-NEXT:      %[[LB:.*]] = memref.alloc() {alignment = 128 : i64} : memref<2x64x64xf32, #dsp.local>
// DB-NEXT:      %[[TB:.*]] = memref.alloc() : memref<2x1xi32>
// DB-NEXT:      %[[TA0:.*]] = memref.subview %[[TA]][%[[C0]], 0] [1, 1] [1, 1]
// DB-NEXT:      %[[LA0:.*]] = memref.subview %[[LA]][%[[C0]], 0, 0] [1, 128, 64] [1, 1, 1] : memref<2x128x64xf32, #dsp.local> to memref<128x64xf32, strided<[64, 1], offset: ?>, #dsp.local>
// DB-NEXT:      %[[NA:.*]] = arith.constant 8192 : index
// DB-NEXT:      memref.dma_start %[[A]][%[[C0]], %[[C0]]], %[[LA0]][%[[C0]], %[[C0]]], %[[NA]], %[[TA0]][%[[C0]]], %[[C256]], %[[C64]]
// DB-NEXT:      %[[TB0:.*]] = memref.subview %[[TB]][%[[C0]], 0] [1, 1] [1, 1]
// DB-NEXT:      %[[LB0:.*]] = memref.subview %[[LB]][%[[C0]], 0, 0] [1, 64, 64] [1, 1, 1]
// DB-NEXT:      %[[NB:.*]] = arith.constant 4096 : index
// DB-NEXT:      memref.dma_start %[[B]][%[[C0]], %[[C0]]], %[[LB0]][%[[C0]], %[[C0]]], %[[NB]], %[[TB0]][%[[C0]]] :
// DB-NEXT:      scf.for %[[K:.*]] = %[[C0]] to %[[C256]] step %[[C64]] {
// DB-NEXT:        %[[SLOT:.*]] = affine.apply #{{.*}}(%[[K]])
// DB-NEXT:        %[[TBK:.*]] = memref.subview %[[TB]][%[[SLOT]], 0] [1, 1] [1, 1]
// DB-NEXT:        %[[LBK:.*]] = memref.subview %[[LB]][%[[SLOT]], 0, 0] [1, 64, 64] [1, 1, 1]
// DB-NEXT:        %[[TAK:.*]] = memref.subview %[[TA]][%[[SLOT]], 0] [1, 1] [1, 1]
// DB-NEXT:        %[[LAK:.*]] = memref.subview %[[LA]][%[[SLOT]], 0, 0] [1, 128, 64] [1, 1, 1]
// DB-NEXT:        %[[NEXT:.*]] = arith.addi %[[K]], %[[C64]] : index
// DB-NEXT:        %[[HAS:.*]] = arith.cmpi slt, %[[NEXT]], %[[C256]] : index
// DB-NEXT:        scf.if %[[HAS]] {
// DB-NEXT:          %[[NSLOT:.*]] = affine.apply #{{.*}}(%[[NEXT]])
// DB-NEXT:          %[[TAN:.*]] = memref.subview %[[TA]][%[[NSLOT]], 0] [1, 1] [1, 1]
// DB-NEXT:          %[[LAN:.*]] = memref.subview %[[LA]][%[[NSLOT]], 0, 0] [1, 128, 64] [1, 1, 1]
// DB-NEXT:          memref.dma_start %[[A]][%[[C0]], %[[NEXT]]], %[[LAN]][%[[C0]], %[[C0]]], %[[NA]], %[[TAN]][%[[C0]]], %[[C256]], %[[C64]]
// DB-NEXT:          %[[TBN:.*]] = memref.subview %[[TB]][%[[NSLOT]], 0] [1, 1] [1, 1]
// DB-NEXT:          %[[LBN:.*]] = memref.subview %[[LB]][%[[NSLOT]], 0, 0] [1, 64, 64] [1, 1, 1]
// DB-NEXT:          memref.dma_start %[[B]][%[[NEXT]], %[[C0]]], %[[LBN]][%[[C0]], %[[C0]]], %[[NB]], %[[TBN]][%[[C0]]] :
// DB-NEXT:        }
// DB-NEXT:        memref.dma_wait %[[TAK]][%[[C0]]], %[[NA]]
// DB-NEXT:        memref.dma_wait %[[TBK]][%[[C0]]], %[[NB]]
// DB-NEXT:        linalg.matmul ins(%[[LAK]], %[[LBK]] : memref<128x64xf32, strided<[64, 1], offset: ?>, #dsp.local>, memref<64x64xf32, strided<[64, 1], offset: ?>, #dsp.local>) outs(%[[C]] : memref<128x64xf32>)
// DB-NEXT:      } {nanodsp.cache_loop}
// SINGLE:       %[[LA:.*]] = memref.alloc() {alignment = 128 : i64} : memref<128x64xf32, #dsp.local>
// SINGLE-NEXT:  %[[TA:.*]] = memref.alloc() : memref<1xi32>
// SINGLE-NEXT:  %[[LB:.*]] = memref.alloc() {alignment = 128 : i64} : memref<64x64xf32, #dsp.local>
// SINGLE-NEXT:  %[[TB:.*]] = memref.alloc() : memref<1xi32>
// SINGLE-NEXT:  scf.for %[[K:.*]] = %[[C0]] to %[[C256]] step %[[C64]] {
// SINGLE-NEXT:    %[[NA:.*]] = arith.constant 8192 : index
// SINGLE-NEXT:    memref.dma_start %[[A]][%[[C0]], %[[K]]], %[[LA]][%[[C0]], %[[C0]]], %[[NA]], %[[TA]][%[[C0]]], %[[C256]], %[[C64]]
// SINGLE-NEXT:    memref.dma_wait %[[TA]][%[[C0]]], %[[NA]]
// SINGLE-NEXT:    %[[NB:.*]] = arith.constant 4096 : index
// SINGLE-NEXT:    memref.dma_start %[[B]][%[[K]], %[[C0]]], %[[LB]][%[[C0]], %[[C0]]], %[[NB]], %[[TB]][%[[C0]]] :
// SINGLE-NEXT:    memref.dma_wait %[[TB]][%[[C0]]], %[[NB]]
// SINGLE-NEXT:    linalg.matmul ins(%[[LA]], %[[LB]] : memref<128x64xf32, #dsp.local>, memref<64x64xf32, #dsp.local>) outs(%[[C]] : memref<128x64xf32>)
// SINGLE-NEXT:  } {nanodsp.cache_loop}
// CHECK-DAG:    memref.dealloc %[[LA]] :
// CHECK-DAG:    memref.dealloc %[[TA]] :
// CHECK-DAG:    memref.dealloc %[[LB]] :
// CHECK-DAG:    memref.dealloc %[[TB]] :
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

// CHECK-LABEL: func.func @invariant_tile
// CHECK:       %[[LA:.*]] = memref.alloc() {alignment = 128 : i64} : memref<64x256xf32, #dsp.local>
// DB:          memref.alloc() {alignment = 128 : i64} : memref<2x256x64xf32, #dsp.local>
// SINGLE:      memref.alloc() {alignment = 128 : i64} : memref<256x64xf32, #dsp.local>
// CHECK:       scf.for %[[M:.*]] =
// CHECK-NEXT:    arith.constant 16384
// CHECK-NEXT:    memref.dma_start %{{.*}}[%[[M]], %{{.*}}], %[[LA]]
// CHECK-NEXT:    memref.dma_wait
// DB:            memref.dma_start
// CHECK:         scf.for
// DB:              scf.if
// CHECK:             memref.dma_start {{.*}} : memref<256x128xf32>, memref<256x64xf32, {{.*}}#dsp.local>
// CHECK:           memref.dma_wait
// CHECK:           linalg.matmul ins(%[[LA]], %{{.*}} :
// CHECK:         } {nanodsp.cache_loop}
// CHECK-NEXT:  }
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

// -----

// CHECK-LABEL: func.func @single_buffer_fallback
// CHECK:       memref.alloc() {alignment = 128 : i64} : memref<256x128xf32, #dsp.local>
// CHECK:       memref.alloc() {alignment = 128 : i64} : memref<128x64xf32, #dsp.local>
// CHECK:       scf.for
// CHECK-NOT:     scf.if
// CHECK:         memref.dma_start
// CHECK-NEXT:    memref.dma_wait
// CHECK:         memref.dma_start
// CHECK-NEXT:    memref.dma_wait
// CHECK-NEXT:    linalg.matmul
func.func @single_buffer_fallback(%a: memref<256x256xf32>, %b: memref<256x64xf32>, %c: memref<256x64xf32>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  scf.for %k = %c0 to %c256 step %c128 {
    %sa = memref.subview %a[0, %k] [256, 128] [1, 1] : memref<256x256xf32> to memref<256x128xf32, strided<[256, 1], offset: ?>>
    %sb = memref.subview %b[%k, 0] [128, 64] [1, 1] : memref<256x64xf32> to memref<128x64xf32, strided<[64, 1], offset: ?>>
    linalg.matmul ins(%sa, %sb : memref<256x128xf32, strided<[256, 1], offset: ?>>, memref<128x64xf32, strided<[64, 1], offset: ?>>)
                  outs(%c : memref<256x64xf32>)
  } {nanodsp.cache_loop}
  return
}
