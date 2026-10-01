// RUN: nanodsp-opt %s -split-input-file -nanodsp-lower-local -verify-diagnostics \
// RUN: | FileCheck %s

// A strided DMA into a #dsp.local buffer becomes a synchronous linalg.copy
// of the same tile; the wait and the tag buffer disappear, and the local
// buffer moves to the default memory space.
// CHECK-LABEL: func.func @strided_dma
// CHECK-SAME:    (%[[A:.*]]: memref<128x256xf32>, %[[K:.*]]: index)
// CHECK-NOT:   memref<1xi32>
// CHECK:       %[[L:.*]] = memref.alloc() {alignment = 128 : i64} : memref<128x64xf32>
// CHECK-NEXT:  %[[S:.*]] = memref.subview %[[A]][0, %[[K]]] [128, 64] [1, 1] : memref<128x256xf32> to memref<128x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:  linalg.copy ins(%[[S]] : memref<128x64xf32, strided<[256, 1], offset: ?>>) outs(%[[L]] : memref<128x64xf32>)
// CHECK-NEXT:  %[[V:.*]] = memref.load %[[L]]
// CHECK-NEXT:  memref.dealloc %[[L]] : memref<128x64xf32>
// CHECK-NEXT:  return %[[V]]
// CHECK-NOT:   #dsp.local
// CHECK-NOT:   memref.dma
func.func @strided_dma(%a: memref<128x256xf32>, %k: index) -> f32 {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c256 = arith.constant 256 : index
  %n = arith.constant 8192 : index
  %buf = memref.alloc() {alignment = 128 : i64} : memref<128x64xf32, #dsp.local>
  %tag = memref.alloc() : memref<1xi32>
  memref.dma_start %a[%c0, %k], %buf[%c0, %c0], %n, %tag[%c0], %c256, %c64
      : memref<128x256xf32>, memref<128x64xf32, #dsp.local>, memref<1xi32>
  memref.dma_wait %tag[%c0], %n : memref<1xi32>
  %v = memref.load %buf[%c0, %c0] : memref<128x64xf32, #dsp.local>
  memref.dealloc %tag : memref<1xi32>
  memref.dealloc %buf : memref<128x64xf32, #dsp.local>
  return %v : f32
}

// -----

// The destination may be one slot of a multi-buffer (a subview); the copy
// targets that slot.
// CHECK-LABEL: func.func @slot_dma
// CHECK:       %[[MB:.*]] = memref.alloc() : memref<2x64x64xf32>
// CHECK:       %[[SLOT:.*]] = memref.subview %[[MB]][%{{.*}}, 0, 0] [1, 64, 64] [1, 1, 1] : memref<2x64x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
// CHECK:       %[[SRC:.*]] = memref.subview %{{.*}}[%{{.*}}, 0] [64, 64] [1, 1]
// CHECK-NEXT:  linalg.copy ins(%[[SRC]] : {{.*}}) outs(%[[SLOT]] : memref<64x64xf32, strided<[64, 1], offset: ?>>)
func.func @slot_dma(%b: memref<256x64xf32>, %k: index, %i: index) -> f32 {
  %c0 = arith.constant 0 : index
  %n = arith.constant 4096 : index
  %mb = memref.alloc() : memref<2x64x64xf32, #dsp.local>
  %slot = memref.subview %mb[%i, 0, 0] [1, 64, 64] [1, 1, 1]
      : memref<2x64x64xf32, #dsp.local> to memref<64x64xf32, strided<[64, 1], offset: ?>, #dsp.local>
  %tag = memref.alloc() : memref<2x1xi32>
  %tslot = memref.subview %tag[%i, 0] [1, 1] [1, 1] : memref<2x1xi32> to memref<1xi32, strided<[1], offset: ?>>
  memref.dma_start %b[%k, %c0], %slot[%c0, %c0], %n, %tslot[%c0]
      : memref<256x64xf32>, memref<64x64xf32, strided<[64, 1], offset: ?>, #dsp.local>, memref<1xi32, strided<[1], offset: ?>>
  memref.dma_wait %tslot[%c0], %n : memref<1xi32, strided<[1], offset: ?>>
  %v = memref.load %slot[%c0, %c0] : memref<64x64xf32, strided<[64, 1], offset: ?>, #dsp.local>
  memref.dealloc %tag : memref<2x1xi32>
  memref.dealloc %mb : memref<2x64x64xf32, #dsp.local>
  return %v : f32
}

// -----

// Only whole-buffer DMAs, the form -nanodsp-promote-local emits, have a
// known tile shape.
func.func @partial_dma(%a: memref<128x256xf32>, %buf: memref<128x64xf32, #dsp.local>, %tag: memref<1xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = arith.constant 64 : index
  // expected-error @+1 {{'memref.dma_start' op can only be lowered when it fills a whole statically shaped destination of the source's rank, starting at index 0}}
  memref.dma_start %a[%c0, %c0], %buf[%c1, %c0], %n, %tag[%c0]
      : memref<128x256xf32>, memref<128x64xf32, #dsp.local>, memref<1xi32>
  return
}
