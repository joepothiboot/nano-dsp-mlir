// RUN: nanodsp-opt %s | nanodsp-opt | FileCheck %s

// CHECK-LABEL: func.func @local_buffer
// CHECK:       memref.alloc() : memref<2x64x32xf32, #dsp.local>
// CHECK:       memref.subview {{.*}} : memref<2x64x32xf32, #dsp.local> to memref<64x32xf32, strided<[32, 1], offset: ?>, #dsp.local>
func.func @local_buffer(%i: index) -> memref<64x32xf32, strided<[32, 1], offset: ?>, #dsp.local> {
  %buf = memref.alloc() : memref<2x64x32xf32, #dsp.local>
  %s = memref.subview %buf[%i, 0, 0] [1, 64, 32] [1, 1, 1]
      : memref<2x64x32xf32, #dsp.local> to memref<64x32xf32, strided<[32, 1], offset: ?>, #dsp.local>
  return %s : memref<64x32xf32, strided<[32, 1], offset: ?>, #dsp.local>
}
