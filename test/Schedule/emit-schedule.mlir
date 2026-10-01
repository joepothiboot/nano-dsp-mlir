// Tile sizes come from the TargetModel: the same ops get different schedules
// for different targets.
//
// RUN: nanodsp-opt %s -convert-dsp-to-linalg \
// RUN:     -nanodsp-emit-schedule=target=host-neon \
// RUN: | FileCheck %s --check-prefixes=CHECK,NEON
// RUN: nanodsp-opt %s -convert-dsp-to-linalg \
// RUN:     -nanodsp-emit-schedule=target=x86-avx2 \
// RUN: | FileCheck %s --check-prefixes=CHECK,AVX2
// RUN: nanodsp-opt %s -convert-dsp-to-linalg -nanodsp-emit-schedule=target=host-neon \
// RUN: | FileCheck %s --check-prefix=NOLOCAL
// RUN: nanodsp-opt %s -convert-dsp-to-linalg -nanodsp-emit-schedule=target=x86-avx2 \
// RUN: | FileCheck %s --check-prefix=NOLOCAL
// RUN: nanodsp-opt %s -convert-dsp-to-linalg \
// RUN:     -nanodsp-emit-schedule=target=hexagon-hvx128 \
// RUN: | FileCheck %s --check-prefix=HEX

// CHECK-LABEL: func.func @matmul
// NEON: linalg.generic {{.*}}nanodsp.cache_tile = array<i64: 64, 96, 64>, nanodsp.loop_ranges = array<i64: 128, 96, 256>, nanodsp.reg_tile = array<i64: 4, 16, 1>, nanodsp.tag = "op0", nanodsp.working_set_bytes = 65536 : i64
// AVX2: linalg.generic {{.*}}nanodsp.cache_tile = array<i64: 32, 96, 8>, nanodsp.loop_ranges = array<i64: 128, 96, 256>, nanodsp.reg_tile = array<i64: 4, 24, 1>, nanodsp.tag = "op0", nanodsp.working_set_bytes = 16384 : i64
func.func @matmul(%a: tensor<128x256xf32>, %b: tensor<256x96xf32>) -> tensor<128x96xf32> {
  %r = dsp.matmul %a, %b : (tensor<128x256xf32>, tensor<256x96xf32>) -> tensor<128x96xf32>
  return %r : tensor<128x96xf32>
}

// CHECK-LABEL: func.func @conv
// CHECK: linalg.generic {{.*}}nanodsp.tag = "op1"
func.func @conv(%i: tensor<1x10x10x3xf32>, %f: tensor<3x3x3x8xf32>) -> tensor<1x8x8x8xf32> {
  %r = dsp.conv2d %i, %f : (tensor<1x10x10x3xf32>, tensor<3x3x3x8xf32>) -> tensor<1x8x8x8xf32>
  return %r : tensor<1x8x8x8xf32>
}

// CHECK-LABEL: func.func @conv_strided
// CHECK: linalg.generic {{.*}}nanodsp.tag = "op2"
func.func @conv_strided(%i: tensor<1x9x9x3xf32>, %f: tensor<3x3x3x8xf32>) -> tensor<1x4x4x8xf32> {
  %r = dsp.conv2d %i, %f {strides = array<i64: 2, 2>}
      : (tensor<1x9x9x3xf32>, tensor<3x3x3x8xf32>) -> tensor<1x4x4x8xf32>
  return %r : tensor<1x4x4x8xf32>
}

// CHECK-LABEL: func.func @relu
// CHECK: linalg.generic {{.*}}nanodsp.tag = "op3"
func.func @relu(%a: tensor<6x20xf32>) -> tensor<6x20xf32> {
  %r = dsp.relu %a : tensor<6x20xf32>
  return %r : tensor<6x20xf32>
}

// CHECK: module attributes {transform.with_named_sequence}
// CHECK: transform.named_sequence @__transform_main

// Matmul (m, n, k): cache tile first, then the register tile. The register
// tile's reduction size is always 1 (bit-exact accumulation order).
//  NEON, 32 regs x 4 lanes: 4x16 register tile, 64x96x64 cache tile (64 KiB).
//  AVX2, 16 regs x 8 lanes: 4x24 register tile, 32x96x8 cache tile (16 KiB).
// CHECK:      %[[MM:[^ ]+]] = transform.structured.match ops{["linalg.generic"]} attributes {nanodsp.tag = "op0"}
// NEON-NEXT:  %[[MMC:[^ ,]+]], %{{[^ ]+}}:2 = transform.structured.tile_using_for %[[MM]] tile_sizes [64, 0, 64]
// NEON-NEXT:  %{{[^ ,]+}}, %[[MML:[^ :]+]]:3 = transform.structured.tile_using_for %[[MMC]] tile_sizes [4, 16, 1]
// AVX2-NEXT:  %[[MMC:[^ ,]+]], %{{[^ ]+}}:2 = transform.structured.tile_using_for %[[MM]] tile_sizes [32, 0, 8]
// AVX2-NEXT:  %{{[^ ,]+}}, %[[MML:[^ :]+]]:3 = transform.structured.tile_using_for %[[MMC]] tile_sizes [4, 24, 1]

// Conv (n, oh, ow, f, kh, kw, c): ow x f register tile, every reduction dim 1.
// CHECK:      %[[CV:[^ ]+]] = transform.structured.match {{.*}}nanodsp.tag = "op1"
// CHECK-NEXT: %[[CVC:[^ ,]+]], %{{[^ ]+}} = transform.structured.tile_using_for %[[CV]] tile_sizes [0, 1, 0, 0, 0, 0, 0]
// CHECK-NEXT: %{{[^ ,]+}}, %[[CVL:[^ :]+]]:4 = transform.structured.tile_using_for %[[CVC]] tile_sizes [0, 0, 4, 0, 1, 1, 1]

// Strided conv: ow is read with stride 2, so it gets no register blocking.
// CHECK:      transform.structured.match {{.*}}nanodsp.tag = "op2"
// CHECK-NEXT: tile_using_for
// CHECK-NEXT: tile_using_for %{{.+}} tile_sizes [0, 0, 1, 0, 1, 1, 1]

// Elementwise: no reuse, so no cache tile; register tile only.
// CHECK:      %[[EW:[^ ]+]] = transform.structured.match {{.*}}nanodsp.tag = "op3"
// NEON-NEXT:  %{{[^ ,]+}}, %[[EWL:[^ :]+]]:2 = transform.structured.tile_using_for %[[EW]] tile_sizes [1, 4]
// AVX2-NEXT:  %{{[^ ,]+}}, %[[EWL:[^ ]+]] = transform.structured.tile_using_for %[[EW]] tile_sizes [1, 0]

// Unit dims of every register tile are folded first (a matmul tile's k, a
// conv tile's n/oh/kh/kw/c), then each tile is found again through its
// innermost loop, since folding replaces the op and drops its tag.
// CHECK:      transform.apply_patterns to
// CHECK-NEXT:   transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
// CHECK:      %[[MMF:[^ ]+]] = transform.structured.match ops{["linalg.generic"]} in %[[MML]]#2
// CHECK-NEXT: transform.structured.vectorize %[[MMF]]
// CHECK:      %[[CVF:[^ ]+]] = transform.structured.match ops{["linalg.generic"]} in %[[CVL]]#3
// CHECK-NEXT: transform.structured.vectorize %[[CVF]]
// CHECK:      transform.structured.match ops{["linalg.generic"]} in
// CHECK-NEXT: transform.structured.vectorize
// CHECK:      %[[EWF:[^ ]+]] = transform.structured.match ops{["linalg.generic"]} in %[[EWL]]
// CHECK-NEXT: transform.structured.vectorize %[[EWF]]

// Only a target with local memory (hexagon-hvx128: 256 KiB VTCM) marks the
// innermost cache-tile loop for -nanodsp-promote-local; ops without a cache
// tile (relu) get no marker.
// NOLOCAL:     transform.named_sequence @__transform_main
// NOLOCAL-NOT: transform.annotate
// HEX:      %{{[^ ,]+}}, %[[L0:[^ :]+]] = transform.structured.tile_using_for {{.*}} tile_sizes [0, 0, 128]
// HEX-NEXT: transform.annotate %[[L0]] "nanodsp.cache_loop" : !transform.any_op
// HEX:      %{{[^ ,]+}}, %[[L1:[^ :]+]] = transform.structured.tile_using_for {{.*}} tile_sizes [0, 1, 0, 0, 0, 0, 0]
// HEX-NEXT: transform.annotate %[[L1]] "nanodsp.cache_loop" : !transform.any_op
// HEX:      %{{[^ ,]+}}, %[[L2:[^ :]+]] = transform.structured.tile_using_for {{.*}} tile_sizes [0, 1, 0, 0, 0, 0, 0]
// HEX-NEXT: transform.annotate %[[L2]] "nanodsp.cache_loop" : !transform.any_op
// HEX-NOT:  transform.annotate
