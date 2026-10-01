# Mojo tensor API: traits, views and compile-time tiles 🧩

`mojo/nanodsp/` started as four kernels over one owned `Tensor` type. This
note covers the layer added on top of it: a `TensorLike` trait, a borrowed
`TensorView`, and `matmul_tiled`, one generic kernel whose operand types,
tile shape and SIMD width are all fixed at compile time. Results stay bit
for bit equal to the untiled `matmul` and to the scalar C++ reference.

```mojo
from nanodsp import Tensor, TensorView, matmul_tiled

var a = Tensor[DType.float32]([64, 32], fill=1.0)
var b = Tensor[DType.float32]([32, 48], fill=0.5)
var big = Tensor[DType.float32]([128, 128])

# Write a 64 x 48 product into the middle of a larger buffer.
var c = big.view().tile(16, 40, 64, 48)
matmul_tiled[DType.float32, 4, 16](a, b.view(), c)
```

The pieces live in `mojo/nanodsp/layout.mojo` (trait, view) and
`mojo/nanodsp/kernels.mojo` (kernel); the tests are in
`mojo/tests/test_layout.mojo`.

## 🧱 The API

```mojo
trait TensorLike:
    comptime element_dtype: DType
    def rows(self) -> Int
    def cols(self) -> Int
    def load[width: Int](self, i: Int, j: Int) -> SIMD[Self.element_dtype, width]
    def store[width: Int](mut self, i: Int, j: Int, v: SIMD[Self.element_dtype, width])

struct TensorView[mut: Bool, //, dtype: DType, origin: Origin[mut=mut]](
    ImplicitlyCopyable, TensorLike
):
    def __init__(out self, span: Span[Scalar[dtype], origin], rows: Int, cols: Int) raises
    def __init__(out self, span: Span[Scalar[dtype], origin], rows: Int, cols: Int, row_stride: Int) raises
    def row_stride(self) -> Int
    def tile(self, i0: Int, j0: Int, h: Int, w: Int) raises -> Self
    def readonly(self) -> TensorView[dtype, Origin[mut=False](origin)]
    def __getitem__(self, i: Int, j: Int) -> Scalar[dtype]

struct Tensor[dtype: DType](Copyable, Movable, TensorLike):
    def view(ref self) raises -> TensorView[dtype, origin_of(self.data)]

def matmul_tiled[
    A: TensorLike, B: TensorLike, C: TensorLike, //,
    dtype: DType, tile_m: Int, tile_n: Int,
    width: Int = simd_width_of[dtype](),
](a: A, b: B, mut c: C) raises
```

The existing `add`, `relu`, `matmul`, `conv2d` and `qmatmul` are unchanged.

## 🧩 Trait or concrete type

The kernels before this change took `Tensor[dtype]` and reached into its
`List`. That is the right call for a tensor that is always whole and
contiguous, but it means a kernel cannot write into part of a buffer, or
read a matrix stored with padded rows, without a copy.

`TensorLike` is the smallest interface a dense rank-2 kernel needs: the shape
and SIMD loads and stores at `(row, col)`. Only the innermost dimension is
assumed contiguous, which is what lets one interface cover both a `Tensor`
(row stride = `cols`) and a tile of a larger buffer (row stride = the
parent's width).

Mojo traits are resolved at compile time. `matmul_tiled` with `A = Tensor`
and with `A = TensorView` are two separate instantiations, each with its
`load` inlined. There is no vtable and no boxing, so the generic version
costs nothing over a hand-specialized one (see the benchmark below, where
the view instantiation is in fact the faster one).

A concrete type still has a place. `Tensor` owns its storage and is what the
other kernels, the tests and the benchmark construct. The trait is for code
that should not care where the elements live.

Two details of the trait are shaped by Mojo 1.1:

- **The element type is `element_dtype`, not `dtype`.** A struct parameter
  does not satisfy a trait's `comptime` requirement, and a struct cannot
  declare `comptime dtype` next to a parameter named `dtype`
  ("invalid redefinition"). Renaming the trait member keeps `Tensor[dtype]`
  source-compatible.
- **There is one trait, not a read trait and a write trait.** Whether a view
  can write depends on its origin, a type parameter, and Mojo 1.1 has no
  conditional conformance: a `where Self.mut` clause on `store` is rejected
  when checking the trait ("lacking evidence to prove correctness").
  Instead `store` is always declared and refuses to compile for a read-only
  origin (next section).

## 📐 Views, origins and ownership

A `TensorView` is four words: a pointer, rows, cols and a row stride.
Copying one never copies elements, and `tile()` returns a new view into the
same buffer, keeping the parent's stride. What makes it safe is the second
parameter, `origin`.

`Tensor.view(ref self)` returns `TensorView[dtype, origin_of(self.data)]`.
The origin records which variable the pointer was derived from and whether
that access was mutable, and the compiler uses it in three ways. The
listings are real output from `mojo run --fp-mode contract=off -I mojo
<file>`, with the scratch directory trimmed from the paths.

**The buffer stays put while the view is in use.** Moving the tensor ends
its lifetime, and a later use of the view counts as a use of the tensor:

```mojo
var t = Tensor[DType.float32]([2, 2], fill=1.0)
var v = t.view()
var owner = t^  # move the buffer while `v` still points into it
print(v[0, 0], owner.numel())
```

```text
use_after_move.mojo:7:12: error: use of uninitialized value 't'
    print(v[0, 0], owner.numel())
           ^
use_after_move.mojo:4:9: note: 't' declared here
    var t = Tensor[DType.float32]([2, 2], fill=1.0)
        ^
mojo: error: failed to run the pass manager
```

**Mutability follows the borrow.** `view(ref self)` infers `mut` from how
`self` was accessed. A `var` gives a mutable view; a read-only argument
gives a read-only one, and `store` on it fails at compile time through a
`comptime assert Self.mut`:

```mojo
def zero_first(t: Tensor[DType.float32]) raises:
    var v = t.view()  # `t` is a read-only borrow, so `v` is read-only too
    v.store[1](0, 0, 0.0)
```

The error is a chain of instantiation notes that starts with
`error: function instantiation failed` at `main` and ends at the assertion.
Shown here are the call site and the end of the chain:

```text
readonly_store.mojo:5:15: note: call expansion failed with parameter value(s): ("mut": False, ..., "width": 1)
    v.store[1](0, 0, 0.0)
              ^
...
layout.mojo:130:9: note: constraint failed: store through a read-only TensorView
        comptime assert Self.mut, "store through a read-only TensorView"
        ^
mojo: error: failed to run the pass manager
```

**A view counts for argument exclusivity.** Mojo rejects a call that passes
the same memory both mutably and immutably. Because the view's type carries
`origin_of(t.data)`, that check sees through it, so an in-place
`c = t * t` is caught:

```mojo
var t = Tensor[DType.float32]([8, 8], fill=1.0)
var c = t.view()
matmul_tiled[DType.float32, 4, 8](t, t, c)  # c = t * t, in place
```

```text
output_aliases_input.mojo:6:38: error: aliasing values passed immutably to 'a' argument and passed mutably to 'c' argument in 'matmul_tiled' call
    matmul_tiled[DType.float32, 4, 8](t, t, c)  # c = t * t, in place
                                     ^~     ~
output_aliases_input.mojo:6:38: note: 'origin_of(t.data)' memory accessed through reference embedded in value of type 'TensorView[.float32, origin_of(t.data)]'
    matmul_tiled[DType.float32, 4, 8](t, t, c)  # c = t * t, in place
                                     ^
mojo: error: failed to parse the provided Mojo source module
```

That check is coarse: it works per variable, not per element range. Two
disjoint tiles of one tensor share `origin_of(t.data)`, so reading one tile
while writing another in the same call is rejected as well, and
`readonly()` (which demotes a view's origin to immutable) does not change
that, since the conflict is between an immutable and a mutable use of the
same origin. `readonly()` is for the case where every use is a read: the
tests pass `t.view().readonly()` next to `t` to compare a view against its
own tensor, which the mutable view alone would not be allowed to do.
Writing one tile of a buffer from another tile of the same buffer needs a
copy today, or an API that splits a buffer into provably disjoint origins,
which this library does not have.

What origins do not cover: a view built from a `Span` checks its shape and
stride against the span's length once, in the constructor, but `load` and
`store` are unchecked, like the pointer accesses in the other kernels.
`tile()` checks its bounds and raises.

## ⚙️ Compile-time specialization

`matmul_tiled` has four compile-time parameters the caller sets and three
it infers:

| Parameter         | Set by                      | Used for                                       |
| ----------------- | --------------------------- | ---------------------------------------------- |
| `A`, `B`, `C`     | inferred from the arguments | which `load`/`store` gets inlined              |
| `dtype`           | caller                      | `comptime assert A.element_dtype == dtype` ... |
| `tile_m`,`tile_n` | caller                      | register tile shape, unrolled                  |
| `width`           | caller, default native SIMD | lanes per accumulator                          |

A full `tile_m x tile_n` block runs a micro-kernel whose loops over the tile
are `comptime for`, so they are unrolled away: `tile_m * tile_n / width`
accumulators of type `SIMD[dtype, width]` live in a fixed-size `Array`
with only compile-time indices, which LLVM can promote to registers. Each step over `k` loads `tile_n / width` vectors of a
row of `b`, broadcasts one element of `a` per tile row, and does
`acc = acc + a_ik * b_kj` per accumulator. Partial blocks at the bottom and
right edges go through a runtime-sized path, one row at a time, with a
scalar tail for the last `N mod width` columns. The constraints are checked
when the kernel is instantiated, not at run time: `tile_n % width == 0`,
positive tile sizes, and all three operands having element type `dtype`.

Because the tile shape is a parameter, trying a new one is a one-token
change. One run of each config on the M2 (`W = 4` f32 lanes on NEON),
`Tensor` operands:

| `tile_m x tile_n` | 256³ GFLOP/s | 512³ GFLOP/s |
| ----------------- | ------------ | ------------ |
| 1 x 4             | 11.4         | 10.1         |
| 2 x 16            | 47.7         | 48.0         |
| 4 x 8             | 42.6         | 43.2         |
| 4 x 16            | 49.0         | 47.9         |
| 6 x 16            | 45.8         | 48.3         |
| 8 x 8             | 44.2         | 44.3         |
| 8 x 12            | 36.3         | 40.5         |

A 1 x 4 tile is one accumulator with no reuse and is slower than the
untiled `matmul`; anything from 2 x 16 to 6 x 16 lands within a few percent
of the others. The benchmark uses 4 x 16.

## 🎯 Why only i and j are tiled

Floating-point addition is not associative, so a kernel can match another
bit for bit only if every output element is built from the same products,
added in the same order, each rounded the same way. For matmul that order
is fixed by the C++ reference: `acc = 0`, then `acc = acc + a[i,k] * b[k,j]`
for increasing `k`.

Tiling i and j changes which outputs are computed together, not how any one
of them is computed, so it is free. The reduction is a different matter.
This experiment (`--fp-mode contract=off`, f32, 64 x 256 x 64, inputs that
make products and sums round) compares three ways of tiling `k` against
`matmul`:

```text
elements: 4096
k blocked by 32, one accumulator:  0 differ
k split in 2 (split-K):            3733 differ
k register tile of 4 lanes:        3757 differ
```

Blocking `k` for the cache is harmless as long as one running accumulator
carries across the blocks in order: the sequence of additions is unchanged.
What breaks bit-exactness is giving `k` its own independent partial sums,
either across threads (split-K) or across SIMD lanes or registers (a
register tile with a reduction extent above 1, which is what a dot-product
instruction or a `vector.contract` lowering does). Those sums are combined
at the end in a different association, and about 91% of the outputs
change in their last bits. That is the same rule the MLIR schedules follow (register
tiles keep the reduction dims at 1; see `docs/05-soundness.md`), so
`matmul_tiled` tiles i and j only and walks all of `k` inside each block.

### Mojo fuses multiply-add unless told not to

The order is necessary but not sufficient: the multiply and the add must
also round separately. Mojo 1.1 defaults to `--fp-mode contract=fast`, which
`mojo build --help` describes as like Clang's `-ffp-contract=fast`: it
"fuses `a + b*c` into an FMA across statements". Kernel source that writes
`acc + a * b` therefore does not get a separately rounded multiply. On a
7 x 13 x 19 f32 matmul with inexact inputs, the default build differed from
an unfused reference in 92 of 133 elements, and matched it in all of them
with `--fp-mode contract=off`.

The tests that existed before this change used small integers, where an FMA
and a separate multiply and add give the same answer, so they could not see
this. The pixi tasks now pass `--fp-mode contract=off` (the counterpart of
the C++ reference's `-ffp-contract=off`), and `test_layout.mojo` guards it
two ways:

- `test_build_does_not_contract` computes `x * x - y` with
  `x = 1 + 2^-12` and `y = 1 + 2^-11` read through `argv` so the expression
  cannot be constant-folded (folding happens before contraction and would
  hide it). Unfused, that is 0; fused, it is `2^-24`.
- `unfused_matmul_f32` is a reference that cannot be fused: each f32 product
  is computed exactly in f64, offset by an opaque zero so LLVM cannot shrink
  it back to an f32 multiply, then rounded to f32 before the add.

Run without the flag, the canary fails first:

```text
Unhandled exception caught during execution: At .../mojo/tests/test_layout.mojo:126:17: AssertionError: `left == right` comparison failed:
   left: 5.9604645e-08
  right: 0.0
  reason: mul and add were fused into an FMA: build with --fp-mode contract=off (see pixi.toml)
```

and with the canary removed, the unfused reference catches it on the first
config:

```text
  reason: tile 1x4 w4 on 7x13x19 vs unfused at (0, 1): -3.4088433 vs -3.4088428
```

As a check that the tests can tell the two apart, replacing the
micro-kernel's `acc + a * b` with an explicit `fma` fails them under
`contract=off`, on the first config:

```text
  reason: tile 1x4 w4 on 7x13x19 vs matmul at (0, 1): -3.4088433 vs -3.4088428
```

## 📊 Performance

`pixi run bench` on an Apple M2, one core, best of 5 (one run):

| Kernel                          | 256³ | 512³ |
| ------------------------------- | ---- | ---- |
| `matmul` (untiled, i-k-j)       | 20.6 | 17.8 |
| `matmul_tiled` 1 x 4            | 10.8 | 9.7  |
| `matmul_tiled` 4 x 16, `Tensor` | 46.5 | 46.9 |
| `matmul_tiled` 4 x 16, views    | 50.9 | 51.0 |

(GFLOP/s.) The register tile roughly doubles throughput and, unlike the
untiled kernel, does not drop at 512. The view instantiation was faster
than the `Tensor` one in each of six runs, by 5 to 13%: `Tensor.load` computes
`i * self.shape[1] + j`, reading the row length out of a `List` on every
access, while a view holds its stride in a plain field.

## 🧪 Tests

`pixi run test-mojo` runs `test-kernels` (the original file) and
`test-layout` (`mojo/tests/test_layout.mojo`, 11 tests):

- **Views:** shape and stride of `Tensor.view()`, tiles keeping the parent's
  stride (including a tile of a tile), writes through a mutable tile, a view
  over a padded `Span`, and the bounds and rank checks.
- **Tile configs:** six compile-time configs (1 x W, 4 x 2W, 3 x 3W, 8 x 4
  with width 4, 2 x 16 with width 8, 5 x 3 with width 1), each on 7x13x19,
  16x9x32, 33x17x37, 1x1x1 and 3x40x2, so full tiles, edge tiles and scalar
  tails all run. f64 with the default width as well.
- **Operand types:** the same product with `Tensor`, `TensorView` and mixed
  operands; and A, B, C as tiles inside larger buffers, checking that no
  element outside the destination tile is written.

Every comparison is on bit patterns (`to_bits()`), against `matmul`, the
naive i-j-k loop and the unfused reference.

## ⚠️ Mojo 1.1 notes

Found while writing this, all confirmed by compiling:

- `UnsafePointer` is deprecated in favor of `Pointer`, which carries an
  origin. Writing through it needs a mutable origin; a view generic over
  mutability casts with `unsafe_mut_cast[True]()` after
  `comptime assert Self.mut`.
- `@parameter for` and `@parameter if` are gone: use `comptime for` and
  `comptime if`.
- `Origin[mut=False](o)` converts an origin to its read-only form;
  `ImmutOrigin` and `ImmutableOrigin` do not exist.
- Infer-only parameters (`A: TensorLike, //`) must come first; a required
  parameter cannot follow one with a default.
- `ref` is a keyword, so it cannot be a variable name.
- `List(length=n)` needs `fill=`.

## 🔜 Next

- A rank-N layout (shape and strides as compile-time or runtime tuples)
  instead of the rank-2 trait.
- Cache blocking over `k` (in order, one accumulator) and over `j`, with
  tile sizes from `docs/02-tiling-model.md`.
- `conv2d` on views, so a convolution can write into a padded buffer.
