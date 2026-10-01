# Tests for the TensorLike / TensorView API and the generic `matmul_tiled`.
#
#   - views: shape, row stride, tiles, write-through, bounds checks.
#   - matmul_tiled: the same kernel instantiated for Tensor and TensorView
#     operands, several compile-time tile shapes and SIMD widths, odd sizes
#     (so the edge tiles and scalar tails run), and views into larger
#     buffers (row stride != cols).
#
# Every matmul check is exact equality against the existing `matmul` and a
# naive i-j-k loop. The inputs are deliberately not small integers: products
# and partial sums round, so a reordered reduction or a fused multiply-add
# would show up as a mismatch.
#
# Mojo contracts `a + b * c` into an FMA by default (`--fp-mode
# contract=fast`), even across statements, so the pixi tasks pass
# `--fp-mode contract=off`, the counterpart of the C++ reference's
# `-ffp-contract=off`. `test_build_does_not_contract` fails loudly when that
# flag is missing, and the f32 reference below is unfused by construction.
#
# Run from the repo root:  pixi run test-layout  (or: pixi run test-mojo)

from std.sys import argv, simd_width_of
from std.testing import assert_equal, assert_raises, assert_true

from nanodsp import Tensor, TensorLike, TensorView, matmul, matmul_tiled

comptime F32 = DType.float32
comptime F64 = DType.float64
comptime W = simd_width_of[F32]()


def inexact[dtype: DType](var shape: List[Int], seed: Int) -> Tensor[dtype]:
    """Values like -2.2059999 that make f32 products and sums round."""
    var t = Tensor[dtype](shape^)
    for i in range(t.numel()):
        t.data[i] = Scalar[dtype]((i * 7 + seed) % 13 - 6) * Scalar[dtype](
            0.37
        ) + Scalar[dtype](0.011) * Scalar[dtype]((i * 3 + seed) % 5)
    return t^


def naive_matmul[dtype: DType](a: Tensor[dtype], b: Tensor[dtype]) -> Tensor[dtype]:
    """The C++ reference's loop: i-j-k, one scalar accumulator."""
    var m = a.shape[0]
    var k = a.shape[1]
    var n = b.shape[1]
    var r = Tensor[dtype]([m, n])
    for i in range(m):
        for j in range(n):
            var acc: Scalar[dtype] = 0
            for kk in range(k):
                acc += a[i * k + kk] * b[kk * n + j]
            r.data[i * n + j] = acc
    return r^


def opaque_zero() -> Float64:
    """0.0, computed from argv so the optimizer cannot fold it away."""
    return Float64(len(argv()) - 1) * 0.0


def unfused_matmul_f32(a: Tensor[F32], b: Tensor[F32]) -> Tensor[F32]:
    """The i-j-k loop with each product rounded on its own, whatever the flags.

    An f32 x f32 product is exact in f64 (48 <= 53 significand bits), so
    rounding it to f32 gives exactly the IEEE f32 product. Adding an opaque
    zero in f64 first stops LLVM from shrinking the f64 multiply back to an
    f32 one, so no f32 `fmul` ever feeds the f32 `fadd` and there is nothing
    to fuse, even under `--fp-mode contract=fast`. This is the C++
    reference's arithmetic.
    """
    var zero = opaque_zero()
    var m = a.shape[0]
    var k = a.shape[1]
    var n = b.shape[1]
    var r = Tensor[F32]([m, n])
    for i in range(m):
        for j in range(n):
            var acc: Float32 = 0
            for kk in range(k):
                var p = (
                    a[i * k + kk].cast[F64]() * b[kk * n + j].cast[F64]() + zero
                ).cast[F32]()
                acc = acc + p
            r.data[i * n + j] = acc
    return r^


def assert_bit_exact[
    X: TensorLike, Y: TensorLike
](actual: X, expected: Y, what: String) raises:
    assert_equal(actual.rows(), expected.rows(), msg=what + ": rows")
    assert_equal(actual.cols(), expected.cols(), msg=what + ": cols")
    for i in range(expected.rows()):
        for j in range(expected.cols()):
            var x = actual.load[1](i, j)
            var y = rebind[Scalar[X.element_dtype]](expected.load[1](i, j))
            # Compare bit patterns, not values: also catches -0.0 vs +0.0.
            assert_equal(
                x.to_bits(),
                y.to_bits(),
                msg=what
                + " at ("
                + String(i)
                + ", "
                + String(j)
                + "): "
                + String(x)
                + " vs "
                + String(y),
            )


# --- build mode -------------------------------------------------------------


def test_build_does_not_contract() raises:
    # x = 1 + 2^-12: x*x = 1 + 2^-11 + 2^-24 rounds to 1 + 2^-11 in f32, so
    # the unfused x*x - (1 + 2^-11) is 0, while an FMA keeps the 2^-24.
    # The inputs come from argv so the expression is not constant-folded
    # (folding happens before contraction and would hide it).
    var one = Float32(1.0) + opaque_zero().cast[F32]()
    var x = one + Float32(2.0) ** -12
    var y = one + Float32(2.0) ** -11
    var r = x * x - y
    assert_equal(
        r,
        0.0,
        msg=(
            "mul and add were fused into an FMA: build with --fp-mode"
            " contract=off (see pixi.toml)"
        ),
    )


# --- views ------------------------------------------------------------------


def test_view_of_tensor() raises:
    var t = inexact[F32]([5, 8], 1)
    var v = t.view()
    assert_equal(v.rows(), 5)
    assert_equal(v.cols(), 8)
    assert_equal(v.row_stride(), 8)
    # `v` borrows `t` mutably, so passing both to one call is rejected by
    # the exclusivity check; a read-only copy of the view is accepted.
    assert_bit_exact(v.readonly(), t, "view")


def test_tile_keeps_parent_stride() raises:
    var t = inexact[F32]([5, 8], 2)
    var tl = t.view().tile(1, 2, 3, 4)
    assert_equal(tl.rows(), 3)
    assert_equal(tl.cols(), 4)
    assert_equal(tl.row_stride(), 8)
    for i in range(3):
        for j in range(4):
            assert_equal(tl[i, j], t[(1 + i) * 8 + (2 + j)])
    # A tile of a tile is still addressed in the root buffer.
    var inner = tl.tile(1, 1, 2, 2)
    assert_equal(inner[1, 1], t[3 * 8 + 4])  # (1+1+1, 2+1+1)


def test_mutable_view_writes_through() raises:
    var t = Tensor[F32]([4, 6], fill=0.0)
    var tl = t.view().tile(1, 1, 2, 4)
    tl.store[4](1, 0, SIMD[F32, 4](1.0, 2.0, 3.0, 4.0))
    assert_equal(t[2 * 6 + 1], 1.0)
    assert_equal(t[2 * 6 + 4], 4.0)
    assert_equal(t[2 * 6 + 5], 0.0)  # just past the tile: untouched


def test_view_over_span() raises:
    # A 3 x 4 matrix stored with a padded row stride of 6.
    var buf = List[Float32](length=18, fill=-1.0)
    var v = TensorView(Span(buf), 3, 4, 6)
    v.store[1](2, 3, 7.0)
    assert_equal(buf[2 * 6 + 3], 7.0)
    assert_equal(v.row_stride(), 6)


def test_view_bounds() raises:
    var t = Tensor[F32]([4, 6])
    with assert_raises(contains="out of bounds"):
        _ = t.view().tile(2, 0, 3, 6)
    with assert_raises(contains="out of bounds"):
        _ = t.view().tile(0, 5, 1, 2)
    var buf = List[Float32](length=10, fill=0.0)
    with assert_raises(contains="overrun"):
        _ = TensorView(Span(buf), 3, 4, 4)
    with assert_raises(contains="row_stride"):
        _ = TensorView(Span(buf), 2, 4, 3)
    with assert_raises(contains="rank 2"):
        _ = Tensor[F32]([2, 3, 4]).view()


# --- matmul_tiled on tensors ------------------------------------------------


def check_tensor_config[
    tile_m: Int, tile_n: Int, width: Int
](m: Int, k: Int, n: Int) raises:
    var a = inexact[F32]([m, k], 3)
    var b = inexact[F32]([k, n], 5)
    var c = Tensor[F32]([m, n], fill=99.0)  # overwritten, not accumulated
    matmul_tiled[F32, tile_m, tile_n, width](a, b, c)
    var what = (
        "tile "
        + String(tile_m)
        + "x"
        + String(tile_n)
        + " w"
        + String(width)
        + " on "
        + String(m)
        + "x"
        + String(k)
        + "x"
        + String(n)
    )
    assert_bit_exact(c, matmul(a, b), what + " vs matmul")
    assert_bit_exact(c, naive_matmul(a, b), what + " vs naive")
    assert_bit_exact(c, unfused_matmul_f32(a, b), what + " vs unfused")


def check_shapes[tile_m: Int, tile_n: Int, width: Int]() raises:
    # Odd sizes put edge tiles on both sides and a scalar column tail;
    # 16 x 32 is a whole number of tiles for most configs.
    check_tensor_config[tile_m, tile_n, width](7, 13, 19)
    check_tensor_config[tile_m, tile_n, width](16, 9, 32)
    check_tensor_config[tile_m, tile_n, width](33, 17, 37)
    check_tensor_config[tile_m, tile_n, width](1, 1, 1)
    check_tensor_config[tile_m, tile_n, width](3, 40, 2)


def test_tiled_configs_on_tensors() raises:
    check_shapes[1, W, W]()  # one vector per tile
    check_shapes[4, 2 * W, W]()  # the default-width register tile
    check_shapes[3, 3 * W, W]()  # tile sizes that divide nothing
    check_shapes[8, 4, 4]()  # explicit 4-lane width
    check_shapes[2, 16, 8]()  # wider than native: LLVM splits the vectors
    check_shapes[5, 3, 1]()  # width 1: scalar register tile


def test_tiled_default_width_f64() raises:
    var a = inexact[F64]([9, 11], 1)
    var b = inexact[F64]([11, 21], 2)
    var c = Tensor[F64]([9, 21])
    matmul_tiled[F64, 4, 2 * simd_width_of[F64]()](a, b, c)
    assert_bit_exact(c, matmul(a, b), "f64 vs matmul")
    assert_bit_exact(c, naive_matmul(a, b), "f64 vs naive")


# --- matmul_tiled on views --------------------------------------------------


def test_tiled_on_views_into_larger_buffers() raises:
    comptime m = 7
    comptime k = 13
    comptime n = 19
    # Each operand lives inside a bigger buffer, so row_stride != cols.
    var big_a = inexact[F32]([20, 31], 3)
    var big_b = inexact[F32]([17, 40], 5)
    var big_c = Tensor[F32]([12, 27], fill=-5.0)
    var av = big_a.view().tile(2, 3, m, k)
    var bv = big_b.view().tile(1, 4, k, n)
    var cv = big_c.view().tile(3, 5, m, n)
    matmul_tiled[F32, 4, 2 * W](av, bv, cv)

    # Expected: the same operands copied out into contiguous tensors.
    var a = Tensor[F32]([m, k])
    var b = Tensor[F32]([k, n])
    for i in range(m):
        for j in range(k):
            a.data[i * k + j] = av[i, j]
    for i in range(k):
        for j in range(n):
            b.data[i * n + j] = bv[i, j]
    var expected = matmul(a, b)
    assert_bit_exact(big_c.view().tile(3, 5, m, n), expected, "view result")
    assert_bit_exact(
        big_c.view().tile(3, 5, m, n), unfused_matmul_f32(a, b), "vs unfused"
    )

    # Nothing outside the destination tile was written.
    for i in range(12):
        for j in range(27):
            if i < 3 or i >= 3 + m or j < 5 or j >= 5 + n:
                assert_equal(big_c[i * 27 + j], -5.0, msg="clobbered")


def test_same_kernel_tensor_and_view() raises:
    # One instantiation per operand-type combination; all must agree.
    var a = inexact[F32]([11, 10], 7)
    var b = inexact[F32]([10, 23], 8)
    var c1 = Tensor[F32]([11, 23])
    var c2 = Tensor[F32]([11, 23])
    var c3 = Tensor[F32]([11, 23])
    matmul_tiled[F32, 4, 2 * W](a, b, c1)  # Tensor, Tensor -> Tensor
    var c2v = c2.view()
    matmul_tiled[F32, 4, 2 * W](a.view(), b.view(), c2v)  # views throughout
    matmul_tiled[F32, 4, 2 * W](a, b.view(), c3)  # mixed
    assert_bit_exact(c1, matmul(a, b), "Tensor operands")
    assert_bit_exact(c2, c1, "view operands")
    assert_bit_exact(c3, c1, "mixed operands")


def test_tiled_rejects_mismatch() raises:
    var a = Tensor[F32]([2, 3])
    var b = Tensor[F32]([4, 2])
    var c = Tensor[F32]([2, 2])
    with assert_raises(contains="inner dimensions differ"):
        matmul_tiled[F32, 1, W](a, b, c)
    var b2 = Tensor[F32]([3, 2])
    var c2 = Tensor[F32]([2, 3])
    with assert_raises(contains="output shape"):
        matmul_tiled[F32, 1, W](a, b2, c2)


def main() raises:
    test_build_does_not_contract()
    test_view_of_tensor()
    test_tile_keeps_parent_stride()
    test_mutable_view_writes_through()
    test_view_over_span()
    test_view_bounds()
    test_tiled_configs_on_tensors()
    test_tiled_default_width_f64()
    test_tiled_on_views_into_larger_buffers()
    test_same_kernel_tensor_and_view()
    test_tiled_rejects_mismatch()
    print("nanodsp layout: 11 tests passed")
