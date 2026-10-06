from std.sys import argv, simd_width_of
from std.testing import assert_equal, assert_raises, assert_true

from nanodsp import Tensor, TensorLike, TensorView, matmul, matmul_tiled

comptime F32 = DType.float32
comptime F64 = DType.float64
comptime W = simd_width_of[F32]()


def inexact[dtype: DType](var shape: List[Int], seed: Int) -> Tensor[dtype]:
    var t = Tensor[dtype](shape^)

    for i in range(t.numel()):
        t.data[i] = Scalar[dtype]((i * 7 + seed) % 13 - 6) * Scalar[dtype](
            0.37
        ) + Scalar[dtype](0.011) * Scalar[dtype]((i * 3 + seed) % 5)

    return t^


def naive_matmul[
    dtype: DType
](a: Tensor[dtype], b: Tensor[dtype]) -> Tensor[dtype]:
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
    return Float64(len(argv()) - 1) * 0.0


def unfused_matmul_f32(a: Tensor[F32], b: Tensor[F32]) -> Tensor[F32]:
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


def test_build_does_not_contract() raises:
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


def test_view_of_tensor() raises:
    var t = inexact[F32]([5, 8], 1)
    var v = t.view()
    assert_equal(v.rows(), 5)
    assert_equal(v.cols(), 8)
    assert_equal(v.row_stride(), 8)
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

    var inner = tl.tile(1, 1, 2, 2)
    assert_equal(inner[1, 1], t[3 * 8 + 4])


def test_mutable_view_writes_through() raises:
    var t = Tensor[F32]([4, 6], fill=0.0)
    var tl = t.view().tile(1, 1, 2, 4)
    tl.store[4](1, 0, SIMD[F32, 4](1.0, 2.0, 3.0, 4.0))
    assert_equal(t[2 * 6 + 1], 1.0)
    assert_equal(t[2 * 6 + 4], 4.0)
    assert_equal(t[2 * 6 + 5], 0.0)


def test_view_over_span() raises:
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


def check_tensor_config[
    tile_m: Int, tile_n: Int, width: Int
](m: Int, k: Int, n: Int) raises:
    var a = inexact[F32]([m, k], 3)
    var b = inexact[F32]([k, n], 5)
    var c = Tensor[F32]([m, n], fill=99.0)
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
    check_tensor_config[tile_m, tile_n, width](7, 13, 19)
    check_tensor_config[tile_m, tile_n, width](16, 9, 32)
    check_tensor_config[tile_m, tile_n, width](33, 17, 37)
    check_tensor_config[tile_m, tile_n, width](1, 1, 1)
    check_tensor_config[tile_m, tile_n, width](3, 40, 2)


def test_tiled_configs_on_tensors() raises:
    check_shapes[1, W, W]()
    check_shapes[4, 2 * W, W]()
    check_shapes[3, 3 * W, W]()
    check_shapes[8, 4, 4]()
    check_shapes[2, 16, 8]()
    check_shapes[5, 3, 1]()


def test_tiled_default_width_f64() raises:
    var a = inexact[F64]([9, 11], 1)
    var b = inexact[F64]([11, 21], 2)
    var c = Tensor[F64]([9, 21])
    matmul_tiled[F64, 4, 2 * simd_width_of[F64]()](a, b, c)
    assert_bit_exact(c, matmul(a, b), "f64 vs matmul")
    assert_bit_exact(c, naive_matmul(a, b), "f64 vs naive")


def test_tiled_on_views_into_larger_buffers() raises:
    comptime m = 7
    comptime k = 13
    comptime n = 19
    var big_a = inexact[F32]([20, 31], 3)
    var big_b = inexact[F32]([17, 40], 5)
    var big_c = Tensor[F32]([12, 27], fill=-5.0)
    var av = big_a.view().tile(2, 3, m, k)
    var bv = big_b.view().tile(1, 4, k, n)
    var cv = big_c.view().tile(3, 5, m, n)
    matmul_tiled[F32, 4, 2 * W](av, bv, cv)

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

    for i in range(12):
        for j in range(27):
            if i < 3 or i >= 3 + m or j < 5 or j >= 5 + n:
                assert_equal(big_c[i * 27 + j], -5.0, msg="clobbered")


def test_same_kernel_tensor_and_view() raises:
    var a = inexact[F32]([11, 10], 7)
    var b = inexact[F32]([10, 23], 8)
    var c1 = Tensor[F32]([11, 23])
    var c2 = Tensor[F32]([11, 23])
    var c3 = Tensor[F32]([11, 23])
    matmul_tiled[F32, 4, 2 * W](a, b, c1)
    var c2v = c2.view()
    matmul_tiled[F32, 4, 2 * W](a.view(), b.view(), c2v)
    matmul_tiled[F32, 4, 2 * W](a, b.view(), c3)
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
