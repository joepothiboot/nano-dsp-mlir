# Tests for the nanodsp Mojo kernels.
#
# Two kinds of check:
#   - golden: the same inputs and expected outputs as test/Integration/, so a
#     disagreement between Mojo and the MLIR pipeline shows up as a failure
#     on one side.
#   - differential: SIMD kernel vs. a naive scalar loop nest on odd sizes, so
#     every SIMD tail path runs. Inputs are small integers, so every sum is
#     exact in f32 and the comparison can be exact equality.
#
# Run from the repo root:  pixi run test-mojo

from std.testing import assert_equal, assert_raises, assert_true

from nanodsp import QuantParams, Tensor, add, conv2d, matmul, qmatmul, relu, requantize

comptime F32 = DType.float32


def f32(var shape: List[Int], var values: List[Float32]) raises -> Tensor[F32]:
    return Tensor[F32](shape^, values^)


def pattern(var shape: List[Int], seed: Int) raises -> Tensor[F32]:
    """Deterministic small-integer values in [-6, 6]."""
    var t = Tensor[F32](shape^)
    for i in range(t.numel()):
        t.data[i] = Float32((i * 7 + seed) % 13) - 6.0
    return t^


def assert_same(actual: Tensor[F32], expected: Tensor[F32]) raises:
    assert_true(actual.shape == expected.shape, msg="shape mismatch")
    for i in range(expected.numel()):
        assert_equal(actual[i], expected[i], msg="at flat index " + String(i))


def assert_values(actual: Tensor[F32], expected: List[Float32]) raises:
    assert_equal(actual.numel(), len(expected))
    for i in range(len(expected)):
        assert_equal(actual[i], expected[i], msg="at flat index " + String(i))


# --- naive scalar references ------------------------------------------------


def naive_matmul(a: Tensor[F32], b: Tensor[F32]) raises -> Tensor[F32]:
    var m = a.shape[0]
    var k = a.shape[1]
    var n = b.shape[1]
    var r = Tensor[F32]([m, n])
    for i in range(m):
        for j in range(n):
            var acc: Float32 = 0
            for kk in range(k):
                acc += a[i * k + kk] * b[kk * n + j]
            r.data[i * n + j] = acc
    return r^


def naive_conv2d(
    input: Tensor[F32], filter: Tensor[F32], sh: Int, sw: Int, dh: Int, dw: Int
) raises -> Tensor[F32]:
    var nb = input.shape[0]
    var h = input.shape[1]
    var w = input.shape[2]
    var c = input.shape[3]
    var kh = filter.shape[0]
    var kw = filter.shape[1]
    var f = filter.shape[3]
    var oh = (h - (kh - 1) * dh - 1) // sh + 1
    var ow = (w - (kw - 1) * dw - 1) // sw + 1
    var r = Tensor[F32]([nb, oh, ow, f])
    for n in range(nb):
        for y in range(oh):
            for x in range(ow):
                for fo in range(f):
                    var acc: Float32 = 0
                    for ky in range(kh):
                        for kx in range(kw):
                            for ch in range(c):
                                var iy = y * sh + ky * dh
                                var ix = x * sw + kx * dw
                                acc += (
                                    input[((n * h + iy) * w + ix) * c + ch]
                                    * filter[((ky * kw + kx) * c + ch) * f + fo]
                                )
                    r.data[((n * oh + y) * ow + x) * f + fo] = acc
    return r^


# --- add --------------------------------------------------------------------


def test_add_golden() raises:
    # test/Integration/end-to-end.mlir
    var r = add(f32([4], [1.0, 2.0, 3.0, 4.0]), f32([4], [1.0, 1.0, 1.0, 1.0]))
    assert_values(r, [2.0, 3.0, 4.0, 5.0])


def test_add_tail() raises:
    # 37 is not a multiple of any SIMD width, so the scalar tail runs.
    var a = pattern([37], 1)
    var b = pattern([37], 5)
    var r = add(a, b)
    for i in range(37):
        assert_equal(r[i], a[i] + b[i])


def test_add_rejects_broadcast() raises:
    with assert_raises(contains="shapes differ"):
        _ = add(pattern([2, 3], 0), pattern([3], 0))


# --- relu -------------------------------------------------------------------


def test_relu_values() raises:
    var r = relu(f32([6], [-2.0, -0.5, 0.0, 0.5, 2.0, -7.0]))
    assert_values(r, [0.0, 0.0, 0.0, 0.5, 2.0, 0.0])


def test_relu_propagates_nan() raises:
    # arith.maximumf semantics: NaN in, NaN out -- in the SIMD body and the
    # scalar tail alike.
    var zero: Float32 = 0
    var nan = zero / zero
    var t = Tensor[F32]([33], fill=nan)
    var r = relu(t)
    for i in range(33):
        assert_true(r[i] != r[i], msg="NaN lost at index " + String(i))


# --- matmul -----------------------------------------------------------------


def test_matmul_golden() raises:
    # test/Integration/DSPToLinalg/matmul.mlir -- non-square on purpose.
    var a = f32([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    var b = f32([3, 2], [1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    assert_values(matmul(a, b), [4.0, 5.0, 10.0, 11.0])


def test_matmul_matches_naive() raises:
    var a = pattern([7, 13], 2)
    var b = pattern([13, 19], 3)
    assert_same(matmul(a, b), naive_matmul(a, b))


def test_matmul_rejects_mismatch() raises:
    with assert_raises(contains="inner dimensions"):
        _ = matmul(pattern([2, 3], 0), pattern([2, 3], 0))


# --- conv2d -----------------------------------------------------------------


def test_conv2d_golden() raises:
    # test/Integration/DSPToLinalg/conv2d.mlir: 4x4 input 1..16, 3x3 ones.
    var vals = List[Float32]()
    for i in range(16):
        vals.append(Float32(i + 1))
    var input = f32([1, 4, 4, 1], vals^)
    var filter = Tensor[F32]([3, 3, 1, 1], fill=1.0)
    assert_values(conv2d(input, filter), [54.0, 63.0, 90.0, 99.0])


def test_conv2d_matches_naive() raises:
    # Multi-channel, multi-filter, F=11 so the SIMD tail runs.
    var input = pattern([2, 9, 8, 3], 4)
    var filter = pattern([3, 2, 3, 11], 6)
    assert_same(conv2d(input, filter), naive_conv2d(input, filter, 1, 1, 1, 1))


def test_conv2d_strided_dilated() raises:
    var input = pattern([1, 11, 10, 2], 1)
    var filter = pattern([3, 3, 2, 5], 2)
    var r = conv2d(input, filter, stride_h=2, stride_w=3, dilation_h=2)
    assert_same(r, naive_conv2d(input, filter, 2, 3, 2, 1))


def test_conv2d_rejects_channel_mismatch() raises:
    with assert_raises(contains="channel counts"):
        _ = conv2d(pattern([1, 4, 4, 2], 0), pattern([3, 3, 1, 1], 0))


# --- qmatmul ----------------------------------------------------------------

comptime I8 = DType.int8


def i8(var shape: List[Int], var values: List[Int8]) raises -> Tensor[I8]:
    return Tensor[I8](shape^, values^)


def assert_i8(actual: Tensor[I8], expected: List[Int8]) raises:
    assert_equal(actual.numel(), len(expected))
    for i in range(len(expected)):
        assert_equal(actual[i], expected[i], msg="at flat index " + String(i))


def test_qmatmul_golden_zero_points() raises:
    # test/Integration/DSPToLinalg/qmatmul.mlir, case 1: non-zero zero
    # points, saturation at both ends, and both .5 ties.
    var a = i8([2, 3], [-128, 0, 127, 10, -7, 50])
    var b = i8([3, 4], [1, -3, 127, -128, 4, 2, 0, 9, -1, 5, -55, 30])
    var q = QuantParams(3, -2, 1073741824, 3, -5)
    assert_i8(qmatmul(a, b, q), [-23, 57, -128, 127, -4, 13, -105, 27])


def test_qmatmul_golden_fractional_multiplier() raises:
    # Case 2: multiplier ~0.7071, not a power of two.
    var a = i8([2, 4], [127, -128, 64, -1, -50, 33, -2, 90])
    var b = i8([4, 2], [3, -7, -2, 11, 100, -100, -9, 4])
    var q = QuantParams(0, 0, 1518500250, 5, 1)
    assert_i8(qmatmul(a, b, q), [127, -128, -26, 29])


def test_requantize_rounds_half_up() raises:
    # scale 1/16: 8 -> 0.5 -> 1, -8 -> -0.5 -> 0, -24 -> -1.5 -> -1.
    var q = QuantParams(0, 0, 1073741824, 3, 0)
    assert_equal(requantize(8, q), 1)
    assert_equal(requantize(-8, q), 0)
    assert_equal(requantize(-24, q), -1)
    assert_equal(requantize(24, q), 2)


def test_qmatmul_matches_naive() raises:
    # N = 19 so the i32 SIMD tail runs; values span the full i8 range.
    var m = 5
    var k = 37
    var n = 19
    var a = Tensor[I8]([m, k])
    var b = Tensor[I8]([k, n])
    for i in range(a.numel()):
        a.data[i] = Int8((i * 73 + 11) % 256 - 128)
    for i in range(b.numel()):
        b.data[i] = Int8((i * 151 + 7) % 256 - 128)
    var q = QuantParams(-7, 12, 1276901417, 9, 4)
    var r = qmatmul(a, b, q)
    for i in range(m):
        for j in range(n):
            var acc: Int32 = 0
            for kk in range(k):
                acc += (a[i * k + kk].cast[DType.int32]() - q.lhs_zp) * (
                    b[kk * n + j].cast[DType.int32]() - q.rhs_zp
                )
            assert_equal(r[i * n + j], requantize(acc, q), msg="at " + String(i) + "," + String(j))


def test_qmatmul_rejects_bad_params() raises:
    var a = i8([1, 1], [1])
    with assert_raises(contains="multiplier"):
        _ = qmatmul(a, a, QuantParams(0, 0, 1, 0, 0))
    with assert_raises(contains="zero point"):
        _ = qmatmul(a, a, QuantParams(0, 200, 1073741824, 0, 0))


def main() raises:
    test_add_golden()
    test_add_tail()
    test_add_rejects_broadcast()
    test_relu_values()
    test_relu_propagates_nan()
    test_matmul_golden()
    test_matmul_matches_naive()
    test_matmul_rejects_mismatch()
    test_conv2d_golden()
    test_conv2d_matches_naive()
    test_conv2d_strided_dilated()
    test_conv2d_rejects_channel_mismatch()
    test_qmatmul_golden_zero_points()
    test_qmatmul_golden_fractional_multiplier()
    test_requantize_rounds_half_up()
    test_qmatmul_matches_naive()
    test_qmatmul_rejects_bad_params()
    print("nanodsp: 17 tests passed")
