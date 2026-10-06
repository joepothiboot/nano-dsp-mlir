from std.testing import assert_equal, assert_raises, assert_true

from max.gpu.host import DeviceContext
from nanodsp import Tensor, conv2d, matmul
from nanodsp.gpu import BLOCKED, NAIVE, TILED, conv2d_gpu, matmul_gpu

comptime F32 = DType.float32


def inexact(var shape: List[Int], seed: Int) -> Tensor[F32]:
    var t = Tensor[F32](shape^)

    for i in range(t.numel()):
        t.data[i] = Float32(((i + seed) * 2654435761) % 1000003) / 997.0 - 500.0

    return t^


def assert_bits(
    actual: Tensor[F32], expected: Tensor[F32], what: String
) raises:
    assert_true(actual.shape == expected.shape, msg=what + ": shape mismatch")

    for i in range(expected.numel()):
        assert_equal(
            actual.data[i].to_bits(),
            expected.data[i].to_bits(),
            msg=what + ": bits differ at flat index " + String(i),
        )


def check_variant[variant: Int](ctx: DeviceContext, name: String) raises:
    var shapes: List[Tuple[Int, Int, Int]] = [
        (1, 1, 1),
        (16, 16, 16),
        (64, 64, 64),
        (17, 33, 5),
        (65, 31, 47),
        (3, 130, 70),
        (100, 1, 257),
    ]

    for s in shapes:
        var m = s[0]
        var n = s[1]
        var k = s[2]
        var a = inexact([m, k], 1)
        var b = inexact([k, n], 2)
        assert_bits(
            matmul_gpu[F32, variant](ctx, a, b),
            matmul(a, b),
            name + " " + String(m) + "x" + String(n) + "x" + String(k),
        )


def test_naive(ctx: DeviceContext) raises:
    check_variant[NAIVE](ctx, "naive")


def test_tiled(ctx: DeviceContext) raises:
    check_variant[TILED](ctx, "tiled")


def test_blocked(ctx: DeviceContext) raises:
    check_variant[BLOCKED](ctx, "blocked")


def test_inputs_detect_fma(ctx: DeviceContext) raises:
    var a = inexact([65, 47], 1)
    var b = inexact([47, 31], 2)
    var gpu = matmul_gpu[F32, BLOCKED](ctx, a, b)
    var differ = 0

    for i in range(65):
        for j in range(31):
            var acc = Float32(0)

            for kk in range(47):
                acc = a.data[i * 47 + kk].fma(b.data[kk * 31 + j], acc)

            if acc.to_bits() != gpu.data[i * 31 + j].to_bits():
                differ += 1

    assert_true(
        differ > 0,
        msg="FMA gave identical bits; the inputs can't detect fusion",
    )


def test_zero_sizes(ctx: DeviceContext) raises:
    var empty = matmul_gpu[F32](ctx, Tensor[F32]([0, 3]), Tensor[F32]([3, 4]))
    assert_equal(empty.numel(), 0)
    var zero_k = matmul_gpu[F32](ctx, Tensor[F32]([2, 0]), Tensor[F32]([0, 3]))
    assert_bits(zero_k, Tensor[F32]([2, 3]), "k = 0")


def test_conv2d(ctx: DeviceContext) raises:
    var input = inexact([2, 13, 11, 5], 3)
    var filter = inexact([3, 3, 5, 7], 4)
    assert_bits(
        conv2d_gpu(ctx, input, filter), conv2d(input, filter), "conv2d 3x3"
    )
    assert_bits(
        conv2d_gpu(ctx, input, filter, stride_h=2, stride_w=3, dilation_h=2),
        conv2d(input, filter, stride_h=2, stride_w=3, dilation_h=2),
        "conv2d strided, dilated",
    )
    var wide = inexact([1, 8, 8, 64], 5)
    var wide_filter = inexact([3, 3, 64, 64], 6)
    assert_bits(
        conv2d_gpu(ctx, wide, wide_filter),
        conv2d(wide, wide_filter),
        "conv2d 64 -> 64",
    )


def test_conv2d_golden(ctx: DeviceContext) raises:
    var input = Tensor[F32]([1, 4, 4, 1])

    for i in range(16):
        input.data[i] = Float32(i + 1)

    var result = conv2d_gpu(ctx, input, Tensor[F32]([3, 3, 1, 1], fill=1.0))
    var expected: List[Float32] = [54.0, 63.0, 90.0, 99.0]

    for i in range(4):
        assert_equal(result.data[i], expected[i])


def test_rejects_mismatch(ctx: DeviceContext) raises:
    with assert_raises(contains="inner dimensions"):
        _ = matmul_gpu[F32](ctx, Tensor[F32]([2, 3]), Tensor[F32]([4, 2]))

    with assert_raises(contains="channel counts"):
        _ = conv2d_gpu(
            ctx, Tensor[F32]([1, 4, 4, 2]), Tensor[F32]([3, 3, 1, 1])
        )


def main() raises:
    var ctx = DeviceContext()
    test_naive(ctx)
    test_tiled(ctx)
    test_blocked(ctx)
    test_inputs_detect_fma(ctx)
    test_zero_sizes(ctx)
    test_conv2d(ctx)
    test_conv2d_golden(ctx)
    test_rejects_mismatch(ctx)
    print("nanodsp gpu: 8 tests passed on", ctx.name())
