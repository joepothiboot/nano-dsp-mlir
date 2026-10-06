from max.gpu.host import DeviceContext
from nanodsp import Tensor, conv2d, matmul
from nanodsp.gpu import (
    BLOCKED,
    NAIVE,
    TILED,
    Conv2dShape,
    enqueue_conv2d,
    enqueue_matmul,
)

from bench_kernels import time_samples

comptime F32 = DType.float32


def inexact(var shape: List[Int], seed: Int) -> Tensor[F32]:
    var t = Tensor[F32](shape^)

    for i in range(t.numel()):
        t.data[i] = Float32(((i + seed) * 2654435761) % 1000003) / 997.0 - 500.0

    return t^


def bench[
    variant: Int
](
    ctx: DeviceContext,
    name: String,
    n: Int,
    expected: Tensor[F32],
    a: Tensor[F32],
    b: Tensor[F32],
) raises -> String:
    var da = ctx.enqueue_create_buffer[F32](n * n)
    var db = ctx.enqueue_create_buffer[F32](n * n)
    var dc = ctx.enqueue_create_buffer[F32](n * n)
    ctx.enqueue_copy(da, a.data.unsafe_ptr())
    ctx.enqueue_copy(db, b.data.unsafe_ptr())
    enqueue_matmul[F32, variant](ctx, da, db, dc, n, n, n)
    var got = List[Float32](length=n * n, fill=0)
    ctx.enqueue_copy(got.unsafe_ptr(), dc)
    ctx.synchronize()

    for i in range(n * n):
        if got[i].to_bits() != expected.data[i].to_bits():
            raise Error(
                name
                + " "
                + String(n)
                + ": differs from CPU matmul at flat index "
                + String(i)
            )

    def run() raises {ctx, da, db, dc, n} -> None:
        enqueue_matmul[F32, variant](ctx, da, db, dc, n, n, n)
        ctx.synchronize()

    var t = time_samples(run)
    var ops = 2 * n * n * n
    var bytes = 3 * n * n * 4
    var shape = String(n) + "x" + String(n) + "x" + String(n)

    return (
        '    {"name": "matmul/'
        + shape
        + "/mojo-gpu-"
        + name
        + '", "op": "matmul", "shape": "'
        + shape
        + '", "impl": "mojo-gpu-'
        + name
        + '", "config": "'
        + ctx.name()
        + '", "real_time": '
        + String(t.best_ns)
        + ', "median_time": '
        + String(t.median_ns)
        + ', "sample_stddev": '
        + String(t.sample_stddev_ns)
        + ', "time_unit": "ns", "aggregate": "min", "samples": 10, "ops": '
        + String(ops)
        + ', "rate": '
        + String(Float64(ops) / t.best_ns)
        + ', "rate_unit": "GFLOP/s", "bytes": '
        + String(bytes)
        + ', "intensity": '
        + String(Float64(ops) / Float64(bytes))
        + ', "checked": "bit-exact"}'
    )


def bench_conv2d(ctx: DeviceContext, h: Int, c: Int, f: Int) raises -> String:
    var input = inexact([1, h, h, c], 3)
    var filter = inexact([3, 3, c, f], 4)
    var expected = conv2d(input, filter)
    var shape = Conv2dShape(input.shape, filter.shape)
    var di = ctx.enqueue_create_buffer[F32](input.numel())
    var df = ctx.enqueue_create_buffer[F32](filter.numel())
    var dr = ctx.enqueue_create_buffer[F32](shape.total)
    var dims = shape.device(ctx)
    ctx.enqueue_copy(di, input.data.unsafe_ptr())
    ctx.enqueue_copy(df, filter.data.unsafe_ptr())
    enqueue_conv2d(ctx, di, df, dr, dims, shape.total)
    var got = List[Float32](length=shape.total, fill=0)
    ctx.enqueue_copy(got.unsafe_ptr(), dr)
    ctx.synchronize()

    for i in range(shape.total):
        if got[i].to_bits() != expected.data[i].to_bits():
            raise Error(
                "conv2d "
                + String(h)
                + ": differs from CPU conv2d at flat index "
                + String(i)
            )

    def run() raises {ctx, di, df, dr, dims, shape} -> None:
        enqueue_conv2d(ctx, di, df, dr, dims, shape.total)
        ctx.synchronize()

    var t = time_samples(run)
    var o = h - 2
    var ops = 2 * o * o * f * 9 * c
    var bytes = (input.numel() + filter.numel() + shape.total) * 4
    var name = String(h) + "x" + String(h) + "x" + String(c) + "->" + String(f)

    return (
        '    {"name": "conv2d/'
        + name
        + '/mojo-gpu", "op": "conv2d", "shape": "'
        + name
        + '", "impl": "mojo-gpu", "config": "'
        + ctx.name()
        + '", "real_time": '
        + String(t.best_ns)
        + ', "median_time": '
        + String(t.median_ns)
        + ', "sample_stddev": '
        + String(t.sample_stddev_ns)
        + ', "time_unit": "ns", "aggregate": "min", "samples": 10, "ops": '
        + String(ops)
        + ', "rate": '
        + String(Float64(ops) / t.best_ns)
        + ', "rate_unit": "GFLOP/s", "bytes": '
        + String(bytes)
        + ', "intensity": '
        + String(Float64(ops) / Float64(bytes))
        + ', "checked": "bit-exact"}'
    )


def main() raises:
    var ctx = DeviceContext()
    var rows = List[String]()

    for n in [256, 512, 1024, 2048]:
        var a = inexact([n, n], 1)
        var b = inexact([n, n], 2)
        var expected = matmul(a, b)
        rows.append(bench[NAIVE](ctx, "naive", n, expected, a, b))
        rows.append(bench[TILED](ctx, "tiled", n, expected, a, b))
        rows.append(bench[BLOCKED](ctx, "blocked", n, expected, a, b))

    rows.append(bench_conv2d(ctx, 56, 64, 64))
    rows.append(bench_conv2d(ctx, 28, 128, 128))
    print(
        '{\n  "context": {"device": "'
        + ctx.name()
        + '", "timing": "one launch + synchronize per call; best of 10 samples'
        ' of >= 50 ms"},'
    )
    print('  "benchmarks": [')

    for i in range(len(rows)):
        print(rows[i] + ("," if i + 1 < len(rows) else ""))

    print("  ]\n}")
