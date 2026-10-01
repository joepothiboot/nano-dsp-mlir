# Throughput of the Mojo kernels. Reports the best of `reps` runs.
#
# Run from the repo root:  pixi run bench
#
# Planned: the same shapes through the MLIR pipeline (untiled and with the
# Stage 3 schedule) and the C++ reference, reported side by side.

from std.sys import simd_width_of
from std.time import perf_counter_ns

from std.benchmark import keep

from nanodsp import Tensor, conv2d, matmul, matmul_tiled

comptime F32 = DType.float32
comptime W = simd_width_of[F32]()


def bench_matmul(n: Int, reps: Int) raises:
    var a = Tensor[F32]([n, n], fill=1.0)
    var b = Tensor[F32]([n, n], fill=2.0)
    var best = Float64.MAX
    for _ in range(reps):
        var t0 = perf_counter_ns()
        var r = matmul(a, b)
        var t1 = perf_counter_ns()
        keep(r.data.unsafe_ptr())
        best = min(best, Float64(t1 - t0))
    var gflops = 2.0 * Float64(n * n * n) / best
    print("matmul  ", n, "x", n, "x", n, "  ", best / 1e6, "ms  ", gflops, "GFLOP/s")


def bench_matmul_tiled[
    tile_m: Int, tile_n: Int, views: Bool = False
](n: Int, reps: Int) raises:
    var a = Tensor[F32]([n, n], fill=1.0)
    var b = Tensor[F32]([n, n], fill=2.0)
    var c = Tensor[F32]([n, n])
    var best = Float64.MAX
    for _ in range(reps):
        var t0: Int
        var t1: Int
        comptime if views:
            var cv = c.view()
            t0 = perf_counter_ns()
            matmul_tiled[F32, tile_m, tile_n](a.view(), b.view(), cv)
            t1 = perf_counter_ns()
        else:
            t0 = perf_counter_ns()
            matmul_tiled[F32, tile_m, tile_n](a, b, c)
            t1 = perf_counter_ns()
        keep(c.data.unsafe_ptr())
        best = min(best, Float64(t1 - t0))
    var gflops = 2.0 * Float64(n * n * n) / best
    print(
        "tiled",
        tile_m,
        "x",
        tile_n,
        "(views)" if views else "",
        " ",
        n,
        "x",
        n,
        "x",
        n,
        "  ",
        best / 1e6,
        "ms  ",
        gflops,
        "GFLOP/s",
    )


def bench_conv2d(hw: Int, c: Int, f: Int, reps: Int) raises:
    var input = Tensor[F32]([1, hw, hw, c], fill=1.0)
    var filter = Tensor[F32]([3, 3, c, f], fill=0.5)
    var best = Float64.MAX
    for _ in range(reps):
        var t0 = perf_counter_ns()
        var r = conv2d(input, filter)
        var t1 = perf_counter_ns()
        keep(r.data.unsafe_ptr())
        best = min(best, Float64(t1 - t0))
    var o = hw - 2
    var gflops = 2.0 * Float64(o * o * f * 9 * c) / best
    print("conv2d  ", hw, "x", hw, "x", c, "->", f, "  ", best / 1e6, "ms  ", gflops, "GFLOP/s")


def main() raises:
    for n in [64, 128, 256, 512]:
        bench_matmul(n, 5)
    for n in [256, 512]:
        bench_matmul_tiled[1, W](n, 5)
        bench_matmul_tiled[4, 4 * W](n, 5)
        bench_matmul_tiled[4, 4 * W, views=True](n, 5)
    bench_conv2d(56, 64, 64, 3)
    bench_conv2d(28, 128, 128, 3)
