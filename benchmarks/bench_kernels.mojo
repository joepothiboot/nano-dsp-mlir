# Throughput of the Mojo kernels, using the timing policy in harness.cpp.
#
# Run from the repo root:  pixi run bench

from std.sys import simd_width_of
from std.time import perf_counter_ns
from std.benchmark import keep
from std.math import sqrt

from nanodsp import QuantParams, Tensor, conv2d, matmul, matmul_tiled, qmatmul

comptime F32 = DType.float32
comptime I8 = DType.int8
comptime SAMPLES = 10
comptime MIN_SAMPLE_NS = 50_000_000
comptime W = simd_width_of[F32]()


struct Timing(Copyable, Movable):
    var best_ns: Float64
    var median_ns: Float64
    var sample_stddev_ns: Float64

    def __init__(
        out self,
        best_ns: Float64,
        median_ns: Float64,
        sample_stddev_ns: Float64,
    ):
        self.best_ns = best_ns
        self.median_ns = median_ns
        self.sample_stddev_ns = sample_stddev_ns


def time_samples(run: Some[def() raises]) raises -> Timing:
    run()
    var iterations = 1
    var t0 = perf_counter_ns()
    for _ in range(iterations):
        run()
    var elapsed = perf_counter_ns() - t0
    while elapsed < MIN_SAMPLE_NS:
        iterations *= 2
        t0 = perf_counter_ns()
        for _ in range(iterations):
            run()
        elapsed = perf_counter_ns() - t0
    var sample_times = List[Float64]()
    sample_times.append(Float64(elapsed) / Float64(iterations))
    for _ in range(1, SAMPLES):
        t0 = perf_counter_ns()
        for _ in range(iterations):
            run()
        elapsed = perf_counter_ns() - t0
        sample_times.append(Float64(elapsed) / Float64(iterations))

    var best = sample_times[0]
    var total = Float64(0)
    var ordered_times = sample_times.copy()
    for i in range(SAMPLES):
        total += sample_times[i]
        best = min(best, sample_times[i])
        var smallest = i
        for j in range(i + 1, SAMPLES):
            if ordered_times[j] < ordered_times[smallest]:
                smallest = j
        var swap = ordered_times[i]
        ordered_times[i] = ordered_times[smallest]
        ordered_times[smallest] = swap

    var mean = total / Float64(SAMPLES)
    var squared_deviations = Float64(0)
    for sample in sample_times:
        var difference = sample - mean
        squared_deviations += difference * difference
    var sample_stddev = sqrt(squared_deviations / Float64(SAMPLES - 1))
    var median = (
        ordered_times[SAMPLES // 2 - 1] + ordered_times[SAMPLES // 2]
    ) / 2.0
    return Timing(best, median, sample_stddev)


def report(
    label: String,
    shape: String,
    ops: Int,
    timing: Timing,
    unit: String = "GFLOP/s",
):
    print(
        label,
        " ",
        shape,
        "  ",
        timing.best_ns / 1000.0,
        " us min / ",
        timing.median_ns / 1000.0,
        " us median ± ",
        timing.sample_stddev_ns / 1000.0,
        " us sd  ",
        Float64(ops) / timing.best_ns,
        " ",
        unit,
    )


def fill_value(i: Int, j: Int, k: Int = 0, l: Int = 0) -> Float32:
    var pattern = (7 * i + 3 * j + 5 * k + 11 * l) % 13
    return Float32(pattern) * Float32(0.37) - Float32(1.9)


def make_matrix(n: Int, salt: Int) raises -> Tensor[F32]:
    var result = Tensor[F32]([n, n])
    for i in range(n):
        for j in range(n):
            result.data[i * n + j] = fill_value(i, j, salt)
    return result^


def bench_matmul(n: Int) raises:
    var a = make_matrix(n, 0)
    var b = make_matrix(n, 5)

    def run() raises {a, b} -> None:
        var result = matmul(a, b)
        keep(result.data.unsafe_ptr())

    var timing = time_samples(run)
    report(
        "matmul-alloc",
        String(n) + "x" + String(n) + "x" + String(n),
        2 * n * n * n,
        timing,
    )


def bench_matmul_tiled[
    tile_m: Int, tile_n: Int, allocate_output: Bool
](n: Int) raises:
    var a = make_matrix(n, 0)
    var b = make_matrix(n, 5)
    var c = Tensor[F32]([n, n])

    def run_alloc() raises {a, b, n} -> None:
        var output = Tensor[F32]([n, n])
        matmul_tiled[F32, tile_m, tile_n](a, b, output)
        keep(output.data.unsafe_ptr())

    def run_reuse() raises {a, b, mut c} -> None:
        matmul_tiled[F32, tile_m, tile_n](a, b, c)
        keep(c.data.unsafe_ptr())

    var mode = "tiled-reuse"
    comptime if allocate_output:
        mode = "tiled-alloc"
    var shape = String(n) + "x" + String(n) + "x" + String(n)
    var label = mode + " " + String(tile_m) + "x" + String(tile_n)
    comptime if allocate_output:
        report(label, shape, 2 * n * n * n, time_samples(run_alloc))
    else:
        report(label, shape, 2 * n * n * n, time_samples(run_reuse))


def make_conv_input(hw: Int, channels: Int, salt: Int) raises -> Tensor[F32]:
    var result = Tensor[F32]([1, hw, hw, channels])
    for y in range(hw):
        for x in range(hw):
            for c in range(channels):
                result.data[(y * hw + x) * channels + c] = fill_value(
                    salt, y, x, c
                )
    return result^


def make_conv_filter(
    channels: Int, filters: Int, salt: Int
) raises -> Tensor[F32]:
    var result = Tensor[F32]([3, 3, channels, filters])
    for ky in range(3):
        for kx in range(3):
            for c in range(channels):
                for f in range(filters):
                    result.data[
                        ((ky * 3 + kx) * channels + c) * filters + f
                    ] = fill_value(ky + salt, kx, c, f)
    return result^


def bench_conv2d(hw: Int, c: Int, f: Int) raises:
    var input = make_conv_input(hw, c, 0)
    var filter = make_conv_filter(c, f, 2)

    def run() raises {input, filter} -> None:
        var result = conv2d(input, filter)
        keep(result.data.unsafe_ptr())

    var o = hw - 2
    var label = "conv2d-alloc"
    var shape = (
        String(hw) + "x" + String(hw) + "x" + String(c) + "->" + String(f)
    )
    report(label, shape, 2 * o * o * f * 9 * c, time_samples(run))


def make_qmatrix(n: Int, multiplier: Int, addend: Int) raises -> Tensor[I8]:
    var result = Tensor[I8]([n, n])
    for i in range(n * n):
        result.data[i] = Int8((i * multiplier + addend) % 256 - 128)
    return result^


def bench_qmatmul(n: Int) raises:
    var a = make_qmatrix(n, 73, 11)
    var b = make_qmatrix(n, 151, 7)
    var q = QuantParams(-7, 12, 1276901417, 9, 4)

    def run() raises {a, b, q} -> None:
        var result = qmatmul(a, b, q)
        keep(result.data.unsafe_ptr())

    var shape = String(n) + "x" + String(n) + "x" + String(n)
    report("qmatmul-alloc", shape, 2 * n * n * n, time_samples(run), "GOP/s")


def main() raises:
    for n in [64, 128, 256, 512]:
        bench_matmul(n)
    for n in [256, 512]:
        bench_matmul_tiled[4, 4 * W, allocate_output=True](n)
        bench_matmul_tiled[4, 4 * W, allocate_output=False](n)
    bench_conv2d(56, 64, 64)
    bench_conv2d(28, 128, 128)
    bench_qmatmul(256)
