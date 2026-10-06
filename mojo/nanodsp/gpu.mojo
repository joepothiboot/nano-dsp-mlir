from max.gpu import barrier, block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from std.memory import stack_allocation

from .constants import (
    BLOCK_K,
    BLOCK_M,
    BLOCK_N,
    CONV_BLOCK_THREADS,
    THREAD_M,
    THREAD_N,
    TILE,
)
from .tensor import Tensor

comptime NAIVE = 0
comptime TILED = 1
comptime BLOCKED = 2


def _naive[
    dtype: DType
](
    a: Pointer[Scalar[dtype], MutAnyOrigin],
    b: Pointer[Scalar[dtype], MutAnyOrigin],
    c: Pointer[Scalar[dtype], MutAnyOrigin],
    m: Int32,
    n: Int32,
    k: Int32,
):
    var row = Int(block_idx.y * block_dim.y + thread_idx.y)
    var col = Int(block_idx.x * block_dim.x + thread_idx.x)
    var M = Int(m)
    var N = Int(n)
    var K = Int(k)

    if row >= M or col >= N:
        return

    var acc = Scalar[dtype](0)

    for kk in range(K):
        acc = (
            acc + a[unsafe_offset=row * K + kk] * b[unsafe_offset=kk * N + col]
        )

    c[unsafe_offset=row * N + col] = acc


def _tiled[
    dtype: DType
](
    a: Pointer[Scalar[dtype], MutAnyOrigin],
    b: Pointer[Scalar[dtype], MutAnyOrigin],
    c: Pointer[Scalar[dtype], MutAnyOrigin],
    m: Int32,
    n: Int32,
    k: Int32,
):
    var as_ = stack_allocation[
        TILE * TILE, Scalar[dtype], address_space=AddressSpace.SHARED
    ]()
    var bs = stack_allocation[
        TILE * TILE, Scalar[dtype], address_space=AddressSpace.SHARED
    ]()
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var row = Int(block_idx.y) * TILE + ty
    var col = Int(block_idx.x) * TILE + tx
    var M = Int(m)
    var N = Int(n)
    var K = Int(k)
    var acc = Scalar[dtype](0)
    var k0 = 0

    while k0 < K:
        as_[unsafe_offset=ty * TILE + tx] = (
            a[unsafe_offset=row * K + k0 + tx] if row < M and k0 + tx < K else 0
        )
        bs[unsafe_offset=ty * TILE + tx] = (
            b[unsafe_offset=(k0 + ty) * N + col] if k0 + ty < K
            and col < N else 0
        )
        barrier()

        for kk in range(min(TILE, K - k0)):
            acc = (
                acc
                + as_[unsafe_offset=ty * TILE + kk]
                * bs[unsafe_offset=kk * TILE + tx]
            )

        barrier()
        k0 += TILE

    if row < M and col < N:
        c[unsafe_offset=row * N + col] = acc


def _blocked[
    dtype: DType
](
    a: Pointer[Scalar[dtype], MutAnyOrigin],
    b: Pointer[Scalar[dtype], MutAnyOrigin],
    c: Pointer[Scalar[dtype], MutAnyOrigin],
    m: Int32,
    n: Int32,
    k: Int32,
):
    comptime threads = (BLOCK_M // THREAD_M) * (BLOCK_N // THREAD_N)
    var as_ = stack_allocation[
        BLOCK_M * BLOCK_K, Scalar[dtype], address_space=AddressSpace.SHARED
    ]()
    var bs = stack_allocation[
        BLOCK_K * BLOCK_N, Scalar[dtype], address_space=AddressSpace.SHARED
    ]()
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var tid = ty * (BLOCK_N // THREAD_N) + tx
    var row0 = Int(block_idx.y) * BLOCK_M
    var col0 = Int(block_idx.x) * BLOCK_N
    var M = Int(m)
    var N = Int(n)
    var K = Int(k)

    var acc = stack_allocation[THREAD_M * THREAD_N, Scalar[dtype]]()
    comptime for i in range(THREAD_M * THREAD_N):
        acc[unsafe_offset=i] = 0

    var k0 = 0

    while k0 < K:
        comptime for r in range(BLOCK_M * BLOCK_K // threads):
            var e = tid + r * threads
            var ar = e // BLOCK_K
            var ak = e % BLOCK_K
            as_[unsafe_offset=e] = (
                a[unsafe_offset=(row0 + ar) * K + k0 + ak] if row0 + ar < M
                and k0 + ak < K else 0
            )
            var bk = e // BLOCK_N
            var bc = e % BLOCK_N
            bs[unsafe_offset=e] = (
                b[unsafe_offset=(k0 + bk) * N + col0 + bc] if k0 + bk < K
                and col0 + bc < N else 0
            )

        barrier()

        for kk in range(min(BLOCK_K, K - k0)):
            comptime for i in range(THREAD_M):
                var av = as_[
                    unsafe_offset=(ty + i * (BLOCK_M // THREAD_M)) * BLOCK_K
                    + kk
                ]
                comptime for j in range(THREAD_N):
                    acc[unsafe_offset=i * THREAD_N + j] = (
                        acc[unsafe_offset=i * THREAD_N + j]
                        + av
                        * bs[
                            unsafe_offset=kk * BLOCK_N
                            + tx
                            + j * (BLOCK_N // THREAD_N)
                        ]
                    )

        barrier()
        k0 += BLOCK_K

    comptime for i in range(THREAD_M):
        comptime for j in range(THREAD_N):
            var r = row0 + ty + i * (BLOCK_M // THREAD_M)
            var cc = col0 + tx + j * (BLOCK_N // THREAD_N)

            if r < M and cc < N:
                c[unsafe_offset=r * N + cc] = acc[
                    unsafe_offset=i * THREAD_N + j
                ]


def _blocks(extent: Int, per_block: Int) -> Int:
    return (extent + per_block - 1) // per_block


def enqueue_matmul[
    dtype: DType, variant: Int = BLOCKED
](
    ctx: DeviceContext,
    a: DeviceBuffer[dtype],
    b: DeviceBuffer[dtype],
    c: DeviceBuffer[dtype],
    m: Int,
    n: Int,
    k: Int,
) raises:
    if len(a) < m * k or len(b) < k * n or len(c) < m * n:
        raise Error("enqueue_matmul: a buffer is smaller than its shape")

    if max(m, n, k) > Int(Int32.MAX):
        raise Error("enqueue_matmul: dimensions must fit in Int32")

    var pa = a.unsafe_ptr()
    var pb = b.unsafe_ptr()
    var pc = c.unsafe_ptr()
    comptime if variant == NAIVE:
        ctx.enqueue_function[_naive[dtype]](
            pa,
            pb,
            pc,
            Int32(m),
            Int32(n),
            Int32(k),
            grid_dim=(_blocks(n, TILE), _blocks(m, TILE)),
            block_dim=(TILE, TILE),
        )
    elif variant == TILED:
        ctx.enqueue_function[_tiled[dtype]](
            pa,
            pb,
            pc,
            Int32(m),
            Int32(n),
            Int32(k),
            grid_dim=(_blocks(n, TILE), _blocks(m, TILE)),
            block_dim=(TILE, TILE),
        )
    else:
        comptime assert (
            variant == BLOCKED
        ), "variant must be NAIVE, TILED or BLOCKED"
        ctx.enqueue_function[_blocked[dtype]](
            pa,
            pb,
            pc,
            Int32(m),
            Int32(n),
            Int32(k),
            grid_dim=(_blocks(n, BLOCK_N), _blocks(m, BLOCK_M)),
            block_dim=(BLOCK_N // THREAD_N, BLOCK_M // THREAD_M),
        )


def matmul_gpu[
    dtype: DType, variant: Int = BLOCKED
](ctx: DeviceContext, a: Tensor[dtype], b: Tensor[dtype]) raises -> Tensor[
    dtype
]:
    if a.rank() != 2 or b.rank() != 2:
        raise Error("matmul_gpu: operands must be rank 2")

    var m = a.shape[0]
    var k = a.shape[1]
    var n = b.shape[1]

    if b.shape[0] != k:
        raise Error("matmul_gpu: inner dimensions differ")

    var result = Tensor[dtype]([m, n])

    if m == 0 or n == 0:
        return result^

    var da = ctx.enqueue_create_buffer[dtype](max(m * k, 1))
    var db = ctx.enqueue_create_buffer[dtype](max(k * n, 1))
    var dc = ctx.enqueue_create_buffer[dtype](m * n)

    if m * k > 0:
        ctx.enqueue_copy(da, a.data.unsafe_ptr())

    if k * n > 0:
        ctx.enqueue_copy(db, b.data.unsafe_ptr())

    enqueue_matmul[dtype, variant](ctx, da, db, dc, m, n, k)
    ctx.enqueue_copy(result.data.unsafe_ptr(), dc)
    ctx.synchronize()

    return result^


def _conv2d[
    dtype: DType
](
    input: Pointer[Scalar[dtype], MutAnyOrigin],
    filter: Pointer[Scalar[dtype], MutAnyOrigin],
    result: Pointer[Scalar[dtype], MutAnyOrigin],
    dims: Pointer[Int32, MutAnyOrigin],
):
    var h = Int(dims[unsafe_offset=1])
    var w = Int(dims[unsafe_offset=2])
    var c = Int(dims[unsafe_offset=3])
    var kh = Int(dims[unsafe_offset=4])
    var kw = Int(dims[unsafe_offset=5])
    var f = Int(dims[unsafe_offset=6])
    var oh = Int(dims[unsafe_offset=7])
    var ow = Int(dims[unsafe_offset=8])
    var t = Int(block_idx.x * block_dim.x + thread_idx.x)

    if t >= Int(dims[unsafe_offset=0]) * oh * ow * f:
        return

    var fo = t % f
    var x = (t // f) % ow
    var y = (t // (f * ow)) % oh
    var n = t // (f * ow * oh)
    var acc = Scalar[dtype](0)

    for ky in range(kh):
        var iy = y * Int(dims[unsafe_offset=9]) + ky * Int(
            dims[unsafe_offset=11]
        )

        for kx in range(kw):
            var ix = x * Int(dims[unsafe_offset=10]) + kx * Int(
                dims[unsafe_offset=12]
            )
            var in_off = ((n * h + iy) * w + ix) * c

            for ch in range(c):
                acc = (
                    acc
                    + input[unsafe_offset=in_off + ch]
                    * filter[unsafe_offset=((ky * kw + kx) * c + ch) * f + fo]
                )

    result[unsafe_offset=t] = acc


struct Conv2dShape(Copyable, Movable):
    var values: List[Int]
    var output_shape: List[Int]
    var total: Int

    def __init__(
        out self,
        input_shape: List[Int],
        filter_shape: List[Int],
        stride_h: Int = 1,
        stride_w: Int = 1,
        dilation_h: Int = 1,
        dilation_w: Int = 1,
    ) raises:
        if len(input_shape) != 4 or len(filter_shape) != 4:
            raise Error("conv2d_gpu: input and filter must be rank 4")

        if stride_h <= 0 or stride_w <= 0 or dilation_h <= 0 or dilation_w <= 0:
            raise Error("conv2d_gpu: strides and dilations must be positive")

        if filter_shape[2] != input_shape[3]:
            raise Error("conv2d_gpu: channel counts differ")

        var kh = filter_shape[0]
        var kw = filter_shape[1]
        var oh = (input_shape[1] - (kh - 1) * dilation_h - 1) // stride_h + 1
        var ow = (input_shape[2] - (kw - 1) * dilation_w - 1) // stride_w + 1

        if oh <= 0 or ow <= 0:
            raise Error("conv2d_gpu: filter is larger than the input")

        self.output_shape = [input_shape[0], oh, ow, filter_shape[3]]
        self.total = input_shape[0] * oh * ow * filter_shape[3]
        var inputs = (
            input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]
        )
        var filters = kh * kw * filter_shape[2] * filter_shape[3]

        if max(self.total, inputs, filters) > Int(Int32.MAX):
            raise Error(
                "conv2d_gpu: tensors must have fewer than 2^31 elements"
            )

        self.values = [
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
            kh,
            kw,
            filter_shape[3],
            oh,
            ow,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
        ]

    def device(self, ctx: DeviceContext) raises -> DeviceBuffer[DType.int32]:
        var dims = List[Int32](capacity=len(self.values))

        for v in self.values:
            dims.append(Int32(v))

        var buffer = ctx.enqueue_create_buffer[DType.int32](len(dims))
        ctx.enqueue_copy(buffer, dims.unsafe_ptr())

        return buffer


def enqueue_conv2d[
    dtype: DType
](
    ctx: DeviceContext,
    input: DeviceBuffer[dtype],
    filter: DeviceBuffer[dtype],
    result: DeviceBuffer[dtype],
    dims: DeviceBuffer[DType.int32],
    total: Int,
) raises:
    if total == 0:
        return

    ctx.enqueue_function[_conv2d[dtype]](
        input.unsafe_ptr(),
        filter.unsafe_ptr(),
        result.unsafe_ptr(),
        dims.unsafe_ptr(),
        grid_dim=_blocks(total, CONV_BLOCK_THREADS),
        block_dim=CONV_BLOCK_THREADS,
    )


def conv2d_gpu[
    dtype: DType
](
    ctx: DeviceContext,
    input: Tensor[dtype],
    filter: Tensor[dtype],
    stride_h: Int = 1,
    stride_w: Int = 1,
    dilation_h: Int = 1,
    dilation_w: Int = 1,
) raises -> Tensor[dtype]:
    var shape = Conv2dShape(
        input.shape, filter.shape, stride_h, stride_w, dilation_h, dilation_w
    )
    var result = Tensor[dtype](shape.output_shape.copy())

    if shape.total == 0:
        return result^

    var di = ctx.enqueue_create_buffer[dtype](max(input.numel(), 1))
    var df = ctx.enqueue_create_buffer[dtype](max(filter.numel(), 1))
    var dr = ctx.enqueue_create_buffer[dtype](shape.total)

    if input.numel() > 0:
        ctx.enqueue_copy(di, input.data.unsafe_ptr())

    if filter.numel() > 0:
        ctx.enqueue_copy(df, filter.data.unsafe_ptr())

    var dims = shape.device(ctx)
    enqueue_conv2d(ctx, di, df, dr, dims, shape.total)
    ctx.enqueue_copy(result.data.unsafe_ptr(), dr)
    ctx.synchronize()
    _ = dims^

    return result^
