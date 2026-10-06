from std.sys import simd_width_of

from .layout import TensorLike
from .tensor import Tensor


def add[
    dtype: DType
](a: Tensor[dtype], b: Tensor[dtype]) raises -> Tensor[dtype]:
    _require(a.shape == b.shape, "add: operand shapes differ")
    var result = Tensor[dtype](a.shape.copy())
    var n = a.numel()
    var pa = a.data.unsafe_ptr()
    var pb = b.data.unsafe_ptr()
    var pr = result.data.unsafe_ptr()

    comptime width = simd_width_of[dtype]()
    var i = 0

    while i + width <= n:
        pr.unsafe_store(
            i, pa.unsafe_load[width=width](i) + pb.unsafe_load[width=width](i)
        )
        i += width

    while i < n:
        pr[unsafe_offset=i] = pa[unsafe_offset=i] + pb[unsafe_offset=i]
        i += 1

    return result^


def relu[dtype: DType](a: Tensor[dtype]) -> Tensor[dtype]:
    var result = Tensor[dtype](a.shape.copy())
    var n = a.numel()
    var pa = a.data.unsafe_ptr()
    var pr = result.data.unsafe_ptr()

    comptime width = simd_width_of[dtype]()
    var zeros = SIMD[dtype, width](0)
    var i = 0

    while i + width <= n:
        var v = pa.unsafe_load[width=width](i)
        pr.unsafe_store(i, v.lt(zeros).select(zeros, v))
        i += width

    while i < n:
        var x = pa[unsafe_offset=i]
        pr[unsafe_offset=i] = 0 if x < 0 else x
        i += 1

    return result^


def matmul[
    dtype: DType
](a: Tensor[dtype], b: Tensor[dtype]) raises -> Tensor[dtype]:
    _require(a.rank() == 2 and b.rank() == 2, "matmul: operands must be rank 2")
    var m = a.shape[0]
    var k = a.shape[1]
    var n = b.shape[1]
    _require(b.shape[0] == k, "matmul: inner dimensions differ")

    var result = Tensor[dtype]([m, n])

    for i in range(m):
        for kk in range(k):
            _axpy(result.data, i * n, b.data, kk * n, a.data[i * k + kk], n)

    return result^


def matmul_tiled[
    A: TensorLike,
    B: TensorLike,
    C: TensorLike,
    //,
    dtype: DType,
    tile_m: Int,
    tile_n: Int,
    width: Int = simd_width_of[dtype](),
](a: A, b: B, mut c: C) raises:
    comptime assert A.element_dtype == dtype, "matmul_tiled: dtype of a"
    comptime assert B.element_dtype == dtype, "matmul_tiled: dtype of b"
    comptime assert C.element_dtype == dtype, "matmul_tiled: dtype of c"
    comptime assert tile_m > 0 and tile_n > 0 and width > 0
    comptime assert (
        tile_n % width == 0
    ), "matmul_tiled: tile_n must be a multiple of width"

    var m = a.rows()
    var k = a.cols()
    var n = b.cols()
    _require(b.rows() == k, "matmul_tiled: inner dimensions differ")
    _require(
        c.rows() == m and c.cols() == n,
        "matmul_tiled: output shape is not M x N",
    )

    for i0 in range(0, m, tile_m):
        var h = min(tile_m, m - i0)

        for j0 in range(0, n, tile_n):
            var w = min(tile_n, n - j0)

            if h == tile_m and w == tile_n:
                _matmul_micro[dtype, tile_m, tile_n, width](a, b, c, i0, j0, k)
            else:
                _matmul_edge[dtype, width](a, b, c, i0, j0, h, w, k)


@always_inline
def _matmul_micro[
    dtype: DType,
    tile_m: Int,
    tile_n: Int,
    width: Int,
    A: TensorLike,
    B: TensorLike,
    C: TensorLike,
](a: A, b: B, mut c: C, i0: Int, j0: Int, k: Int):
    comptime nv = tile_n // width
    var acc = Array[SIMD[dtype, width], tile_m * nv](fill=0)

    for kk in range(k):
        var bv = Array[SIMD[dtype, width], nv](fill=0)
        comptime for v in range(nv):
            bv[v] = rebind[SIMD[dtype, width]](
                b.load[width](kk, j0 + v * width)
            )

        comptime for r in range(tile_m):
            var av = SIMD[dtype, width](
                rebind[Scalar[dtype]](a.load[1](i0 + r, kk))
            )
            comptime for v in range(nv):
                acc[r * nv + v] = acc[r * nv + v] + av * bv[v]

    comptime for r in range(tile_m):
        comptime for v in range(nv):
            c.store[width](
                i0 + r,
                j0 + v * width,
                rebind[SIMD[C.element_dtype, width]](acc[r * nv + v]),
            )


def _matmul_edge[
    dtype: DType, width: Int, A: TensorLike, B: TensorLike, C: TensorLike
](a: A, b: B, mut c: C, i0: Int, j0: Int, h: Int, w: Int, k: Int):
    for i in range(i0, i0 + h):
        var j = j0

        while j < j0 + w:
            c.store[1](i, j, 0)
            j += 1

        for kk in range(k):
            var alpha = rebind[Scalar[dtype]](a.load[1](i, kk))
            var av = SIMD[dtype, width](alpha)
            j = j0

            while j + width <= j0 + w:
                var cv = rebind[SIMD[dtype, width]](c.load[width](i, j))
                var bv = rebind[SIMD[dtype, width]](b.load[width](kk, j))
                c.store[width](
                    i, j, rebind[SIMD[C.element_dtype, width]](cv + av * bv)
                )
                j += width

            while j < j0 + w:
                var cs = rebind[Scalar[dtype]](c.load[1](i, j))
                var bs = rebind[Scalar[dtype]](b.load[1](kk, j))
                c.store[1](
                    i, j, rebind[Scalar[C.element_dtype]](cs + alpha * bs)
                )
                j += 1


def conv2d[
    dtype: DType
](
    input: Tensor[dtype],
    filter: Tensor[dtype],
    stride_h: Int = 1,
    stride_w: Int = 1,
    dilation_h: Int = 1,
    dilation_w: Int = 1,
) raises -> Tensor[dtype]:
    _require(
        input.rank() == 4 and filter.rank() == 4,
        "conv2d: input and filter must be rank 4",
    )
    _require(
        stride_h > 0 and stride_w > 0 and dilation_h > 0 and dilation_w > 0,
        "conv2d: strides and dilations must be positive",
    )
    var nb = input.shape[0]
    var h = input.shape[1]
    var w = input.shape[2]
    var c = input.shape[3]
    var kh = filter.shape[0]
    var kw = filter.shape[1]
    var f = filter.shape[3]
    _require(filter.shape[2] == c, "conv2d: channel counts differ")

    var oh = (h - (kh - 1) * dilation_h - 1) // stride_h + 1
    var ow = (w - (kw - 1) * dilation_w - 1) // stride_w + 1
    _require(oh > 0 and ow > 0, "conv2d: filter is larger than the input")

    var result = Tensor[dtype]([nb, oh, ow, f])

    for n in range(nb):
        for y in range(oh):
            for x in range(ow):
                var out_off = ((n * oh + y) * ow + x) * f

                for ky in range(kh):
                    var iy = y * stride_h + ky * dilation_h

                    for kx in range(kw):
                        var ix = x * stride_w + kx * dilation_w
                        var in_off = ((n * h + iy) * w + ix) * c

                        for ch in range(c):
                            var f_off = ((ky * kw + kx) * c + ch) * f
                            _axpy(
                                result.data,
                                out_off,
                                filter.data,
                                f_off,
                                input.data[in_off + ch],
                                f,
                            )

    return result^


@always_inline
def _axpy[
    dtype: DType
](
    mut dst: List[Scalar[dtype]],
    dst_off: Int,
    src: List[Scalar[dtype]],
    src_off: Int,
    alpha: Scalar[dtype],
    n: Int,
):
    comptime width = simd_width_of[dtype]()
    var pd = dst.unsafe_ptr().unsafe_offset(dst_off)
    var ps = src.unsafe_ptr().unsafe_offset(src_off)
    var va = SIMD[dtype, width](alpha)
    var j = 0

    while j + width <= n:
        pd.unsafe_store(
            j,
            pd.unsafe_load[width=width](j)
            + va * ps.unsafe_load[width=width](j),
        )
        j += width

    while j < n:
        pd[unsafe_offset=j] = pd[unsafe_offset=j] + alpha * ps[unsafe_offset=j]
        j += 1


@always_inline
def _require(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(msg)
