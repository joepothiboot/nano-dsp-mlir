"""SIMD kernels for the four `dsp` ops.

All inner loops run over the innermost, contiguous dimension in chunks of the
target's native SIMD width, with a scalar tail. Reductions keep the same
order as the naive loop nest (and as `linalg.generic` after
`-convert-linalg-to-loops`), and multiply and add are kept separate rather
than fused, so results are bit-identical to the scalar reference, not just
close.
"""

from std.sys import simd_width_of

from .tensor import Tensor


def add[dtype: DType](a: Tensor[dtype], b: Tensor[dtype]) raises -> Tensor[dtype]:
    """Elementwise `a + b`, like `dsp.add`: shapes must match exactly.

    Raises:
        If the shapes differ (there is no implicit broadcasting).
    """
    _require(a.shape == b.shape, "add: operand shapes differ")
    var result = Tensor[dtype](a.shape.copy())
    var n = a.numel()
    var pa = a.data.unsafe_ptr()
    var pb = b.data.unsafe_ptr()
    var pr = result.data.unsafe_ptr()

    comptime width = simd_width_of[dtype]()
    var i = 0
    while i + width <= n:
        pr.unsafe_store(i, pa.unsafe_load[width=width](i) + pb.unsafe_load[width=width](i))
        i += width
    while i < n:
        pr[unsafe_offset=i] = pa[unsafe_offset=i] + pb[unsafe_offset=i]
        i += 1
    return result^


def relu[dtype: DType](a: Tensor[dtype]) -> Tensor[dtype]:
    """Elementwise `max(x, 0)` with NaN propagation, like `dsp.relu`.

    Written as `x < 0 ? 0 : x` so that NaN (for which `x < 0` is false)
    passes through, matching `arith.maximumf` and `numpy.maximum`.
    """
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
    """Row-major `(M x K) * (K x N) -> (M x N)`, like `dsp.matmul`.

    Uses i-k-j order so the inner loop streams a row of `b` and a row of the
    result, both contiguous.

    Raises:
        If either operand is not rank 2 or the inner dimensions differ.
    """
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
    """2-D cross-correlation, NHWC input x HWCF filter -> NHWF output.

    Matches `dsp.conv2d` / `linalg.conv_2d_nhwc_hwcf`: no kernel flip and
    'valid' padding only. The SIMD dimension is F, which is contiguous in
    both the filter and the output.

    Raises:
        If the ranks or channel counts do not match, or the output would be
        empty.
    """
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
    """`dst[dst_off:][:n] += alpha * src[src_off:][:n]`, unfused."""
    comptime width = simd_width_of[dtype]()
    var pd = dst.unsafe_ptr().unsafe_offset(dst_off)
    var ps = src.unsafe_ptr().unsafe_offset(src_off)
    var va = SIMD[dtype, width](alpha)
    var j = 0
    while j + width <= n:
        pd.unsafe_store(j, pd.unsafe_load[width=width](j) + va * ps.unsafe_load[width=width](j))
        j += width
    while j < n:
        pd[unsafe_offset=j] = pd[unsafe_offset=j] + alpha * ps[unsafe_offset=j]
        j += 1


@always_inline
def _require(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(msg)
