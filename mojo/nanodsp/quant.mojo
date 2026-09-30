"""int8 quantized matmul with the same semantics as `dsp.qmatmul`.

acc[i, j] = sum_k (a[i, k] - lhs_zp) * (b[k, j] - rhs_zp) in i32, then
out = clamp(out_zp + ((acc * multiplier + 2^(s-1)) >> s), -128, 127) with
s = 31 + shift, in i64. The arithmetic right shift makes that round half up
(-100.5 -> -100), TFLite's single-rounding convention. Integer arithmetic is
exact, so the SIMD kernel, the MLIR lowering and the C++ reference agree bit
for bit.
"""

from std.sys import simd_width_of

from .tensor import Tensor


comptime I8 = DType.int8
comptime I32 = DType.int32


struct QuantParams(Copyable, Movable):
    """Per-tensor quantization of a qmatmul; see dsp.qmatmul in DSPOps.td."""

    var lhs_zp: Int32
    var rhs_zp: Int32
    var multiplier: Int32
    """Q0.31 fixed point, normalized to [2^30, 2^31)."""
    var shift: Int32
    """Right shift in [0, 31]."""
    var out_zp: Int32

    def __init__(
        out self,
        lhs_zp: Int32,
        rhs_zp: Int32,
        multiplier: Int32,
        shift: Int32,
        out_zp: Int32,
    ):
        self.lhs_zp = lhs_zp
        self.rhs_zp = rhs_zp
        self.multiplier = multiplier
        self.shift = shift
        self.out_zp = out_zp


def qmatmul(
    a: Tensor[I8], b: Tensor[I8], q: QuantParams
) raises -> Tensor[I8]:
    """Quantized `(M x K) * (K x N) -> (M x N)`, like `dsp.qmatmul`.

    Uses i-k-j order: each row of `b` is widened to i32 lanes and
    multiply-accumulated into one i32 row of accumulators, which is then
    requantized to i8.

    Raises:
        If the operands are not rank 2, the inner dimensions differ, or the
        quantization parameters are out of the ranges the dialect verifies.
    """
    _require(a.rank() == 2 and b.rank() == 2, "qmatmul: operands must be rank 2")
    var m = a.shape[0]
    var k = a.shape[1]
    var n = b.shape[1]
    _require(b.shape[0] == k, "qmatmul: inner dimensions differ")
    _require(k <= 33025, "qmatmul: K can overflow the i32 accumulator")
    for zp in [q.lhs_zp, q.rhs_zp, q.out_zp]:
        _require(zp >= -128 and zp <= 127, "qmatmul: zero point out of range")
    _require(q.multiplier >= (1 << 30), "qmatmul: multiplier not normalized")
    _require(q.shift >= 0 and q.shift <= 31, "qmatmul: shift out of range")

    comptime width = simd_width_of[I32]()
    var result = Tensor[I8]([m, n])
    var acc = List[Int32](length=n, fill=0)
    var pacc = acc.unsafe_ptr()
    var pb = b.data.unsafe_ptr()
    var rzp = SIMD[I32, width](q.rhs_zp)
    for i in range(m):
        for j in range(n):
            pacc[unsafe_offset=j] = 0
        for kk in range(k):
            var av = a.data[i * k + kk].cast[I32]() - q.lhs_zp
            var va = SIMD[I32, width](av)
            var prow = pb.unsafe_offset(kk * n)
            var j = 0
            while j + width <= n:
                var w = prow.unsafe_load[width=width](j).cast[I32]() - rzp
                pacc.unsafe_store(j, pacc.unsafe_load[width=width](j) + va * w)
                j += width
            while j < n:
                var w = prow[unsafe_offset=j].cast[I32]() - q.rhs_zp
                pacc[unsafe_offset=j] = pacc[unsafe_offset=j] + av * w
                j += 1
        for j in range(n):
            result.data[i * n + j] = requantize(pacc[unsafe_offset=j], q)
    return result^


@always_inline
def requantize(acc: Int32, q: QuantParams) -> Int8:
    """`clamp(out_zp + ((acc * multiplier + 2^(s-1)) >> s), -128, 127)`."""
    var s = Int64(31) + q.shift.cast[DType.int64]()
    var scaled = (
        acc.cast[DType.int64]() * q.multiplier.cast[DType.int64]()
        + (Int64(1) << (s - 1))
    ) >> s
    var v = q.out_zp.cast[DType.int64]() + scaled
    if v < -128:
        v = -128
    if v > 127:
        v = 127
    return v.cast[I8]()


@always_inline
def _require(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(msg)
