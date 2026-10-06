from std.sys import simd_width_of

from .constants import I8_MAX, I8_MIN, QUANT_SHIFT_BASE
from .tensor import Tensor


comptime I8 = DType.int8
comptime I32 = DType.int32


struct QuantParams(Copyable, Movable):
    var lhs_zp: Int32
    var rhs_zp: Int32
    var multiplier: Int32
    var shift: Int32
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


def qmatmul(a: Tensor[I8], b: Tensor[I8], q: QuantParams) raises -> Tensor[I8]:
    _require(
        a.rank() == 2 and b.rank() == 2, "qmatmul: operands must be rank 2"
    )
    var m = a.shape[0]
    var k = a.shape[1]
    var n = b.shape[1]
    _require(b.shape[0] == k, "qmatmul: inner dimensions differ")
    _require(k <= 33025, "qmatmul: K can overflow the i32 accumulator")

    for zp in [q.lhs_zp, q.rhs_zp, q.out_zp]:
        _require(
            zp >= I8_MIN and zp <= I8_MAX, "qmatmul: zero point out of range"
        )

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
    var s = Int64(QUANT_SHIFT_BASE) + q.shift.cast[DType.int64]()
    var scaled = (
        acc.cast[DType.int64]() * q.multiplier.cast[DType.int64]()
        + (Int64(1) << (s - 1))
    ) >> s
    var v = q.out_zp.cast[DType.int64]() + scaled

    if v < I8_MIN:
        v = I8_MIN

    if v > I8_MAX:
        v = I8_MAX

    return v.cast[I8]()


@always_inline
def _require(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(msg)
