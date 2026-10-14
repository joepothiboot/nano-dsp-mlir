from nanodsp.quant import QuantParams, requantize

comptime I32 = DType.int32
comptime I32_WIDTH = 32


@export("nd_qmatmul_i8")
def nd_qmatmul_i8(
    a: Pointer[Int8, MutUntrackedOrigin],
    b: Pointer[Int8, MutUntrackedOrigin],
    c: Pointer[Int8, MutUntrackedOrigin],
    acc: Pointer[Int32, MutUntrackedOrigin],
    m: Int32,
    k: Int32,
    n: Int32,
    lhs_zp: Int32,
    rhs_zp: Int32,
    multiplier: Int32,
    shift: Int32,
    out_zp: Int32,
) abi("C"):
    var q = QuantParams(lhs_zp, rhs_zp, multiplier, shift, out_zp)
    var rows = Int(m)
    var depth = Int(k)
    var cols = Int(n)
    var rzp = SIMD[I32, I32_WIDTH](rhs_zp)

    for i in range(rows):
        for j in range(cols):
            acc[unsafe_offset=j] = 0

        for kk in range(depth):
            var av = a[unsafe_offset=i * depth + kk].cast[I32]() - lhs_zp
            var va = SIMD[I32, I32_WIDTH](av)
            var prow = b.unsafe_offset(kk * cols)
            var j = 0

            while j + I32_WIDTH <= cols:
                var w = prow.unsafe_load[width=I32_WIDTH](j).cast[I32]() - rzp
                acc.unsafe_store(
                    j, acc.unsafe_load[width=I32_WIDTH](j) + va * w
                )
                j += I32_WIDTH

            while j < cols:
                var w = prow[unsafe_offset=j].cast[I32]() - rhs_zp
                acc[unsafe_offset=j] = acc[unsafe_offset=j] + av * w
                j += 1

        for j in range(cols):
            c[unsafe_offset=i * cols + j] = requantize(acc[unsafe_offset=j], q)
