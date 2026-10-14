from nanodsp.kernels import _matmul_edge, _matmul_micro
from nanodsp.layout import TensorLike

comptime F32 = DType.float32
comptime F32_WIDTH = 4
comptime TILE_M = 4
comptime TILE_N = 16


struct RawMatrix[dtype: DType](ImplicitlyCopyable, TensorLike):
    comptime element_dtype = Self.dtype

    var _ptr: Pointer[Scalar[Self.dtype], MutUntrackedOrigin]
    var _rows: Int
    var _cols: Int

    def __init__(
        out self,
        ptr: Pointer[Scalar[Self.dtype], MutUntrackedOrigin],
        rows: Int,
        cols: Int,
    ):
        self._ptr = ptr
        self._rows = rows
        self._cols = cols

    @always_inline
    def rows(self) -> Int:
        return self._rows

    @always_inline
    def cols(self) -> Int:
        return self._cols

    @always_inline
    def load[width: Int](self, i: Int, j: Int) -> SIMD[Self.dtype, width]:
        return self._ptr.unsafe_load[width=width](i * self._cols + j)

    @always_inline
    def store[width: Int](mut self, i: Int, j: Int, v: SIMD[Self.dtype, width]):
        self._ptr.unsafe_store(i * self._cols + j, v)


@export("nd_add_f32")
def nd_add_f32(
    a: Pointer[Float32, MutUntrackedOrigin],
    b: Pointer[Float32, MutUntrackedOrigin],
    c: Pointer[Float32, MutUntrackedOrigin],
    n: Int32,
) abi("C"):
    var count = Int(n)
    var i = 0

    while i + F32_WIDTH <= count:
        c.unsafe_store(
            i,
            a.unsafe_load[width=F32_WIDTH](i)
            + b.unsafe_load[width=F32_WIDTH](i),
        )
        i += F32_WIDTH

    while i < count:
        c[unsafe_offset=i] = a[unsafe_offset=i] + b[unsafe_offset=i]
        i += 1


@export("nd_matmul_f32")
def nd_matmul_f32(
    a: Pointer[Float32, MutUntrackedOrigin],
    b: Pointer[Float32, MutUntrackedOrigin],
    c: Pointer[Float32, MutUntrackedOrigin],
    m: Int32,
    k: Int32,
    n: Int32,
) abi("C"):
    var rows = Int(m)
    var depth = Int(k)
    var cols = Int(n)
    var va = RawMatrix[F32](a, rows, depth)
    var vb = RawMatrix[F32](b, depth, cols)
    var vc = RawMatrix[F32](c, rows, cols)

    for i0 in range(0, rows, TILE_M):
        var h = min(TILE_M, rows - i0)

        for j0 in range(0, cols, TILE_N):
            var w = min(TILE_N, cols - j0)

            if h == TILE_M and w == TILE_N:
                _matmul_micro[F32, TILE_M, TILE_N, F32_WIDTH](
                    va, vb, vc, i0, j0, depth
                )
            else:
                _matmul_edge[F32, F32_WIDTH](va, vb, vc, i0, j0, h, w, depth)
