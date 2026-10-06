trait TensorLike:
    comptime element_dtype: DType

    def rows(self) -> Int:
        ...

    def cols(self) -> Int:
        ...

    def load[
        width: Int
    ](self, i: Int, j: Int) -> SIMD[Self.element_dtype, width]:
        ...

    def store[
        width: Int
    ](mut self, i: Int, j: Int, v: SIMD[Self.element_dtype, width]):
        ...


struct TensorView[mut: Bool, //, dtype: DType, origin: Origin[mut=mut]](
    ImplicitlyCopyable, TensorLike
):
    comptime element_dtype = Self.dtype

    var _ptr: Pointer[Scalar[Self.dtype], Self.origin]
    var _rows: Int
    var _cols: Int
    var _row_stride: Int

    def __init__(
        out self,
        span: Span[Scalar[Self.dtype], Self.origin],
        rows: Int,
        cols: Int,
        row_stride: Int,
    ) raises:
        if rows < 0 or cols < 0 or row_stride < cols:
            raise Error(
                "TensorView: need rows, cols >= 0 and row_stride >= cols"
            )

        if rows > 0 and cols > 0 and (rows - 1) * row_stride + cols > len(span):
            raise Error("TensorView: shape and stride overrun the buffer")

        self._ptr = span.unsafe_ptr()
        self._rows = rows
        self._cols = cols
        self._row_stride = row_stride

    def __init__(
        out self,
        span: Span[Scalar[Self.dtype], Self.origin],
        rows: Int,
        cols: Int,
    ) raises:
        self = Self(span, rows, cols, cols)

    @always_inline
    def rows(self) -> Int:
        return self._rows

    @always_inline
    def cols(self) -> Int:
        return self._cols

    @always_inline
    def row_stride(self) -> Int:
        return self._row_stride

    @always_inline
    def load[width: Int](self, i: Int, j: Int) -> SIMD[Self.dtype, width]:
        return self._ptr.unsafe_load[width=width](i * self._row_stride + j)

    @always_inline
    def store[width: Int](mut self, i: Int, j: Int, v: SIMD[Self.dtype, width]):
        comptime assert Self.mut, "store through a read-only TensorView"
        self._ptr.unsafe_mut_cast[True]().unsafe_store(
            i * self._row_stride + j, v
        )

    @always_inline
    def readonly(
        self,
    ) -> TensorView[Self.dtype, Origin[mut=False](Self.origin)]:
        return rebind[TensorView[Self.dtype, Origin[mut=False](Self.origin)]](
            self
        )

    @always_inline
    def __getitem__(self, i: Int, j: Int) -> Scalar[Self.dtype]:
        return self.load[1](i, j)

    def tile(self, i0: Int, j0: Int, h: Int, w: Int) raises -> Self:
        if (
            i0 < 0
            or j0 < 0
            or h < 0
            or w < 0
            or i0 + h > self._rows
            or j0 + w > self._cols
        ):
            raise Error("TensorView.tile: tile is out of bounds")

        var t = self
        t._ptr = self._ptr.unsafe_offset(i0 * self._row_stride + j0)
        t._rows = h
        t._cols = w

        return t
