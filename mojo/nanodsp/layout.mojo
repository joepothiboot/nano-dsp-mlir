"""Rank-2 tensor interface and a borrowed, strided view.

`TensorLike` is the contract the generic kernels are written against: a
shape, and SIMD loads and stores at `(row, col)`. `Tensor` (owned,
contiguous) and `TensorView` (borrowed, row-strided) both conform, so one
kernel source is instantiated for either at compile time, with no dynamic
dispatch.

`TensorView` carries the origin of the buffer it points into. The compiler
uses that origin to keep the buffer alive and un-moved for as long as the
view is in use, and to tell a mutable view from a read-only one at compile
time.
"""


trait TensorLike:
    """A rank-2, row-major tensor addressed by `(row, col)`.

    Only the innermost dimension is assumed contiguous: `load[width](i, j)`
    reads elements `(i, j) .. (i, j + width - 1)`. Rows may be any distance
    apart. Loads and stores are unchecked; callers keep `j + width <=
    cols()`.
    """

    comptime element_dtype: DType
    """Element type. Named apart from the `dtype` struct parameter of the
    conforming types, because a struct parameter cannot satisfy a trait's
    `comptime` requirement and cannot be redeclared under the same name."""

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
    """A non-owning rank-2 window into someone else's buffer.

    Element `(i, j)` lives at `ptr + i * row_stride + j`. A view of a whole
    contiguous tensor has `row_stride == cols`; a tile of it keeps the
    parent's stride, so its rows are not adjacent in memory.

    Copying a view copies four words, never the elements. `origin` ties the
    view to the buffer it was made from: the compiler rejects moving or
    destroying that buffer while the view is still used, and `store` only
    compiles when the origin is mutable.

    Parameters:
        mut: Whether the view can write (inferred from `origin`).
        dtype: Element type.
        origin: The origin of the borrowed buffer.
    """

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
        """Views `span` as `rows x cols` with rows `row_stride` apart.

        Raises:
            If a dimension is negative, `row_stride < cols`, or the last
            row would run past the end of `span`.
        """
        if rows < 0 or cols < 0 or row_stride < cols:
            raise Error(
                "TensorView: need rows, cols >= 0 and row_stride >= cols"
            )
        if rows > 0 and cols > 0 and (rows - 1) * row_stride + cols > len(
            span
        ):
            raise Error("TensorView: shape and stride overrun the buffer")
        self._ptr = span.unsafe_ptr()
        self._rows = rows
        self._cols = cols
        self._row_stride = row_stride

    def __init__(
        out self, span: Span[Scalar[Self.dtype], Self.origin], rows: Int, cols: Int
    ) raises:
        """Views `span` as a contiguous `rows x cols` matrix.

        Raises:
            If `rows * cols` elements do not fit in `span`.
        """
        self = Self(span, rows, cols, cols)

    @always_inline
    def rows(self) -> Int:
        return self._rows

    @always_inline
    def cols(self) -> Int:
        return self._cols

    @always_inline
    def row_stride(self) -> Int:
        """Distance between the starts of consecutive rows, in elements."""
        return self._row_stride

    @always_inline
    def load[width: Int](self, i: Int, j: Int) -> SIMD[Self.dtype, width]:
        return self._ptr.unsafe_load[width=width](i * self._row_stride + j)

    @always_inline
    def store[width: Int](mut self, i: Int, j: Int, v: SIMD[Self.dtype, width]):
        """Writes `v` at `(i, j)`; only compiles for a mutable view."""
        comptime assert Self.mut, "store through a read-only TensorView"
        self._ptr.unsafe_mut_cast[True]().unsafe_store(
            i * self._row_stride + j, v
        )

    @always_inline
    def readonly(
        self,
    ) -> TensorView[Self.dtype, Origin[mut=False](Self.origin)]:
        """The same view with its origin demoted to read-only.

        Useful when a mutable view (say, of a local `var`) is passed next to
        a read-only borrow of the same buffer: the argument exclusivity
        check rejects a mutable-origin view there, but accepts this one.
        """
        return rebind[TensorView[Self.dtype, Origin[mut=False](Self.origin)]](
            self
        )

    @always_inline
    def __getitem__(self, i: Int, j: Int) -> Scalar[Self.dtype]:
        return self.load[1](i, j)

    def tile(self, i0: Int, j0: Int, h: Int, w: Int) raises -> Self:
        """The `h x w` sub-view whose top-left element is `(i0, j0)`.

        The tile borrows from the same buffer with the same origin and keeps
        this view's row stride.

        Raises:
            If the tile does not lie inside this view.
        """
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
