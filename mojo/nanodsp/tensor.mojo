"""A minimal owned, contiguous, row-major tensor used by the kernels."""

from .layout import TensorLike, TensorView


struct Tensor[dtype: DType](Copyable, Movable, TensorLike):
    """Owned row-major storage plus a runtime shape.

    Deliberately small: contiguous, no broadcasting. Strided access goes
    through `view()`, which borrows the storage as a `TensorView`. The
    `TensorLike` methods (`rows`, `cols`, `load`, `store`) treat the tensor
    as a rank-2 matrix and assume `rank() == 2`.

    Parameters:
        dtype: Element type.
    """

    comptime element_dtype = Self.dtype

    var data: List[Scalar[Self.dtype]]
    var shape: List[Int]

    def __init__(out self, var shape: List[Int], fill: Scalar[Self.dtype] = 0):
        """Allocates a tensor of `shape` with every element set to `fill`."""
        self.data = List[Scalar[Self.dtype]](length=_product(shape), fill=fill)
        self.shape = shape^

    def __init__(
        out self, var shape: List[Int], var values: List[Scalar[Self.dtype]]
    ) raises:
        """Wraps `values` (row-major) as a tensor of `shape`.

        Raises:
            If `len(values)` does not equal the product of `shape`.
        """
        var n = _product(shape)
        if n != len(values):
            raise Error(
                "Tensor: got "
                + String(len(values))
                + " values for a shape of "
                + String(n)
                + " elements"
            )
        self.data = values^
        self.shape = shape^

    @always_inline
    def rank(self) -> Int:
        return len(self.shape)

    @always_inline
    def numel(self) -> Int:
        return len(self.data)

    @always_inline
    def __getitem__(self, i: Int) -> Scalar[Self.dtype]:
        """Returns the element at flat (row-major) index `i`."""
        return self.data[i]

    # --- TensorLike (rank 2) ------------------------------------------------

    @always_inline
    def rows(self) -> Int:
        return self.shape[0]

    @always_inline
    def cols(self) -> Int:
        return self.shape[1]

    @always_inline
    def load[width: Int](self, i: Int, j: Int) -> SIMD[Self.dtype, width]:
        return self.data.unsafe_ptr().unsafe_load[width=width](
            i * self.shape[1] + j
        )

    @always_inline
    def store[width: Int](mut self, i: Int, j: Int, v: SIMD[Self.dtype, width]):
        self.data.unsafe_ptr().unsafe_store(i * self.shape[1] + j, v)

    def view(ref self) raises -> TensorView[Self.dtype, origin_of(self.data)]:
        """Borrows this rank-2 tensor as a `TensorView`.

        The view is mutable when `self` is (a `var` or a `mut` argument) and
        read-only when `self` is a read-only borrow. Either way it cannot
        outlive the tensor, and the tensor cannot be moved while the view
        is in use.

        Raises:
            If the tensor is not rank 2.
        """
        if self.rank() != 2:
            raise Error("Tensor.view: tensor must be rank 2")
        return TensorView(Span(self.data), self.shape[0], self.shape[1])


def _product(shape: List[Int]) -> Int:
    var n = 1
    for d in shape:
        n *= d
    return n
