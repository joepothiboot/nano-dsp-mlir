"""A minimal owned, contiguous, row-major tensor used by the kernels."""


struct Tensor[dtype: DType](Copyable, Movable):
    """Owned row-major storage plus a runtime shape.

    Deliberately small: no strides, no views, no broadcasting. The kernels
    only need contiguous buffers, and the `dsp` dialect only has static
    shapes.

    Parameters:
        dtype: Element type.
    """

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


def _product(shape: List[Int]) -> Int:
    var n = 1
    for d in shape:
        n *= d
    return n
