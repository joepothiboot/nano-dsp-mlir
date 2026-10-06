from .layout import TensorLike, TensorView


struct Tensor[dtype: DType](Copyable, Movable, TensorLike):
    comptime element_dtype = Self.dtype

    var data: List[Scalar[Self.dtype]]
    var shape: List[Int]

    def __init__(out self, var shape: List[Int], fill: Scalar[Self.dtype] = 0):
        self.data = List[Scalar[Self.dtype]](length=_product(shape), fill=fill)
        self.shape = shape^

    def __init__(
        out self, var shape: List[Int], var values: List[Scalar[Self.dtype]]
    ) raises:
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
        return self.data[i]

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
        if self.rank() != 2:
            raise Error("Tensor.view: tensor must be rank 2")

        return TensorView(Span(self.data), self.shape[0], self.shape[1])


def _product(shape: List[Int]) -> Int:
    var n = 1

    for d in shape:
        n *= d

    return n
