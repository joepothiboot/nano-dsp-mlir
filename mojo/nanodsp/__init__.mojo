"""SIMD Mojo kernels with the same semantics as the `dsp` MLIR dialect.

Each kernel mirrors one `dsp` op (see include/nanodsp/Dialect/DSP/IR/DSPOps.td):
same layouts, no implicit broadcasting, no implicit padding, and NaN
propagation in `relu`. `qmatmul` is the int8 quantized matmul. The tests check them against the same golden values as
test/Integration/, so the Mojo library and the MLIR pipeline act as oracles
for each other.
"""

from .tensor import Tensor
from .kernels import add, relu, matmul, matmul_tiled, conv2d
from .layout import TensorLike, TensorView
from .quant import QuantParams, qmatmul, requantize
