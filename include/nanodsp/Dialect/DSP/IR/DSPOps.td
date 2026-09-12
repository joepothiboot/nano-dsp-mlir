#ifndef NANODSP_DIALECT_DSP_IR_DSPOPS_TD
#define NANODSP_DIALECT_DSP_IR_DSPOPS_TD

include "nanodsp/Dialect/DSP/IR/DSPBase.td"

//===----------------------------------------------------------------------===//
// dsp.add
//===----------------------------------------------------------------------===//

def DSP_AddOp : DSP_Op<"add", [Pure, SameOperandsAndResultType]> {
  let summary = "Elementwise addition of two identically shaped f32 tensors.";
  let description = [{
    No implicit broadcasting. The frontend materializes rank-expanding or
    splatting behaviour explicitly (see Stage 2 deviation note #2).

    ```mlir
    %0 = dsp.add %a, %b : tensor<2x3xf32>
    ```
  }];

  let arguments = (ins DSP_Tensor:$lhs, DSP_Tensor:$rhs);
  let results = (outs DSP_Tensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($result)
  }];

  let builders = [
    OpBuilder<(ins "::mlir::Value":$lhs, "::mlir::Value":$rhs), [{
      build($_builder, $_state, lhs.getType(), lhs, rhs);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// dsp.relu
//===----------------------------------------------------------------------===//

def DSP_ReluOp : DSP_Op<"relu", [Pure, SameOperandsAndResultType]> {
  let summary = "Elementwise max(x, 0) with NaN propagation.";
  let description = [{
    Semantics match numpy.maximum(x, 0.0): if x is NaN the result is NaN.
    This lowers to arith.maximumf, not arith.maxnumf (which would return the
    non-NaN operand and silently turn NaN into 0.0).

    ```mlir
    %0 = dsp.relu %a : tensor<1x6xf32>
    ```
  }];

  let arguments = (ins DSP_Tensor:$input);
  let results = (outs DSP_Tensor:$result);

  let assemblyFormat = [{
    $input attr-dict `:` type($result)
  }];

  // relu is idempotent: relu(relu(x)) -> relu(x)
  let hasCanonicalizeMethod = 1;

  let builders = [
    OpBuilder<(ins "::mlir::Value":$input), [{
      build($_builder, $_state, input.getType(), input);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// dsp.matmul
//===----------------------------------------------------------------------===//

def DSP_MatmulOp : DSP_Op<"matmul", [Pure]> {
  let summary = "Row-major 2-D matrix multiply: (MxK) * (KxN) -> (MxN).";
  let description = [{
    ```mlir
    %0 = dsp.matmul %a, %b : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
    ```
  }];

  let arguments = (ins DSP_TensorOfRank<2>:$lhs, DSP_TensorOfRank<2>:$rhs);
  let results = (outs DSP_TensorOfRank<2>:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "::mlir::Value":$lhs, "::mlir::Value":$rhs), [{
      build($_builder, $_state, lhs.getType(), lhs, rhs);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// dsp.conv2d
//===----------------------------------------------------------------------===//

def DSP_Conv2DOp : DSP_Op<"conv2d", [Pure]> {
  let summary = "2-D cross-correlation, NHWC input x HWCF filter -> NHWF output.";
  let description = [{
    Layouts are fixed: input : tensor, filter : tensor, result : tensor.
    This is cross-correlation (no kernel flip), matching
    `linalg.conv_2d_nhwc_hwcf` and every ML framework.

    No implicit padding -- 'valid' only. Padding is the frontend's job
    (`tensor.pad`), which keeps it hoistable out of the tiled loop nest in
    Stage 3.

    OH = (H - (KH - 1) * dilation_h - 1) / stride_h + 1
    OW = (W - (KW - 1) * dilation_w - 1) / stride_w + 1

    ```mlir
    %0 = dsp.conv2d %in, %f {strides = array<i64: 1, 1>,
                             dilations = array<i64: 1, 1>}
       : (tensor<1x4x4x1xf32>, tensor<3x3x1x1xf32>) -> tensor<1x2x2x1xf32>
    ```
  }];

  let arguments = (ins
    DSP_TensorOfRank<4>:$input,
    DSP_TensorOfRank<4>:$filter,
    DefaultValuedAttr<DenseI64ArrayAttr, "{1, 1}">:$strides,
    DefaultValuedAttr<DenseI64ArrayAttr, "{1, 1}">:$dilations
  );
  let results = (outs DSP_TensorOfRank<4>:$result);

  let assemblyFormat = [{
    $input `,` $filter attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "::mlir::Value":$input,
                   "::mlir::Value":$filter,
                   "::llvm::ArrayRef<int64_t>":$strides,
                   "::llvm::ArrayRef<int64_t>":$dilations), [{
      build($_builder, $_state, input.getType(), input, filter, strides,
            dilations);
    }]>
  ];
}

#endif // NANODSP_DIALECT_DSP_IR_DSPOPS_TD
