//===- DSPToLinalg.cpp - Lower 'dsp' to linalg.generic --------------------===//
//
// L1 (dsp, value semantics, whole-array) -> L2 (linalg on tensors, DPS).
// Invariants established here and relied upon by Stage 3:
//   * every result is produced by exactly one linalg.generic
//   * every generic is in destination-passing style with a tensor.empty dest
//   * reductions have their destination explicitly zero-filled
//   * all shapes are static, element type is f32
//
//===----------------------------------------------------------------------===//

#include "nanodsp/Conversion/DSPToLinalg/DSPToLinalg.h"
#include "nanodsp/Dialect/DSP/IR/DSPOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace nanodsp {
#define GEN_PASS_DEF_CONVERTDSPTOLINALG
#include "nanodsp/Conversion/Passes.h.inc"
} // namespace nanodsp
} // namespace mlir

using namespace mlir;
using namespace mlir::nanodsp;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// An uninitialized destination tensor. Valid only when every element of the
/// result is written unconditionally (true for elementwise ops).
static Value createEmptyDest(OpBuilder &b, Location loc,
                             RankedTensorType type) {
  return b.create<tensor::EmptyOp>(loc, type.getShape(),
                                   type.getElementType());
}

/// A zero-initialized destination tensor. Required for reductions, where the
/// generic accumulates into the destination.
static Value createZeroDest(OpBuilder &b, Location loc,
                            RankedTensorType type) {
  Value empty = createEmptyDest(b, loc, type);
  Value zero = b.create<arith::ConstantOp>(
      loc, b.getZeroAttr(type.getElementType()));
  return b.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty})
      .getResult(0);
}

static SmallVector<utils::IteratorType> parallelIterators(int64_t n) {
  return SmallVector<utils::IteratorType>(n, utils::IteratorType::parallel);
}

//===----------------------------------------------------------------------===//
// dsp.add -> linalg.generic (rank-N, all-parallel, identity maps)
//===----------------------------------------------------------------------===//

namespace {
struct AddOpLowering : public OpConversionPattern<AddOp> {
  using OpConversionPattern<AddOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    RankedTensorType resTy = op.getResult().getType();
    const int64_t rank = resTy.getRank();

    Value dest = createEmptyDest(rewriter, loc, resTy);
    AffineMap id = rewriter.getMultiDimIdentityMap(rank);

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resTy},
        /*inputs=*/ValueRange{adaptor.getLhs(), adaptor.getRhs()},
        /*outputs=*/ValueRange{dest},
        /*indexingMaps=*/ArrayRef<AffineMap>{id, id, id},
        /*iteratorTypes=*/parallelIterators(rank),
        [](OpBuilder &nested, Location nestedLoc, ValueRange args) {
          Value sum = nested.create<arith::AddFOp>(nestedLoc, args[0], args[1]);
          nested.create<linalg::YieldOp>(nestedLoc, sum);
        });

    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// dsp.relu -> linalg.generic with arith.maximumf(x, 0.0)
//===----------------------------------------------------------------------===//

struct ReluOpLowering : public OpConversionPattern<ReluOp> {
  using OpConversionPattern<ReluOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    RankedTensorType resTy = op.getResult().getType();
    const int64_t rank = resTy.getRank();

    Value dest = createEmptyDest(rewriter, loc, resTy);
    // Hoisted out of the region: loop-invariant, and keeps the generic's body
    // to a single op so Stage 3's vectorizer sees the cleanest possible IR.
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(resTy.getElementType()));

    AffineMap id = rewriter.getMultiDimIdentityMap(rank);

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resTy},
        /*inputs=*/ValueRange{adaptor.getInput()},
        /*outputs=*/ValueRange{dest},
        /*indexingMaps=*/ArrayRef<AffineMap>{id, id},
        /*iteratorTypes=*/parallelIterators(rank),
        [zero](OpBuilder &nested, Location nestedLoc, ValueRange args) {
          // maximumf, not maxnumf: NaN must propagate (numpy.maximum semantics).
          Value r = nested.create<arith::MaximumFOp>(nestedLoc, args[0], zero);
          nested.create<linalg::YieldOp>(nestedLoc, r);
        });

    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// dsp.matmul -> linalg.generic, iterators (m, n, k)
//===----------------------------------------------------------------------===//

struct MatmulOpLowering : public OpConversionPattern<MatmulOp> {
  using OpConversionPattern<MatmulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    RankedTensorType resTy = op.getResult().getType();

    Value dest = createZeroDest(rewriter, loc, resTy);

    AffineExpr m, n, k;
    bindDims(ctx, m, n, k);
    SmallVector<AffineMap> maps = {
        AffineMap::get(3, 0, {m, k}, ctx), // lhs  (M x K)
        AffineMap::get(3, 0, {k, n}, ctx), // rhs  (K x N)
        AffineMap::get(3, 0, {m, n}, ctx), // out  (M x N)
    };

    SmallVector<utils::IteratorType> iters = {utils::IteratorType::parallel,
                                              utils::IteratorType::parallel,
                                              utils::IteratorType::reduction};

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resTy},
        /*inputs=*/ValueRange{adaptor.getLhs(), adaptor.getRhs()},
        /*outputs=*/ValueRange{dest}, maps, iters,
        [](OpBuilder &nested, Location nestedLoc, ValueRange args) {
          Value prod = nested.create<arith::MulFOp>(nestedLoc, args[0], args[1]);
          Value acc = nested.create<arith::AddFOp>(nestedLoc, args[2], prod);
          nested.create<linalg::YieldOp>(nestedLoc, acc);
        });

    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// dsp.conv2d -> linalg.generic, iterators (n, oh, ow, f, kh, kw, c)
//
// input  map: (n, oh*SH + kh*DH, ow*SW + kw*DW, c)
// filter map: (kh, kw, c, f)
// output map: (n, oh, ow, f)
//
// Strides and dilations are folded directly into the affine expressions, so
// they cost nothing at runtime and the op stays a single perfectly-nested
// structured op that Stage 3 can tile.
//===----------------------------------------------------------------------===//

struct Conv2DOpLowering : public OpConversionPattern<Conv2DOp> {
  using OpConversionPattern<Conv2DOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Conv2DOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    RankedTensorType resTy = op.getResult().getType();

    ArrayRef<int64_t> strides = op.getStrides();
    ArrayRef<int64_t> dilations = op.getDilations();

    Value dest = createZeroDest(rewriter, loc, resTy);

    AffineExpr n, oh, ow, f, kh, kw, c;
    bindDims(ctx, n, oh, ow, f, kh, kw, c);

    AffineExpr ih = oh * strides[0] + kh * dilations[0];
    AffineExpr iw = ow * strides[1] + kw * dilations[1];

    SmallVector<AffineMap> maps = {
        AffineMap::get(7, 0, {n, ih, iw, c}, ctx),  // input  NHWC
        AffineMap::get(7, 0, {kh, kw, c, f}, ctx),  // filter HWCF
        AffineMap::get(7, 0, {n, oh, ow, f}, ctx),  // output NHWF
    };

    SmallVector<utils::IteratorType> iters = {
        utils::IteratorType::parallel,  // n
        utils::IteratorType::parallel,  // oh
        utils::IteratorType::parallel,  // ow
        utils::IteratorType::parallel,  // f
        utils::IteratorType::reduction, // kh
        utils::IteratorType::reduction, // kw
        utils::IteratorType::reduction, // c
    };

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resTy},
        /*inputs=*/ValueRange{adaptor.getInput(), adaptor.getFilter()},
        /*outputs=*/ValueRange{dest}, maps, iters,
        [](OpBuilder &nested, Location nestedLoc, ValueRange args) {
          Value prod = nested.create<arith::MulFOp>(nestedLoc, args[0], args[1]);
          Value acc = nested.create<arith::AddFOp>(nestedLoc, args[2], prod);
          nested.create<linalg::YieldOp>(nestedLoc, acc);
        });

    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct ConvertDSPToLinalgPass
    : public mlir::nanodsp::impl::ConvertDSPToLinalgBase<ConvertDSPToLinalgPass> {
  using mlir::nanodsp::impl::ConvertDSPToLinalgBase<
      ConvertDSPToLinalgPass>::ConvertDSPToLinalgBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    ConversionTarget target(*ctx);
    target.addIllegalDialect<DSPDialect>();
    target.addLegalDialect<arith::ArithDialect, linalg::LinalgDialect,
                           tensor::TensorDialect, func::FuncDialect>();
    target.addLegalOp<ModuleOp>();

    RewritePatternSet patterns(ctx);
    populateDSPToLinalgPatterns(patterns);

    // Full conversion: nothing from 'dsp' may survive. If a future op is added
    // without a pattern, this fails loudly instead of silently leaking L1 IR
    // into the Stage 3 schedule.
    if (failed(applyFullConversion(getOperation(), target,
                                   std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void mlir::nanodsp::populateDSPToLinalgPatterns(RewritePatternSet &patterns) {
  patterns.add<AddOpLowering, ReluOpLowering, MatmulOpLowering,
               Conv2DOpLowering>(patterns.getContext());
}