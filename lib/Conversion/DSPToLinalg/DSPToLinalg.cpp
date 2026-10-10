#include "nanodsp/Conversion/DSPToLinalg/DSPToLinalg.h"
#include "nanodsp/Dialect/DSP/IR/DSPConstants.h"
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

namespace mlir::nanodsp {
#define GEN_PASS_DEF_CONVERTDSPTOLINALG
#include "nanodsp/Conversion/Passes.h.inc"
}

using namespace mlir;
using namespace mlir::nanodsp;

static Value createEmptyDest(OpBuilder &b, Location loc,
                             RankedTensorType type) {
  return tensor::EmptyOp::create(b, loc, type.getShape(),
                                 type.getElementType());
}

static Value createZeroDest(OpBuilder &b, Location loc, RankedTensorType type) {
  Value empty = createEmptyDest(b, loc, type);
  Value zero =
      arith::ConstantOp::create(b, loc, b.getZeroAttr(type.getElementType()));
  return linalg::FillOp::create(b, loc, ValueRange{zero}, ValueRange{empty})
      .getResult(0);
}

static SmallVector<utils::IteratorType> parallelIterators(int64_t n) {
  return SmallVector<utils::IteratorType>(n, utils::IteratorType::parallel);
}

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

    auto generic = linalg::GenericOp::create(
        rewriter, loc, TypeRange{resTy},
        ValueRange{adaptor.getLhs(), adaptor.getRhs()}, ValueRange{dest},
        ArrayRef<AffineMap>{id, id, id}, parallelIterators(rank),
        [](OpBuilder &nested, Location nestedLoc, ValueRange args) {
          Value sum =
              arith::AddFOp::create(nested, nestedLoc, args[0], args[1]);
          linalg::YieldOp::create(nested, nestedLoc, sum);
        });

    rewriter.replaceOp(op, generic.getResults());

    return success();
  }
};

struct ReluOpLowering : public OpConversionPattern<ReluOp> {
  using OpConversionPattern<ReluOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    RankedTensorType resTy = op.getResult().getType();
    const int64_t rank = resTy.getRank();

    Value dest = createEmptyDest(rewriter, loc, resTy);
    Value zero = arith::ConstantOp::create(
        rewriter, loc, rewriter.getZeroAttr(resTy.getElementType()));

    AffineMap id = rewriter.getMultiDimIdentityMap(rank);

    auto generic = linalg::GenericOp::create(
        rewriter, loc, TypeRange{resTy}, ValueRange{adaptor.getInput()},
        ValueRange{dest}, ArrayRef<AffineMap>{id, id}, parallelIterators(rank),
        [zero](OpBuilder &nested, Location nestedLoc, ValueRange args) {
          Value r = arith::MaximumFOp::create(nested, nestedLoc, args[0], zero);
          linalg::YieldOp::create(nested, nestedLoc, r);
        });

    rewriter.replaceOp(op, generic.getResults());

    return success();
  }
};

struct MatmulOpLowering : public OpConversionPattern<MatmulOp> {
  using OpConversionPattern<MatmulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    RankedTensorType resTy = op.getResult().getType();

    Value dest = createZeroDest(rewriter, loc, resTy);

    AffineExpr m;
    AffineExpr n;
    AffineExpr k;
    bindDims(ctx, m, n, k);
    SmallVector<AffineMap> maps = {
        AffineMap::get(3, 0, {m, k}, ctx),
        AffineMap::get(3, 0, {k, n}, ctx),
        AffineMap::get(3, 0, {m, n}, ctx),
    };

    SmallVector<utils::IteratorType> iters = {utils::IteratorType::parallel,
                                              utils::IteratorType::parallel,
                                              utils::IteratorType::reduction};

    auto generic = linalg::GenericOp::create(
        rewriter, loc, TypeRange{resTy},
        ValueRange{adaptor.getLhs(), adaptor.getRhs()}, ValueRange{dest}, maps,
        iters, [](OpBuilder &nested, Location nestedLoc, ValueRange args) {
          Value prod =
              arith::MulFOp::create(nested, nestedLoc, args[0], args[1]);
          Value acc = arith::AddFOp::create(nested, nestedLoc, args[2], prod);
          linalg::YieldOp::create(nested, nestedLoc, acc);
        });

    rewriter.replaceOp(op, generic.getResults());

    return success();
  }
};

struct QMatmulOpLowering : public OpConversionPattern<QMatmulOp> {
  using OpConversionPattern<QMatmulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QMatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    RankedTensorType resTy = op.getResult().getType();
    Type i32 = rewriter.getI32Type();
    Type i64 = rewriter.getI64Type();
    auto accTy = RankedTensorType::get(resTy.getShape(), i32);

    auto constI32 = [&](int32_t v) -> Value {
      return arith::ConstantOp::create(rewriter, loc,
                                       rewriter.getI32IntegerAttr(v));
    };

    auto constI64 = [&](int64_t v) -> Value {
      return arith::ConstantOp::create(rewriter, loc,
                                       rewriter.getI64IntegerAttr(v));
    };

    auto sext = [](uint32_t v) {
      return static_cast<int64_t>(static_cast<int32_t>(v));
    };
    Value lhsZp = constI32(static_cast<int32_t>(op.getLhsZp()));
    Value rhsZp = constI32(static_cast<int32_t>(op.getRhsZp()));

    Value accDest = createZeroDest(rewriter, loc, accTy);
    AffineExpr m;
    AffineExpr n;
    AffineExpr k;
    bindDims(ctx, m, n, k);
    SmallVector<AffineMap> accMaps = {
        AffineMap::get(3, 0, {m, k}, ctx),
        AffineMap::get(3, 0, {k, n}, ctx),
        AffineMap::get(3, 0, {m, n}, ctx),
    };

    SmallVector<utils::IteratorType> accIters = {
        utils::IteratorType::parallel, utils::IteratorType::parallel,
        utils::IteratorType::reduction};
    auto accumulate = linalg::GenericOp::create(
        rewriter, loc, TypeRange{accTy},
        ValueRange{adaptor.getLhs(), adaptor.getRhs()}, ValueRange{accDest},
        accMaps, accIters, [&](OpBuilder &b, Location l, ValueRange args) {
          Value a = arith::SubIOp::create(
              b, l, arith::ExtSIOp::create(b, l, i32, args[0]), lhsZp);
          Value w = arith::SubIOp::create(
              b, l, arith::ExtSIOp::create(b, l, i32, args[1]), rhsZp);
          Value prod = arith::MulIOp::create(b, l, a, w);
          Value acc = arith::AddIOp::create(b, l, args[2], prod);
          linalg::YieldOp::create(b, l, acc);
        });

    int64_t totalShift = kQuantShiftBase + sext(op.getShift());
    Value multiplier = constI64(sext(op.getMultiplier()));
    Value round = constI64(int64_t{1} << (totalShift - 1));
    Value shift = constI64(totalShift);
    Value outZp = constI64(sext(op.getOutZp()));
    Value lo = constI64(kInt8Min);
    Value hi = constI64(kInt8Max);

    Value outDest = createEmptyDest(rewriter, loc, resTy);
    AffineMap id = rewriter.getMultiDimIdentityMap(2);
    auto requantize = linalg::GenericOp::create(
        rewriter, loc, TypeRange{resTy}, ValueRange{accumulate.getResult(0)},
        ValueRange{outDest}, ArrayRef<AffineMap>{id, id}, parallelIterators(2),
        [&](OpBuilder &b, Location l, ValueRange args) {
          Value x = arith::ExtSIOp::create(b, l, i64, args[0]);
          x = arith::MulIOp::create(b, l, x, multiplier);
          x = arith::AddIOp::create(b, l, x, round);
          x = arith::ShRSIOp::create(b, l, x, shift);
          x = arith::AddIOp::create(b, l, x, outZp);
          x = arith::MaxSIOp::create(b, l, x, lo);
          x = arith::MinSIOp::create(b, l, x, hi);
          Value r = arith::TruncIOp::create(b, l, resTy.getElementType(), x);
          linalg::YieldOp::create(b, l, r);
        });

    rewriter.replaceOp(op, requantize.getResults());

    return success();
  }
};

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

    AffineExpr n;
    AffineExpr oh;
    AffineExpr ow;
    AffineExpr f;
    AffineExpr kh;
    AffineExpr kw;
    AffineExpr c;
    bindDims(ctx, n, oh, ow, f, kh, kw, c);

    AffineExpr ih = oh * strides[0] + kh * dilations[0];
    AffineExpr iw = ow * strides[1] + kw * dilations[1];

    SmallVector<AffineMap> maps = {
        AffineMap::get(7, 0, {n, ih, iw, c}, ctx),
        AffineMap::get(7, 0, {kh, kw, c, f}, ctx),
        AffineMap::get(7, 0, {n, oh, ow, f}, ctx),
    };

    SmallVector<utils::IteratorType> iters = {
        utils::IteratorType::parallel,  utils::IteratorType::parallel,
        utils::IteratorType::parallel,  utils::IteratorType::parallel,
        utils::IteratorType::reduction, utils::IteratorType::reduction,
        utils::IteratorType::reduction,
    };

    auto generic = linalg::GenericOp::create(
        rewriter, loc, TypeRange{resTy},
        ValueRange{adaptor.getInput(), adaptor.getFilter()}, ValueRange{dest},
        maps, iters,
        [](OpBuilder &nested, Location nestedLoc, ValueRange args) {
          Value prod =
              arith::MulFOp::create(nested, nestedLoc, args[0], args[1]);
          Value acc = arith::AddFOp::create(nested, nestedLoc, args[2], prod);
          linalg::YieldOp::create(nested, nestedLoc, acc);
        });

    rewriter.replaceOp(op, generic.getResults());

    return success();
  }
};

struct ConvertDSPToLinalgPass
    : public mlir::nanodsp::impl::ConvertDSPToLinalgBase<
          ConvertDSPToLinalgPass> {
  using mlir::nanodsp::impl::ConvertDSPToLinalgBase<
      ConvertDSPToLinalgPass>::ConvertDSPToLinalgBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    ConversionTarget target(*ctx);
    target.addIllegalDialect<DSPDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet patterns(ctx);
    populateDSPToLinalgPatterns(patterns);

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};
}

void mlir::nanodsp::populateDSPToLinalgPatterns(RewritePatternSet &patterns) {
  patterns.add<AddOpLowering, ReluOpLowering, MatmulOpLowering,
               QMatmulOpLowering, Conv2DOpLowering>(patterns.getContext());
}
