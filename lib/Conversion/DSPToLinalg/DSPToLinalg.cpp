//===- DSPToLinalg.cpp - Lower 'dsp' to linalg.generic --------------------===//
//
// L1 (dsp, value semantics, whole-array) -> L2 (linalg on tensors, DPS).
// Invariants established here and relied upon by Stage 3:
//   * every result is produced by linalg.generic ops only: exactly one, except
//     dsp.qmatmul, which is an i32 accumulate generic plus an elementwise
//     requantize generic
//   * every generic is in destination-passing style with a tensor.empty dest
//   * reductions have their destination explicitly zero-filled
//   * all shapes are static; element types are f32, or i8/i32 for qmatmul
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
  return tensor::EmptyOp::create(b, loc, type.getShape(),
                                 type.getElementType());
}

/// A zero-initialized destination tensor. Required for reductions, where the
/// generic accumulates into the destination.
static Value createZeroDest(OpBuilder &b, Location loc,
                            RankedTensorType type) {
  Value empty = createEmptyDest(b, loc, type);
  Value zero =
      arith::ConstantOp::create(b, loc, b.getZeroAttr(type.getElementType()));
  return linalg::FillOp::create(b, loc, ValueRange{zero}, ValueRange{empty})
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

    auto generic = linalg::GenericOp::create(
        rewriter, loc, TypeRange{resTy},
        /*inputs=*/ValueRange{adaptor.getLhs(), adaptor.getRhs()},
        /*outputs=*/ValueRange{dest},
        /*indexingMaps=*/ArrayRef<AffineMap>{id, id, id},
        /*iteratorTypes=*/parallelIterators(rank),
        [](OpBuilder &nested, Location nestedLoc, ValueRange args) {
          Value sum =
              arith::AddFOp::create(nested, nestedLoc, args[0], args[1]);
          linalg::YieldOp::create(nested, nestedLoc, sum);
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
    Value zero = arith::ConstantOp::create(
        rewriter, loc, rewriter.getZeroAttr(resTy.getElementType()));

    AffineMap id = rewriter.getMultiDimIdentityMap(rank);

    auto generic = linalg::GenericOp::create(
        rewriter, loc, TypeRange{resTy},
        /*inputs=*/ValueRange{adaptor.getInput()},
        /*outputs=*/ValueRange{dest},
        /*indexingMaps=*/ArrayRef<AffineMap>{id, id},
        /*iteratorTypes=*/parallelIterators(rank),
        [zero](OpBuilder &nested, Location nestedLoc, ValueRange args) {
          // maximumf, not maxnumf: NaN must propagate (numpy.maximum semantics).
          Value r = arith::MaximumFOp::create(nested, nestedLoc, args[0], zero);
          linalg::YieldOp::create(nested, nestedLoc, r);
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

    auto generic = linalg::GenericOp::create(
        rewriter, loc, TypeRange{resTy},
        /*inputs=*/ValueRange{adaptor.getLhs(), adaptor.getRhs()},
        /*outputs=*/ValueRange{dest}, maps, iters,
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

//===----------------------------------------------------------------------===//
// dsp.qmatmul -> two linalg.generic ops
//
//   1. accumulate, iterators (m, n, k), i32 destination zero-filled:
//        acc += (extsi(lhs) - lhs_zp) * (extsi(rhs) - rhs_zp)
//   2. requantize, iterators (m, n), elementwise i32 -> i8, in i64:
//        clamp(out_zp + ((acc * multiplier + 2^(s-1)) >> s), -128, 127)
//
// Kept as two ops rather than fusing the requantization into the reduction's
// region: the accumulate is then a plain matmul-shaped generic that Stage 3
// tiles like the f32 one, and the requantize is a plain elementwise op.
//===----------------------------------------------------------------------===//

struct QMatmulOpLowering : public OpConversionPattern<QMatmulOp> {
  using OpConversionPattern<QMatmulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QMatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    RankedTensorType resTy = op.getResult().getType();
    Type i32 = rewriter.getI32Type(), i64 = rewriter.getI64Type();
    auto accTy = RankedTensorType::get(resTy.getShape(), i32);

    // Loop-invariant constants, hoisted out of the regions (as in relu).
    auto constI32 = [&](int64_t v) -> Value {
      return arith::ConstantOp::create(rewriter, loc,
                                       rewriter.getI32IntegerAttr(v));
    };
    auto constI64 = [&](int64_t v) -> Value {
      return arith::ConstantOp::create(rewriter, loc,
                                       rewriter.getI64IntegerAttr(v));
    };
    // I32Attr accessors return uint32_t; read them as the signed values they
    // are, or a negative zero point zero-extends when widened.
    auto sext = [](uint32_t v) { return int64_t(static_cast<int32_t>(v)); };
    Value lhsZp = constI32(sext(op.getLhsZp()));
    Value rhsZp = constI32(sext(op.getRhsZp()));

    // 1. Accumulate.
    Value accDest = createZeroDest(rewriter, loc, accTy);
    AffineExpr m, n, k;
    bindDims(ctx, m, n, k);
    SmallVector<AffineMap> accMaps = {
        AffineMap::get(3, 0, {m, k}, ctx), // lhs  (M x K)
        AffineMap::get(3, 0, {k, n}, ctx), // rhs  (K x N)
        AffineMap::get(3, 0, {m, n}, ctx), // acc  (M x N)
    };
    SmallVector<utils::IteratorType> accIters = {
        utils::IteratorType::parallel, utils::IteratorType::parallel,
        utils::IteratorType::reduction};
    auto accumulate = linalg::GenericOp::create(
        rewriter, loc, TypeRange{accTy},
        /*inputs=*/ValueRange{adaptor.getLhs(), adaptor.getRhs()},
        /*outputs=*/ValueRange{accDest}, accMaps, accIters,
        [&](OpBuilder &b, Location l, ValueRange args) {
          Value a = arith::SubIOp::create(
              b, l, arith::ExtSIOp::create(b, l, i32, args[0]), lhsZp);
          Value w = arith::SubIOp::create(
              b, l, arith::ExtSIOp::create(b, l, i32, args[1]), rhsZp);
          Value prod = arith::MulIOp::create(b, l, a, w);
          Value acc = arith::AddIOp::create(b, l, args[2], prod);
          linalg::YieldOp::create(b, l, acc);
        });

    // 2. Requantize.
    int64_t totalShift = 31 + sext(op.getShift());
    Value multiplier = constI64(sext(op.getMultiplier()));
    Value round = constI64(int64_t{1} << (totalShift - 1));
    Value shift = constI64(totalShift);
    Value outZp = constI64(sext(op.getOutZp()));
    Value lo = constI64(-128), hi = constI64(127);

    Value outDest = createEmptyDest(rewriter, loc, resTy);
    AffineMap id = rewriter.getMultiDimIdentityMap(2);
    auto requantize = linalg::GenericOp::create(
        rewriter, loc, TypeRange{resTy},
        /*inputs=*/ValueRange{accumulate.getResult(0)},
        /*outputs=*/ValueRange{outDest},
        /*indexingMaps=*/ArrayRef<AffineMap>{id, id},
        /*iteratorTypes=*/parallelIterators(2),
        [&](OpBuilder &b, Location l, ValueRange args) {
          Value x = arith::ExtSIOp::create(b, l, i64, args[0]);
          x = arith::MulIOp::create(b, l, x, multiplier);
          x = arith::AddIOp::create(b, l, x, round);
          // Arithmetic shift = floor division, so the + 2^(s-1) above
          // rounds half up (toward +infinity), for negative values too.
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

    auto generic = linalg::GenericOp::create(
        rewriter, loc, TypeRange{resTy},
        /*inputs=*/ValueRange{adaptor.getInput(), adaptor.getFilter()},
        /*outputs=*/ValueRange{dest}, maps, iters,
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
    // Everything else is left alone: the payload may already contain other
    // dialects, and a Stage 3 schedule (transform.named_sequence) can live in
    // the same module.
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

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
               QMatmulOpLowering, Conv2DOpLowering>(patterns.getContext());
}