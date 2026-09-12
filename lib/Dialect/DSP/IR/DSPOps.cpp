#include "nanodsp/Dialect/DSP/IR/DSPOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::nanodsp;

#define GET_OP_CLASSES
#include "nanodsp/Dialect/DSP/IR/DSPOps.cpp.inc"

//===----------------------------------------------------------------------===//
// ReluOp
//===----------------------------------------------------------------------===//

LogicalResult ReluOp::canonicalize(ReluOp op, PatternRewriter &rewriter) {
  // relu is idempotent -- max(max(x,0),0) == max(x,0), including for NaN.
  auto inner = op.getInput().getDefiningOp<ReluOp>();
  if (!inner)
    return failure();
  rewriter.replaceOp(op, inner.getResult());
  return success();
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

LogicalResult MatmulOp::verify() {
  RankedTensorType lhsTy = getLhs().getType();
  RankedTensorType rhsTy = getRhs().getType();
  RankedTensorType resTy = getResult().getType();

  const int64_t m = lhsTy.getDimSize(0);
  const int64_t kLhs = lhsTy.getDimSize(1);
  const int64_t kRhs = rhsTy.getDimSize(0);
  const int64_t n = rhsTy.getDimSize(1);

  if (kLhs != kRhs)
    return emitOpError() << "contraction dimension mismatch: lhs has K=" << kLhs
                         << " but rhs has K=" << kRhs;

  if (resTy.getDimSize(0) != m || resTy.getDimSize(1) != n)
    return emitOpError() << "result shape must be " << m << "x" << n
                         << ", got " << resTy.getDimSize(0) << "x"
                         << resTy.getDimSize(1);

  return success();
}

//===----------------------------------------------------------------------===//
// Conv2DOp
//===----------------------------------------------------------------------===//

LogicalResult Conv2DOp::verify() {
  RankedTensorType inTy = getInput().getType();
  RankedTensorType fTy = getFilter().getType();
  RankedTensorType resTy = getResult().getType();

  ArrayRef<int64_t> strides = getStrides();
  ArrayRef<int64_t> dilations = getDilations();

  if (strides.size() != 2)
    return emitOpError() << "expected 2 strides, got " << strides.size();
  if (dilations.size() != 2)
    return emitOpError() << "expected 2 dilations, got " << dilations.size();
  for (int64_t s : strides)
    if (s < 1)
      return emitOpError() << "strides must be >= 1, got " << s;
  for (int64_t d : dilations)
    if (d < 1)
      return emitOpError() << "dilations must be >= 1, got " << d;

  // input NHWC, filter HWCF
  const int64_t n = inTy.getDimSize(0);
  const int64_t h = inTy.getDimSize(1);
  const int64_t w = inTy.getDimSize(2);
  const int64_t c = inTy.getDimSize(3);
  const int64_t kh = fTy.getDimSize(0);
  const int64_t kw = fTy.getDimSize(1);
  const int64_t fc = fTy.getDimSize(2);
  const int64_t f = fTy.getDimSize(3);

  if (c != fc)
    return emitOpError() << "channel mismatch: input has C=" << c
                         << " but filter has C=" << fc;

  const int64_t effKH = (kh - 1) * dilations[0] + 1;
  const int64_t effKW = (kw - 1) * dilations[1] + 1;
  if (effKH > h || effKW > w)
    return emitOpError() << "dilated filter " << effKH << "x" << effKW
                         << " does not fit in input spatial extent " << h << "x"
                         << w;

  const int64_t oh =
      computeConv2DOutputDim(h, kh, strides[0], dilations[0]);
  const int64_t ow =
      computeConv2DOutputDim(w, kw, strides[1], dilations[1]);

  if (resTy.getDimSize(0) != n || resTy.getDimSize(1) != oh ||
      resTy.getDimSize(2) != ow || resTy.getDimSize(3) != f)
    return emitOpError() << "result shape must be " << n << "x" << oh << "x"
                         << ow << "x" << f << ", got " << resTy;

  return success();
}