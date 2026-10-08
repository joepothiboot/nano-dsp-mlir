#include "nanodsp/Schedule/TileSizeModel.h"

#include "mlir/IR/AffineMap.h"

using namespace mlir;
using namespace mlir::nanodsp;

static constexpr int64_t kMaxRegisterRows = 16;
static constexpr int64_t kMaxRegisterVectors = 4;

static unsigned elementBytes(Type type) {
  Type elem = getElementTypeOrSelf(type);

  if (!elem.isIntOrFloat())
    return 0;

  return llvm::divideCeil(elem.getIntOrFloatBitWidth(), 8);
}

uint64_t mlir::nanodsp::computeWorkingSetBytes(linalg::LinalgOp op,
                                               ArrayRef<int64_t> tile) {
  SmallVector<int64_t> zero(tile.size(), 0), last;

  for (int64_t t : tile)
    last.push_back(t - 1);

  uint64_t bytes = 0;

  for (OpOperand &operand : op->getOpOperands()) {
    AffineMap map = op.getMatchingIndexingMap(&operand);
    SmallVector<int64_t> lo = map.compose(zero), hi = map.compose(last);
    uint64_t elems = 1;

    for (auto [l, h] : llvm::zip_equal(lo, hi))
      elems *= static_cast<uint64_t>(h - l + 1);

    bytes += elems * elementBytes(operand.get().getType());
  }

  return bytes;
}

static bool hasNonUnitStride(linalg::LinalgOp op, unsigned d) {
  unsigned numLoops = op.getNumLoops();
  SmallVector<int64_t> zero(numLoops, 0), unit(numLoops, 0);
  unit[d] = 1;

  for (AffineMap map : op.getIndexingMapsArray()) {
    SmallVector<int64_t> at0 = map.compose(zero), at1 = map.compose(unit);

    for (auto [a, b] : llvm::zip_equal(at0, at1))
      if (b - a > 1)
        return true;
  }

  return false;
}

static int64_t snapToDivisor(int64_t n, int64_t want, int64_t multipleOf = 1) {
  want = std::max<int64_t>(1, std::min(want, n));

  for (int64_t d = want; d >= 1; --d)
    if (n % d == 0 && d % multipleOf == 0)
      return d;

  for (int64_t d = want; d >= 1; --d)
    if (n % d == 0)
      return d;

  return 1;
}

static int64_t nextDivisor(int64_t n, int64_t cur, int64_t step) {
  for (int64_t d = cur + step; d <= n; d += step)
    if (n % d == 0)
      return d;

  return cur;
}

static std::pair<int64_t, int64_t> chooseRegisterShape(unsigned numRegs,
                                                       bool hasReduction) {
  if (!hasReduction)
    return {1, 4};

  int64_t bestMr = 1, bestNv = 1;
  double bestIntensity = 0.0;

  for (int64_t nv = 1; nv <= kMaxRegisterVectors; ++nv) {
    for (int64_t mr = 1; mr <= kMaxRegisterRows; ++mr) {
      if (mr * nv + nv + 1 > static_cast<int64_t>(numRegs))
        break;

      double intensity = double(mr * nv) / double(mr + nv);

      if (intensity > bestIntensity) {
        bestIntensity = intensity;
        bestMr = mr;
        bestNv = nv;
      }
    }
  }

  return {bestMr, bestNv};
}

static bool keepsReductionOrder(ArrayRef<unsigned> reductionDims,
                                ArrayRef<int64_t> tile,
                                ArrayRef<int64_t> ranges) {
  for (auto [i, a] : llvm::enumerate(reductionDims))
    for (unsigned b : reductionDims.drop_front(i + 1))
      if (tile[a] != 1 && tile[b] != ranges[b])
        return false;

  return true;
}

FailureOr<TileSizes>
mlir::nanodsp::computeTileSizes(linalg::LinalgOp op,
                                const TargetModel &target) {
  if (op.getNumDpsInits() != 1)
    return failure();

  TileSizes sizes;
  sizes.loopRanges = op.getStaticLoopRanges();

  if (llvm::any_of(sizes.loopRanges, ShapedType::isDynamic))
    return failure();

  OpOperand *init = op.getDpsInitOperand(0);
  unsigned elemBytes = elementBytes(init->get().getType());

  if (elemBytes == 0)
    return failure();

  int64_t lanes = target.lanes(elemBytes * 8);

  AffineMap outMap = op.getMatchingIndexingMap(init);
  auto dimOf = [&](int64_t result) -> std::optional<unsigned> {
    if (result < 0)
      return std::nullopt;

    if (auto d = dyn_cast<AffineDimExpr>(outMap.getResult(result)))
      return d.getPosition();

    return std::nullopt;
  };

  int64_t outRank = outMap.getNumResults();
  std::optional<unsigned> vecDim = dimOf(outRank - 1);
  std::optional<unsigned> rowDim = dimOf(outRank - 2);

  SmallVector<utils::IteratorType> iters = op.getIteratorTypesArray();
  SmallVector<unsigned> reductionDims;

  for (auto [i, it] : llvm::enumerate(iters))
    if (it == utils::IteratorType::reduction)
      reductionDims.push_back(i);

  auto [mr, nv] =
      chooseRegisterShape(target.numVectorRegs, !reductionDims.empty());
  sizes.reg.assign(sizes.loopRanges.size(), 1);

  if (vecDim)
    sizes.reg[*vecDim] =
        snapToDivisor(sizes.loopRanges[*vecDim], lanes * nv, lanes);

  if (rowDim)
    sizes.reg[*rowDim] = snapToDivisor(sizes.loopRanges[*rowDim], mr);

  for (std::optional<unsigned> d : {vecDim, rowDim})
    if (d && hasNonUnitStride(op, *d))
      sizes.reg[*d] = 1;

  if (reductionDims.empty()) {
    sizes.cache = sizes.loopRanges;
    sizes.cacheWorkingSetBytes = computeWorkingSetBytes(op, sizes.cache);

    return sizes;
  }

  SmallVector<unsigned> growOrder(reductionDims);

  if (rowDim)
    growOrder.push_back(*rowDim);

  if (vecDim)
    growOrder.push_back(*vecDim);

  sizes.cache = sizes.reg;
  uint64_t budget = target.tileBudgetBytes();

  for (bool grew = true; grew;) {
    grew = false;

    for (unsigned d : growOrder) {
      int64_t next =
          nextDivisor(sizes.loopRanges[d], sizes.cache[d], sizes.reg[d]);
      if (next == sizes.cache[d])
        continue;

      SmallVector<int64_t> candidate(sizes.cache);
      candidate[d] = next;

      if (computeWorkingSetBytes(op, candidate) > budget)
        continue;

      if (!keepsReductionOrder(reductionDims, candidate, sizes.loopRanges))
        continue;

      sizes.cache = std::move(candidate);
      grew = true;
    }
  }

  sizes.cacheWorkingSetBytes = computeWorkingSetBytes(op, sizes.cache);

  return sizes;
}
