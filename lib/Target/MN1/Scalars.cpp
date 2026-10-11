#include "Translator.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace mlir::nanodsp::mn1;

namespace {

struct ParityUse {
  scf::ForOp loop;
  bool next = false;
};

std::optional<int64_t> parityStep(AffineMap map) {
  if (map.getNumDims() != 1 || map.getNumSymbols() != 0 ||
      map.getNumResults() != 1)
    return std::nullopt;

  auto mod = dyn_cast<AffineBinaryOpExpr>(map.getResult(0));

  if (!mod || mod.getKind() != AffineExprKind::Mod)
    return std::nullopt;

  auto two = dyn_cast<AffineConstantExpr>(mod.getRHS());
  auto div = dyn_cast<AffineBinaryOpExpr>(mod.getLHS());

  if (!two || two.getValue() != 2 || !div ||
      div.getKind() != AffineExprKind::FloorDiv ||
      !isa<AffineDimExpr>(div.getLHS()))
    return std::nullopt;

  auto step = dyn_cast<AffineConstantExpr>(div.getRHS());

  if (!step)
    return std::nullopt;

  return step.getValue();
}

std::optional<ParityUse> parityUse(Value value, int64_t step) {
  bool next = false;

  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    std::optional<int64_t> rhs = getConstantIntValue(add.getRhs());

    if (!rhs || *rhs != step)
      return std::nullopt;

    value = add.getLhs();
    next = true;
  }

  auto arg = dyn_cast<BlockArgument>(value);

  if (!arg)
    return std::nullopt;

  auto loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp());

  if (!loop || loop.getInductionVar() != value)
    return std::nullopt;

  std::optional<int64_t> loopStep = getConstantIntValue(loop.getStep());
  std::optional<int64_t> lower = getConstantIntValue(loop.getLowerBound());

  if (!loopStep || *loopStep != step || !lower || *lower % (2 * step) != 0)
    return std::nullopt;

  return ParityUse{loop, next};
}

LogicalResult flatten(AffineExpr expr, int64_t scale, ValueRange operands,
                      SmallVectorImpl<std::pair<Value, int64_t>> &terms,
                      int64_t &constant) {
  if (auto c = dyn_cast<AffineConstantExpr>(expr)) {
    constant += scale * c.getValue();
    return success();
  }

  if (auto d = dyn_cast<AffineDimExpr>(expr)) {
    terms.push_back({operands[d.getPosition()], scale});
    return success();
  }

  auto bin = dyn_cast<AffineBinaryOpExpr>(expr);

  if (!bin)
    return failure();

  if (bin.getKind() == AffineExprKind::Add)
    return success(
        succeeded(flatten(bin.getLHS(), scale, operands, terms, constant)) &&
        succeeded(flatten(bin.getRHS(), scale, operands, terms, constant)));

  auto rhs = dyn_cast<AffineConstantExpr>(bin.getRHS());

  if (bin.getKind() != AffineExprKind::Mul || !rhs)
    return failure();

  return flatten(bin.getLHS(), scale * rhs.getValue(), operands, terms,
                 constant);
}

}

bool mlir::nanodsp::mn1::needsParity(scf::ForOp loop) {
  bool needed = false;

  loop.getBody()->walk([&](affine::AffineApplyOp apply) {
    std::optional<int64_t> step = parityStep(apply.getAffineMap());

    if (!step || apply.getMapOperands().size() != 1)
      return;

    std::optional<ParityUse> use = parityUse(apply.getMapOperands()[0], *step);
    needed = needed || (use && use->loop == loop);
  });

  return needed;
}

FailureOr<int> Translator::takeScalar(Operation *at) {
  if (std::optional<int> reg = scalarRegs_.take())
    return *reg;

  return at->emitError("out of MN1 scalar registers (the backend does not "
                       "spill)");
}

FailureOr<int> Translator::takeVector(Operation *at) {
  if (std::optional<int> reg = vectorRegs_.take())
    return *reg;

  return at->emitError("out of MN1 vector registers (the backend does not "
                       "spill)");
}

FailureOr<Scalar> Translator::scalarOf(Value value, Operation *at) {
  if (std::optional<int64_t> c = getConstantIntValue(value))
    return Scalar::of(*c);

  auto it = scalars_.find(value);

  if (it == scalars_.end())
    return at->emitError("index operand has no MN1 value");

  return it->second;
}

int Translator::materialize(Scalar value, int scratch) {
  if (value.isPlainReg())
    return value.reg;

  if (value.isConstant()) {
    out_.instr(llvm::formatv("li r{0}, {1}", scratch, value.constant));
    return scratch;
  }

  out_.instr(llvm::formatv("li r{0}, {1}", scratch, value.scale));
  out_.instr(llvm::formatv("mul r{0}, r{1}, r{0}", scratch, value.reg));

  if (value.constant != 0)
    out_.instr(llvm::formatv("addi r{0}, r{0}, {1}", scratch, value.constant));

  return scratch;
}

FailureOr<Scalar> Translator::linear(Scalar base, ArrayRef<Term> terms,
                                     Operation *at) {
  int64_t constant = base.constant;
  SmallVector<std::pair<int, int64_t>> dynamic;

  if (!base.isConstant())
    dynamic.push_back({base.reg, base.scale});

  for (const Term &term : terms) {
    constant += term.value.constant * term.coeff;
    int64_t coeff = term.coeff * term.value.scale;

    if (!term.value.isConstant() && coeff != 0)
      dynamic.push_back({term.value.reg, coeff});
  }

  if (dynamic.empty())
    return Scalar::of(constant);

  FailureOr<int> rd = takeScalar(at);

  if (failed(rd))
    return failure();

  for (auto [i, entry] : llvm::enumerate(dynamic)) {
    auto [reg, coeff] = entry;
    int source = reg;

    if (coeff != 1) {
      out_.instr(llvm::formatv("li r{0}, {1}", kScratch0, coeff));
      out_.instr(
          llvm::formatv("mul r{0}, r{1}, r{2}", kScratch1, reg, kScratch0));
      source = kScratch1;
    }

    if (i == 0)
      out_.instr(llvm::formatv("addi r{0}, r{1}, 0", *rd, source));
    else
      out_.instr(llvm::formatv("add r{0}, r{0}, r{1}", *rd, source));
  }

  if (constant != 0)
    out_.instr(llvm::formatv("addi r{0}, r{0}, {1}", *rd, constant));

  return Scalar::inReg(*rd);
}

LogicalResult Translator::emitAddI(arith::AddIOp op) {
  FailureOr<Scalar> lhs = scalarOf(op.getLhs(), op);
  FailureOr<Scalar> rhs = scalarOf(op.getRhs(), op);

  if (failed(lhs) || failed(rhs))
    return failure();

  FailureOr<Scalar> sum = linear(*lhs, {Term{*rhs, 1}}, op);

  if (failed(sum))
    return failure();

  scalars_[op.getResult()] = *sum;
  return success();
}

LogicalResult Translator::emitMulI(arith::MulIOp op) {
  FailureOr<Scalar> lhs = scalarOf(op.getLhs(), op);
  FailureOr<Scalar> rhs = scalarOf(op.getRhs(), op);

  if (failed(lhs) || failed(rhs))
    return failure();

  if (lhs->isConstant() || rhs->isConstant()) {
    Scalar c = lhs->isConstant() ? *lhs : *rhs;
    Scalar x = lhs->isConstant() ? *rhs : *lhs;
    FailureOr<Scalar> product =
        linear(Scalar::of(0), {Term{x, c.constant}}, op);

    if (failed(product))
      return failure();

    scalars_[op.getResult()] = *product;
    return success();
  }

  FailureOr<int> rd = takeScalar(op);

  if (failed(rd))
    return failure();

  int a = materialize(*lhs, kScratch0);
  int b = materialize(*rhs, kScratch1);
  out_.instr(llvm::formatv("mul r{0}, r{1}, r{2}", *rd, a, b));
  scalars_[op.getResult()] = Scalar::inReg(*rd);
  return success();
}

FailureOr<int> Translator::parityOf(Value iv) {
  auto arg = cast<BlockArgument>(iv);
  auto it = parity_.find(arg.getOwner()->getParentOp());

  if (it == parity_.end())
    return failure();

  return it->second;
}

LogicalResult Translator::emitAffineApply(affine::AffineApplyOp op) {
  AffineMap map = op.getAffineMap();
  ValueRange operands = op.getMapOperands();

  if (std::optional<int64_t> step = parityStep(map)) {
    std::optional<ParityUse> use = parityUse(operands[0], *step);

    if (!use)
      return op.emitError("MN1 has no divide: only the double-buffer slot "
                          "(iv floordiv step) mod 2 of an enclosing loop is "
                          "supported");

    FailureOr<int> parity = parityOf(use->loop.getInductionVar());

    if (failed(parity))
      return op.emitError("loop parity register missing");

    scalars_[op.getResult()] = use->next ? Scalar::borrowed(*parity, -1, 1)
                                         : Scalar::borrowed(*parity);
    return success();
  }

  SmallVector<std::pair<Value, int64_t>> flat;
  int64_t constant = 0;

  if (map.getNumResults() != 1 || map.getNumSymbols() != 0 ||
      failed(flatten(map.getResult(0), 1, operands, flat, constant)))
    return op.emitError("MN1 supports only affine maps of adds and constant "
                        "multiplies");

  SmallVector<Term> terms;

  for (auto [value, coeff] : flat) {
    FailureOr<Scalar> s = scalarOf(value, op);

    if (failed(s))
      return failure();

    terms.push_back(Term{*s, coeff});
  }

  FailureOr<Scalar> result = linear(Scalar::of(constant), terms, op);

  if (failed(result))
    return failure();

  scalars_[op.getResult()] = *result;
  return success();
}
