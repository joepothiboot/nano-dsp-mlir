#include "Translator.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace mlir::nanodsp::mn1;

namespace {

struct LoopBounds {
  int64_t lower = 0;
  int64_t upper = 0;
  int64_t step = 1;
};

std::optional<LoopBounds> boundsOf(scf::ForOp loop) {
  std::optional<int64_t> lower = getConstantIntValue(loop.getLowerBound());
  std::optional<int64_t> upper = getConstantIntValue(loop.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(loop.getStep());

  if (!lower || !upper || !step || *step <= 0 || *upper <= *lower ||
      (*upper - *lower) % *step != 0)
    return std::nullopt;

  return LoopBounds{*lower, *upper, *step};
}

bool isNextIterationGuard(arith::CmpIOp cmp) {
  if (cmp.getPredicate() != arith::CmpIPredicate::slt)
    return false;

  auto next = cmp.getLhs().getDefiningOp<arith::AddIOp>();
  auto iv = next ? dyn_cast<BlockArgument>(next.getLhs()) : nullptr;
  auto loop = iv ? dyn_cast<scf::ForOp>(iv.getOwner()->getParentOp()) : nullptr;

  if (!loop || loop.getInductionVar() != iv)
    return false;

  std::optional<LoopBounds> bounds = boundsOf(loop);
  std::optional<int64_t> step = getConstantIntValue(next.getRhs());
  std::optional<int64_t> limit = getConstantIntValue(cmp.getRhs());

  return bounds && step && limit && *step == bounds->step &&
         *limit == bounds->upper;
}

}

LogicalResult Translator::emitFor(scf::ForOp op) {
  if (op.getNumResults() != 0)
    return op.emitError("MN1 loops cannot carry values");

  std::optional<LoopBounds> bounds = boundsOf(op);

  if (!bounds)
    return op.emitError("MN1 loops need static bounds with a whole number "
                        "of iterations");

  FailureOr<int> iv = takeScalar(op);

  if (failed(iv))
    return failure();

  out_.instr(llvm::formatv("li r{0}, {1}", *iv, bounds->lower));
  std::optional<int> parity;

  if (needsParity(op)) {
    FailureOr<int> reg = takeScalar(op);

    if (failed(reg))
      return failure();

    out_.instr(llvm::formatv("li r{0}, 0", *reg));
    parity = reg;
    parity_[op] = *reg;
  }

  scalars_[op.getInductionVar()] = Scalar::borrowed(*iv);
  std::string head = out_.freshLabel("loop");
  out_.label(head);

  if (failed(emitBlock(*op.getBody())))
    return failure();

  if (parity) {
    out_.instr(llvm::formatv("li r{0}, -1", kScratch0));
    out_.instr(llvm::formatv("mul r{0}, r{0}, r{1}", *parity, kScratch0));
    out_.instr(llvm::formatv("addi r{0}, r{0}, 1", *parity));
    parity_.erase(op);
    scalarRegs_.release(*parity);
  }

  out_.instr(llvm::formatv("addi r{0}, r{0}, {1}", *iv, bounds->step));
  out_.instr(llvm::formatv("li r{0}, {1}", kScratch0, bounds->upper));
  out_.instr(llvm::formatv("bne r{0}, r{1}, {2}", *iv, kScratch0, head));
  scalars_.erase(op.getInductionVar());
  scalarRegs_.release(*iv);
  return success();
}

LogicalResult Translator::emitIf(scf::IfOp op) {
  if (op.getNumResults() != 0 || !op.getElseRegion().empty())
    return op.emitError("MN1 supports only scf.if without else or results");

  auto cmp = op.getCondition().getDefiningOp<arith::CmpIOp>();

  if (!cmp || !isNextIterationGuard(cmp))
    return op.emitError("MN1 has no set-less-than: only the guard "
                        "'iv + step < ub' of an enclosing loop is supported");

  FailureOr<Scalar> lhs = scalarOf(cmp.getLhs(), op);
  FailureOr<Scalar> rhs = scalarOf(cmp.getRhs(), op);

  if (failed(lhs) || failed(rhs))
    return failure();

  int l = materialize(*lhs, kScratch1);
  int r = materialize(*rhs, kScratch0);
  std::string then = out_.freshLabel("then");
  std::string done = out_.freshLabel("endif");
  out_.instr(llvm::formatv("bne r{0}, r{1}, {2}", l, r, then));
  out_.instr("jmp " + done);
  out_.label(then);

  if (failed(emitBlock(*op.thenBlock())))
    return failure();

  out_.label(done);
  return success();
}
