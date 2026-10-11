#include "Translator.h"

#include "nanodsp/Dialect/DSP/IR/DSPAttrs.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "llvm/Support/FormatVariadic.h"

#include "llvm/Support/MathExtras.h"
#include <utility>

using namespace mlir;
using namespace mlir::nanodsp;
using namespace mlir::nanodsp::mn1;

namespace {

bool isLocal(MemRefType type) {
  return isa_and_nonnull<LocalMemorySpaceAttr>(type.getMemorySpace());
}

bool isTagBuffer(Value value) {
  return llvm::all_of(value.getUsers(), [](Operation *user) {
    if (auto subview = dyn_cast<memref::SubViewOp>(user))
      return isTagBuffer(subview.getResult());

    return isa<memref::DmaStartOp, memref::DmaWaitOp, memref::DeallocOp>(user);
  });
}

SmallVector<int64_t> stridesOf(MemRefType type) {
  return type.getStridesAndOffset().first;
}

}

LogicalResult Translator::emitAlloc(memref::AllocOp op) {
  MemRefType type = op.getType();

  if (!type.hasStaticShape())
    return op.emitError("MN1 needs statically shaped buffers");

  if (type.getElementType().isInteger(32) && !isLocal(type) &&
      isTagBuffer(op.getResult())) {
    auto count = static_cast<int>(type.getNumElements());

    if (nextTag_ + count > kInternalTag)
      return op.emitError("needs more than ")
             << kInternalTag << " DMA tags (MN1 has " << kNumTags
             << ", one reserved)";

    tags_[op.getResult()] = TagRef{nextTag_, -1, false};
    nextTag_ += count;
    return success();
  }

  if (type.getElementTypeBitWidth() != kElementBytes * 8)
    return op.emitError("MN1 buffers must hold 32-bit elements");

  uint64_t bytes = type.getNumElements() * kElementBytes;
  Buffer buffer;
  buffer.shape = llvm::to_vector(type.getShape());
  buffer.strides = stridesOf(type);
  buffer.local = isLocal(type);

  if (buffer.local) {
    spmTop_ = llvm::alignTo(spmTop_, kVectorBytes);
    buffer.base = Scalar::of(static_cast<int64_t>(spmTop_));
    spmTop_ += llvm::alignTo(bytes, kVectorBytes);

    if (spmTop_ > kSpmBytes)
      return op.emitError("local buffers need ")
             << spmTop_ << " bytes, MN1 has " << kSpmBytes;
  } else {
    if (bytes > kDramSlotBytes || dramTop_ + kDramSlotBytes > kDramBytes)
      return op.emitError("buffer does not fit the MN1 DRAM layout");

    buffer.base = Scalar::of(static_cast<int64_t>(dramTop_));
    out_.note(llvm::formatv("buffer at {0:x} ({1} bytes)", dramTop_, bytes));
    dramTop_ += kDramSlotBytes;
  }

  buffers_[op.getResult()] = std::move(buffer);
  return success();
}

LogicalResult Translator::emitSubView(memref::SubViewOp op) {
  if (!op.hasUnitStride())
    return op.emitError("MN1 supports only unit-stride subviews");

  SmallVector<Scalar> offsets;

  for (OpFoldResult offset : op.getMixedOffsets()) {
    if (std::optional<int64_t> c = getConstantIntValue(offset)) {
      offsets.push_back(Scalar::of(*c));
      continue;
    }

    FailureOr<Scalar> s = scalarOf(cast<Value>(offset), op);

    if (failed(s))
      return failure();

    offsets.push_back(*s);
  }

  SmallVector<int64_t> sourceStrides = stridesOf(op.getSourceType());

  if (auto it = tags_.find(op.getSource()); it != tags_.end()) {
    TagRef tag = it->second;

    for (auto [offset, stride] : llvm::zip_equal(offsets, sourceStrides)) {
      if (offset.isConstant()) {
        tag.base += static_cast<int>(offset.constant * stride);
        continue;
      }

      bool parity =
          !offset.owned && ((offset.scale == 1 && offset.constant == 0) ||
                            (offset.scale == -1 && offset.constant == 1));

      if (stride != 1 || tag.slotReg >= 0 || !parity)
        return op.emitError("MN1 DMA tag slots must be a loop's buffer parity");

      tag.slotReg = offset.reg;
      tag.inverted = offset.scale == -1;
    }

    tags_[op.getResult()] = tag;
    return success();
  }

  auto it = buffers_.find(op.getSource());

  if (it == buffers_.end())
    return op.emitError("subview of a buffer the MN1 backend does not know");

  SmallVector<Term> terms;

  for (auto [offset, stride] : llvm::zip_equal(offsets, sourceStrides))
    terms.push_back(Term{offset, stride * kElementBytes});

  bool zeroOffsets = llvm::all_of(offsets, [](const Scalar &s) {
    return s.isConstant() && s.constant == 0;
  });
  bool steal = zeroOffsets && it->second.base.ownsReg() &&
               llvm::is_contained(releases_[op], op.getSource());
  Scalar base = it->second.base;

  if (!steal) {
    FailureOr<Scalar> computed = linear(it->second.base, terms, op);

    if (failed(computed))
      return failure();

    base = *computed;
  }

  Buffer view;
  view.base = base;
  view.local = it->second.local;
  view.shape = llvm::to_vector(op.getType().getShape());
  view.strides = stridesOf(op.getType());

  if (steal)
    buffers_.erase(op.getSource());

  buffers_[op.getResult()] = std::move(view);
  return success();
}

FailureOr<Scalar> Translator::elementAddress(const Buffer &buffer,
                                             ValueRange indices,
                                             Operation *at) {
  SmallVector<Term> terms;
  bool zero = true;

  for (auto [index, stride] : llvm::zip_equal(indices, buffer.strides)) {
    FailureOr<Scalar> s = scalarOf(index, at);

    if (failed(s))
      return failure();

    zero = zero && s->isConstant() && s->constant == 0;
    terms.push_back(Term{*s, stride * kElementBytes});
  }

  if (zero) {
    Scalar borrowed = buffer.base;
    borrowed.owned = false;
    return borrowed;
  }

  return linear(buffer.base, terms, at);
}

FailureOr<TagRef> Translator::tagOf(Value memref, ValueRange indices,
                                    Operation *at) {
  auto it = tags_.find(memref);

  if (it == tags_.end())
    return at->emitError("DMA tag buffer the MN1 backend does not know");

  TagRef tag = it->second;

  for (Value index : indices) {
    std::optional<int64_t> c = getConstantIntValue(index);

    if (!c)
      return at->emitError("MN1 DMA tag indices must be constant");

    tag.base += static_cast<int>(*c);
  }

  return tag;
}

void Translator::tagged(const TagRef &tag,
                        const std::function<void(int)> &emit) {
  if (tag.slotReg < 0) {
    emit(tag.base);
    return;
  }

  int even = tag.inverted ? tag.base + 1 : tag.base;
  int odd = tag.inverted ? tag.base : tag.base + 1;
  std::string oddLabel = out_.freshLabel("slot1_");
  std::string done = out_.freshLabel("slot_done");
  out_.instr(llvm::formatv("bne r{0}, r0, {1}", tag.slotReg, oddLabel));
  emit(even);
  out_.instr("jmp " + done);
  out_.label(oddLabel);
  emit(odd);
  out_.label(done);
}

void Translator::emitStridedDma(llvm::StringRef mnemonic, const TagRef &tag,
                                int spm, int dram, int64_t rows,
                                int64_t rowBytes, int64_t strideBytes,
                                int counter) {
  auto issue = [&](int t) {
    out_.instr(llvm::formatv("{0} {1}, r{2}, r{3}, r{4}", mnemonic, t,
                             kScratch1, kScratch2, kScratch0));
  };
  auto wait = [&](int t) { out_.instr(llvm::formatv("dma.wait {0}", t)); };

  out_.instr(llvm::formatv("addi r{0}, r{1}, 0", kScratch1, spm));
  out_.instr(llvm::formatv("addi r{0}, r{1}, 0", kScratch2, dram));
  out_.instr(llvm::formatv("li r{0}, {1}", kScratch0, rowBytes));

  if (rows > 1) {
    std::string head = out_.freshLabel("dma_rows");
    out_.instr(llvm::formatv("li r{0}, {1}", counter, rows - 1));
    out_.label(head);
    tagged(tag, issue);
    tagged(tag, wait);
    out_.instr(llvm::formatv("addi r{0}, r{0}, {1}", kScratch1, rowBytes));
    out_.instr(llvm::formatv("addi r{0}, r{0}, {1}", kScratch2, strideBytes));
    out_.instr(llvm::formatv("addi r{0}, r{0}, -1", counter));
    out_.instr(llvm::formatv("bne r{0}, r0, {1}", counter, head));
  }

  tagged(tag, issue);
}

LogicalResult Translator::emitDmaStart(memref::DmaStartOp op) {
  auto src = buffers_.find(op.getSrcMemRef());
  auto dst = buffers_.find(op.getDstMemRef());

  if (src == buffers_.end() || dst == buffers_.end())
    return op.emitError("DMA between buffers the MN1 backend does not know");

  if (src->second.local == dst->second.local)
    return op.emitError("MN1 DMAs move data between DRAM and the scratchpad");

  std::optional<int64_t> count = getConstantIntValue(op.getNumElements());
  FailureOr<Scalar> srcAddr =
      elementAddress(src->second, op.getSrcIndices(), op);
  FailureOr<Scalar> dstAddr =
      elementAddress(dst->second, op.getDstIndices(), op);
  FailureOr<TagRef> tag = tagOf(op.getTagMemRef(), op.getTagIndices(), op);

  if (failed(srcAddr) || failed(dstAddr) || failed(tag))
    return failure();

  if (!count)
    return op.emitError("MN1 DMAs need a constant element count");

  bool load = dst->second.local;
  Scalar spmAddr = load ? *dstAddr : *srcAddr;
  Scalar dramAddr = load ? *srcAddr : *dstAddr;
  llvm::StringRef mnemonic = load ? "dma.ld" : "dma.st";
  int spm = materialize(spmAddr, kScratch1);
  int dram = materialize(dramAddr, kScratch2);

  if (!op.isStrided()) {
    out_.instr(
        llvm::formatv("li r{0}, {1}", kScratch0, *count * kElementBytes));
    tagged(*tag, [&](int t) {
      out_.instr(llvm::formatv("{0} {1}, r{2}, r{3}, r{4}", mnemonic, t, spm,
                               dram, kScratch0));
    });
  } else {
    std::optional<int64_t> stride = getConstantIntValue(op.getStride());
    std::optional<int64_t> run =
        getConstantIntValue(op.getNumElementsPerStride());

    if (!stride || !run || *count % *run != 0)
      return op.emitError("MN1 needs static, whole-row strided DMAs");

    int64_t rowBytes = *run * kElementBytes;
    int64_t strideBytes = *stride * kElementBytes;

    if (rowBytes % kDmaGranule != 0 || strideBytes % kDmaGranule != 0)
      return op.emitError("MN1 DMA rows must be multiples of ")
             << kDmaGranule << " bytes";

    FailureOr<int> counter = takeScalar(op);

    if (failed(counter))
      return failure();

    emitStridedDma(mnemonic, *tag, spm, dram, *count / *run, rowBytes,
                   strideBytes, *counter);
    scalarRegs_.release(*counter);
  }

  for (Scalar address : {*srcAddr, *dstAddr})
    if (address.ownsReg())
      scalarRegs_.release(address.reg);

  return success();
}

LogicalResult Translator::emitDmaWait(memref::DmaWaitOp op) {
  FailureOr<TagRef> tag = tagOf(op.getTagMemRef(), op.getTagIndices(), op);

  if (failed(tag))
    return failure();

  tagged(*tag, [&](int t) { out_.instr(llvm::formatv("dma.wait {0}", t)); });
  return success();
}

FailureOr<int64_t> Translator::zeroBlock(Operation *at) {
  if (zeroBlock_)
    return *zeroBlock_;

  spmTop_ = llvm::alignTo(spmTop_, kVectorBytes);
  zeroBlock_ = static_cast<int64_t>(spmTop_);
  spmTop_ += kZeroBlockBytes;

  if (spmTop_ > kSpmBytes)
    return at->emitError("no scratchpad left for the zero block of a fill");

  return *zeroBlock_;
}

LogicalResult Translator::emitFill(linalg::FillOp op) {
  auto constant = op.getInputs()[0].getDefiningOp<arith::ConstantOp>();
  auto value = constant ? dyn_cast<FloatAttr>(constant.getValue()) : nullptr;

  if (!value || !value.getValue().isPosZero())
    return op.emitError("MN1 supports only fills with +0.0");

  Value target = op.getOutputs()[0];
  auto type = cast<MemRefType>(target.getType());
  auto it = buffers_.find(target);

  if (it == buffers_.end() || !isContiguousRowMajor(type))
    return op.emitError("MN1 fills need a contiguous buffer it knows");

  int64_t bytes = type.getNumElements() * kElementBytes;
  FailureOr<int> zero = takeVector(op);
  FailureOr<int> cursor = takeScalar(op);
  FailureOr<int> counter = takeScalar(op);

  if (failed(zero) || failed(cursor) || failed(counter))
    return failure();

  out_.instr(llvm::formatv("vzero v{0}", *zero));
  int base = materialize(it->second.base, kScratch1);
  out_.instr(llvm::formatv("addi r{0}, r{1}, 0", *cursor, base));

  if (it->second.local) {
    if (bytes % kVectorBytes != 0)
      return op.emitError("MN1 local fills must cover whole vectors");

    std::string head = out_.freshLabel("fill");
    out_.instr(llvm::formatv("li r{0}, {1}", *counter, bytes / kVectorBytes));
    out_.label(head);
    out_.instr(llvm::formatv("vst v{0}, 0(r{1})", *zero, *cursor));
    out_.instr(llvm::formatv("addi r{0}, r{0}, {1}", *cursor, kVectorBytes));
    out_.instr(llvm::formatv("addi r{0}, r{0}, -1", *counter));
    out_.instr(llvm::formatv("bne r{0}, r0, {1}", *counter, head));
  } else {
    FailureOr<int64_t> block = zeroBlock(op);

    if (failed(block) || bytes % kDmaGranule != 0)
      return op.emitError("MN1 DRAM fills must cover whole DMA granules");

    for (int64_t offset = 0; offset < kZeroBlockBytes; offset += kVectorBytes)
      out_.instr(llvm::formatv("vst v{0}, {1}(r0)", *zero, *block + offset));

    out_.instr(llvm::formatv("li r{0}, {1}", kScratch1, *block));
    int64_t chunks = bytes / kZeroBlockBytes;
    int64_t rest = bytes % kZeroBlockBytes;

    if (chunks > 0) {
      std::string head = out_.freshLabel("fill");
      out_.instr(llvm::formatv("li r{0}, {1}", kScratch0, kZeroBlockBytes));
      out_.instr(llvm::formatv("li r{0}, {1}", *counter, chunks));
      out_.label(head);
      out_.instr(llvm::formatv("dma.st {0}, r{1}, r{2}, r{3}", kInternalTag,
                               kScratch1, *cursor, kScratch0));
      out_.instr(llvm::formatv("dma.wait {0}", kInternalTag));
      out_.instr(
          llvm::formatv("addi r{0}, r{0}, {1}", *cursor, kZeroBlockBytes));
      out_.instr(llvm::formatv("addi r{0}, r{0}, -1", *counter));
      out_.instr(llvm::formatv("bne r{0}, r0, {1}", *counter, head));
    }

    if (rest > 0) {
      out_.instr(llvm::formatv("li r{0}, {1}", kScratch0, rest));
      out_.instr(llvm::formatv("dma.st {0}, r{1}, r{2}, r{3}", kInternalTag,
                               kScratch1, *cursor, kScratch0));
      out_.instr(llvm::formatv("dma.wait {0}", kInternalTag));
    }
  }

  vectorRegs_.release(*zero);
  scalarRegs_.release(*cursor);
  scalarRegs_.release(*counter);
  return success();
}
