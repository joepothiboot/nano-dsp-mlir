#ifndef NANODSP_DIALECT_DSP_IR_DSPBASE_TD
#define NANODSP_DIALECT_DSP_IR_DSPBASE_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def DSP_Dialect : Dialect {
  let name = "dsp";
  let summary = "High-level math/image pipeline ops (nano-dsp-mlir L1).";
  let description = [{
    The `dsp` dialect is the graph-level entry point of nano-dsp-mlir. Ops have
    value semantics over statically shaped `f32` tensors, carry no loop
    structure, and no memory. It is deliberately tiny: it exists to be lowered
    to `linalg.generic` (L2), not to be optimized in place.
  }];
  let cppNamespace = "::mlir::nanodsp";
}

//===----------------------------------------------------------------------===//
// Type constraints -- L1 invariants are enforced in the *type system*, so
// ill-typed IR cannot even parse. (dynamic shapes / non-f32 are rejected at
// parse time, not by a verifier that a later pass might forget to run.)
//===----------------------------------------------------------------------===//

def DSP_Tensor : Type<
  StaticShapeTensorOf<[F32]>.predicate,
  "statically shaped tensor of f32",
  "::mlir::RankedTensorType">;

class DSP_TensorOfRank<int rank> : Type<
  And<[
    StaticShapeTensorOf<[F32]>.predicate,
    CPred<"::llvm::cast<::mlir::ShapedType>($_self).getRank() == " # rank>
  ]>,
  "statically shaped rank-" # rank # " tensor of f32",
  "::mlir::RankedTensorType">;

class DSP_Op<string mnemonic, list<Trait> traits = []>
    : Op<DSP_Dialect, mnemonic, traits>;

#endif // NANODSP_DIALECT_DSP_IR_DSPBASE_TD