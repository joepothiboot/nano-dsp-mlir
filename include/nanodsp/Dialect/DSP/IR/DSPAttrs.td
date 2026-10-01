#ifndef NANODSP_DIALECT_DSP_IR_DSPATTRS_TD
#define NANODSP_DIALECT_DSP_IR_DSPATTRS_TD

include "nanodsp/Dialect/DSP/IR/DSPBase.td"
include "mlir/IR/AttrTypeBase.td"

class DSP_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<DSP_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def DSP_LocalMemorySpaceAttr : DSP_Attr<"LocalMemorySpace", "local"> {
  let summary = "Memory space of software-managed local memory (TCM/VTCM).";
  let description = [{
    Marks a memref as living in the target's software-managed local memory
    (a scratchpad such as Hexagon VTCM), not in cached main memory.
    `-nanodsp-promote-local` allocates cache-tile operand buffers here and
    fills them with `memref.dma_start`; `-nanodsp-lower-local` rewrites the
    DMAs into synchronous copies and removes this memory space again, so the
    standard LLVM lowering applies.

    A dialect attribute rather than an integer address space, so it cannot be
    confused with a GPU address space (`1` is global memory on NVVM).

    ```mlir
    %buf = memref.alloc() : memref<2x256x64xf32, #dsp.local>
    ```
  }];
}

#endif // NANODSP_DIALECT_DSP_IR_DSPATTRS_TD
