#ifndef NANODSP_CONVERSION_PASSES_H
#define NANODSP_CONVERSION_PASSES_H

#include "nanodsp/Conversion/DSPToLinalg/DSPToLinalg.h"

namespace mlir {
namespace nanodsp {

#define GEN_PASS_REGISTRATION
#include "nanodsp/Conversion/Passes.h.inc"

} // namespace nanodsp
} // namespace mlir

#endif // NANODSP_CONVERSION_PASSES_H