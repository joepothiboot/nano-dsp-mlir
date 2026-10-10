#ifndef NANODSP_CONVERSION_DSPTOLINALG_DSPTOLINALG_H
#define NANODSP_CONVERSION_DSPTOLINALG_DSPTOLINALG_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace mlir::nanodsp {

void populateDSPToLinalgPatterns(RewritePatternSet &patterns);

#define GEN_PASS_DECL_CONVERTDSPTOLINALG
#include "nanodsp/Conversion/Passes.h.inc"

}

#endif
