#ifndef NANODSP_TARGET_MN1_TRANSLATETOMN1_H
#define NANODSP_TARGET_MN1_TRANSLATETOMN1_H

#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
class Operation;
}

namespace mlir::nanodsp {

LogicalResult translateToMN1(Operation *op, llvm::raw_ostream &os);

void registerToMN1Translation();

}

#endif
