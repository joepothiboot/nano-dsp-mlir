#ifndef NANODSP_LIB_TARGET_MN1_VALUES_H
#define NANODSP_LIB_TARGET_MN1_VALUES_H

#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir::nanodsp::mn1 {

struct Scalar {
  int64_t constant = 0;
  int reg = -1;
  int64_t scale = 1;
  bool owned = true;

  static Scalar of(int64_t value) { return {value, -1, 1, false}; }

  static Scalar inReg(int r) { return {0, r, 1, true}; }

  static Scalar borrowed(int r, int64_t scale = 1, int64_t constant = 0) {
    return {constant, r, scale, false};
  }

  [[nodiscard]] bool isConstant() const { return reg < 0; }

  [[nodiscard]] bool isPlainReg() const {
    return reg >= 0 && scale == 1 && constant == 0;
  }

  [[nodiscard]] bool ownsReg() const { return reg >= 0 && owned; }
};

struct Term {
  Scalar value;
  int64_t coeff = 1;
};

struct Buffer {
  Scalar base;
  bool local = false;
  llvm::SmallVector<int64_t> shape;
  llvm::SmallVector<int64_t> strides;
};

struct VectorParts {
  int64_t rows = 1;
  int64_t partsPerRow = 1;
  llvm::SmallVector<int> regs;

  [[nodiscard]] int at(int64_t row, int64_t part) const {
    return regs[(row * partsPerRow) + part];
  }
};

struct TagRef {
  int base = 0;
  int slotReg = -1;
  bool inverted = false;
};

}

#endif
