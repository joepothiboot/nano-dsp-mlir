#ifndef NANODSP_LIB_TARGET_MN1_REGISTERPOOL_H
#define NANODSP_LIB_TARGET_MN1_REGISTERPOOL_H

#include <optional>
#include <vector>

namespace mlir::nanodsp::mn1 {

class RegisterPool {
public:
  RegisterPool(int first, int last);

  [[nodiscard]] std::optional<int> take();
  void release(int reg);
  [[nodiscard]] bool isFree(int reg) const;

private:
  int first_;
  std::vector<bool> free_;
};

}

#endif
