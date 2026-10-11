#include "RegisterPool.h"

#include <cassert>
#include <cstddef>

using namespace mlir::nanodsp::mn1;

RegisterPool::RegisterPool(int first, int last)
    : first_(first), free_(static_cast<std::size_t>(last - first + 1), true) {}

std::optional<int> RegisterPool::take() {
  for (std::size_t i = 0; i < free_.size(); ++i) {
    if (!free_[i])
      continue;

    free_[i] = false;
    return first_ + static_cast<int>(i);
  }

  return std::nullopt;
}

void RegisterPool::release(int reg) {
  auto index = static_cast<std::size_t>(reg - first_);
  assert(index < free_.size() && !free_[index] && "releasing a free register");
  free_[index] = true;
}

bool RegisterPool::isFree(int reg) const {
  return free_[static_cast<std::size_t>(reg - first_)];
}
