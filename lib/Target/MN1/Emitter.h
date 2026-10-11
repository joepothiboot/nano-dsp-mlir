#ifndef NANODSP_LIB_TARGET_MN1_EMITTER_H
#define NANODSP_LIB_TARGET_MN1_EMITTER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace mlir::nanodsp::mn1 {

class Emitter {
public:
  explicit Emitter(llvm::raw_ostream &os) : os_(os) {}

  void instr(const llvm::Twine &text) { os_ << "  " << text << "\n"; }

  void label(llvm::StringRef name) { os_ << name << ":\n"; }

  void note(const llvm::Twine &text) { os_ << "; " << text << "\n"; }

  [[nodiscard]] std::string freshLabel(llvm::StringRef hint) {
    return (hint + llvm::Twine(next_++)).str();
  }

private:
  llvm::raw_ostream &os_;
  unsigned next_ = 0;
};

}

#endif
