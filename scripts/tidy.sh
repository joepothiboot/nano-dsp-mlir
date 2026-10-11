#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

BUILD_DIR="${BUILD_DIR:-build}"
LLVM_BIN="${LLVM_BIN:-$(brew --prefix llvm)/bin}"
EXTRA=()

if command -v xcrun > /dev/null; then
  EXTRA+=("-isysroot" "$(xcrun --show-sdk-path)")
fi

if [[ ! -f "${BUILD_DIR}/compile_commands.json" ]]; then
  echo "error: ${BUILD_DIR}/compile_commands.json not found; run ./test.sh first" >&2
  exit 1
fi

"${LLVM_BIN}/run-clang-tidy" -quiet -p "${BUILD_DIR}" \
  -clang-tidy-binary "${LLVM_BIN}/clang-tidy" \
  ${EXTRA[@]/#/-extra-arg=} \
  "nano-dsp-mlir/(lib|tools)/"

"${LLVM_BIN}/clang-tidy" --quiet reference/test_reference.cpp \
  --header-filter='nano-dsp-mlir/reference/' -- -std=c++20 "${EXTRA[@]}"

"${LLVM_BIN}/clang-tidy" --quiet cuda/harness.h \
  --header-filter='nano-dsp-mlir/cuda/' -- -x c++ -std=c++20 "${EXTRA[@]}"

echo "clang-tidy clean"
