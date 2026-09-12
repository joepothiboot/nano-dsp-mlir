#!/bin/bash
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
ROOT_DIR="${SCRIPT_PATH%/*}"
[[ "${ROOT_DIR}" == "${SCRIPT_PATH}" ]] && ROOT_DIR="."
ROOT_DIR="$(cd "${ROOT_DIR}" && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"

if [[ -z "${MLIR_DIR:-}" ]]; then
  if command -v llvm-config >/dev/null 2>&1; then
    MLIR_DIR="$(llvm-config --cmakedir)"
  else
    echo "Set MLIR_DIR to the directory containing MLIRConfig.cmake." >&2
    exit 1
  fi
fi

TABLEGEN_EXE="${MLIR_TABLEGEN_EXE:-}"
if [[ -z "${TABLEGEN_EXE}" ]] && command -v mlir-tblgen >/dev/null 2>&1; then
  TABLEGEN_EXE="$(command -v mlir-tblgen)"
fi

LIT_EXE="${LLVM_EXTERNAL_LIT:-}"
if [[ -z "${LIT_EXE}" ]] && command -v lit >/dev/null 2>&1; then
  LIT_EXE="$(command -v lit)"
fi

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -G Ninja \
  -DMLIR_DIR="${MLIR_DIR}" \
  ${TABLEGEN_EXE:+-DMLIR_TABLEGEN_EXE="${TABLEGEN_EXE}"} \
  ${LIT_EXE:+-DLLVM_EXTERNAL_LIT="${LIT_EXE}"}

cmake --build "${BUILD_DIR}" --target check-nanodsp

# If nanodsp-opt fails due to missing conversion passes:
# 1. Check lib/Conversion/CMakeLists.txt: it MUST add_mlir_library 
#    and register the pass header.
# 2. Check if your nanodsp-opt.cpp includes the pass header generated 
#    by TableGen (Passes.h.inc).