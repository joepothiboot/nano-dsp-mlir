#!/bin/bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 SOURCE.mojo FEATURES OUT.o" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLVM_BIN="${LLVM_BIN:-/opt/homebrew/opt/llvm/bin}"
SOURCE=$1
FEATURES=$2
OUT=$3
CPU=hexagonv68
IR="${OUT%.o}.ll"

cd "${ROOT}"
pixi run --quiet mojo build --emit llvm --target-triple riscv32-unknown-elf \
  --fp-mode contract=off -I mojo -o "${IR}.rv32" "${SOURCE}"

if grep -qE '^declare' "${IR}.rv32" &&
  grep -E '^declare' "${IR}.rv32" | grep -qv '@llvm\.'; then
  echo "error: ${SOURCE} needs runtime symbols a bare Hexagon binary lacks:" >&2
  grep -E '^declare' "${IR}.rv32" | grep -v '@llvm\.' >&2
  exit 1
fi

sed -E '/^target (datalayout|triple)/d;
  s/"target-cpu"( "target-features"="[^"]*")?/"target-cpu"="'"${CPU}"'" "target-features"="'"${FEATURES}"'"/' \
  "${IR}.rv32" > "${IR}"

"${LLVM_BIN}/llc" -O2 -mtriple=hexagon-unknown-linux-musl -mcpu="${CPU}" \
  -mattr="${FEATURES}" -fp-contract=off \
  -hexagon-small-data-threshold=0 -filetype=obj "${IR}" -o "${OUT}"
