#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

NANODSP_OPT="${NANODSP_OPT:-build/bin/nanodsp-opt}"
LLVM_BIN="${LLVM_BIN:-/opt/homebrew/opt/llvm/bin}"
CPU="${BENCH_CPU:-native}"
OUT=build/sweep
L1D_BYTES="${SWEEP_L1D_BYTES:-131072}"
REG_TILE="${SWEEP_REG_TILE:-4, 16, 1}"
TMS="${SWEEP_TM:-16 64 256}"
TNS="${SWEEP_TN:-16 64 128 512}"
TKS="${SWEEP_TK:-8 32 128 512}"
N=512

SYSROOT=()
LLC_TRIPLE=()
if [[ "$(uname)" == Darwin ]]; then
  SYSROOT=(-isysroot "$(xcrun --show-sdk-path)")
  LLC_TRIPLE=(-mtriple="$(uname -m)-apple-macosx$(xcrun --show-sdk-version)")
fi

mkdir -p "${OUT}"
sed -E 's/func\.func @([A-Za-z0-9_]+)\(/func.func @\1_sweep(/' benchmarks/kernels.mlir \
  | awk '/@matmul_512_sweep/{p=1} p{print} p&&/^  }/{exit}' \
  | { echo "module {"; cat; echo "}"; } > "${OUT}/matmul.mlir"

write_schedule() {
  local tm=$1 tn=$2 tk=$3 sizes=() loops=0 types="!transform.any_op"
  for t in "${tm}" "${tn}" "${tk}"; do
    if (( t == N )); then sizes+=(0); else sizes+=("${t}"); loops=$((loops + 1)); fi
  done
  local cache_types="${types}"
  for ((i = 0; i < loops; i++)); do cache_types+=", !transform.any_op"; done
  {
    echo 'module attributes {transform.with_named_sequence} {'
    echo '  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {'
    echo '    %mm = transform.structured.match ops{["linalg.generic"]} attributes {nanodsp.tag = "op0"} in %root : (!transform.any_op) -> !transform.any_op'
    if (( loops > 0 )); then
      echo "    %cache, %cache_loops:${loops} = transform.structured.tile_using_for %mm tile_sizes [${sizes[0]}, ${sizes[1]}, ${sizes[2]}] : (!transform.any_op) -> (${cache_types})"
    else
      echo '    %cache = transform.structured.match ops{["linalg.generic"]} in %root : (!transform.any_op) -> !transform.any_op'
    fi
    echo "    %reg, %reg_loops:3 = transform.structured.tile_using_for %cache tile_sizes [${REG_TILE}] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)"
    echo '    transform.structured.vectorize %reg : !transform.any_op'
    echo '    %funcs = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op'
    echo '    transform.apply_patterns to %funcs { transform.apply_patterns.canonicalization } : !transform.any_op'
    echo '    transform.yield'
    echo '  }'
    echo '}'
  } > "${OUT}/schedule.mlir"
}

echo "tm,tn,tk,working_set_kib,l1d_fraction,ms,gflops,bit_exact"

for tm in ${TMS}; do
  for tn in ${TNS}; do
    for tk in ${TKS}; do
      write_schedule "${tm}" "${tn}" "${tk}"
      "${NANODSP_OPT}" "${OUT}/matmul.mlir" -convert-dsp-to-linalg \
          -nanodsp-optimize=schedule-file="${OUT}/schedule.mlir" \
          -nanodsp-lower-to-llvm \
        | "${LLVM_BIN}/mlir-translate" --mlir-to-llvmir -o "${OUT}/k.ll"
      "${LLVM_BIN}/opt" -O3 "${LLC_TRIPLE[@]}" -mcpu="${CPU}" "${OUT}/k.ll" -o "${OUT}/k.bc"
      "${LLVM_BIN}/llc" -O3 "${LLC_TRIPLE[@]}" -mcpu="${CPU}" -filetype=obj \
        "${OUT}/k.bc" -o "${OUT}/k.o"
      "${LLVM_BIN}/clang++" "${SYSROOT[@]}" -std=c++20 -O3 -mcpu="${CPU}" \
        -ffp-contract=off benchmarks/sweep.cpp "${OUT}/k.o" -o "${OUT}/sweep"
      ws=$(( 4 * (tm * tk + tk * tn + tm * tn) ))
      result=$("${OUT}/sweep" || true)
      echo "${tm},${tn},${tk},$(( ws / 1024 )),$(awk -v w=${ws} -v l=${L1D_BYTES} 'BEGIN{printf "%.2f", w/l}'),${result}"
    done
  done
done
