#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

BIN="${NANODSP_BIN:-build/bin}"
MNPU_SIM="${MNPU_SIM:-../mock-npu-bench/build/mnpu-sim}"
OUT=build/sweep-mn1
REG_TILE="${SWEEP_REG_TILE:-4, 16, 1}"
TMS="${SWEEP_TM:-8 16 32 64}"
TNS="${SWEEP_TN:-32 64 128}"
TKS="${SWEEP_TK:-8 16 32 64 128}"
N=128

mkdir -p "${OUT}/data"
"${BIN}/nanodsp-golden" matmul ${N} ${N} ${N} "${OUT}/data"

cat > "${OUT}/matmul.mlir" <<EOF
func.func @mm(%a: tensor<${N}x${N}xf32>, %b: tensor<${N}x${N}xf32>) -> tensor<${N}x${N}xf32> {
  %c = dsp.matmul %a, %b : (tensor<${N}x${N}xf32>, tensor<${N}x${N}xf32>) -> tensor<${N}x${N}xf32>
  return %c : tensor<${N}x${N}xf32>
}
EOF

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
      echo "    transform.annotate %cache_loops#$((loops - 1)) \"nanodsp.cache_loop\" : !transform.any_op"
    else
      echo '    %cache = transform.structured.match ops{["linalg.generic"]} in %root : (!transform.any_op) -> !transform.any_op'
    fi
    echo "    %reg, %reg_loops:3 = transform.structured.tile_using_for %cache tile_sizes [${REG_TILE}] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)"
    echo '    %funcs = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op'
    echo '    transform.apply_patterns to %funcs {'
    echo '      transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices'
    echo '      transform.apply_patterns.canonicalization'
    echo '    } : !transform.any_op'
    echo '    %folded = transform.structured.match ops{["linalg.generic"]} in %reg_loops#2 : (!transform.any_op) -> !transform.any_op'
    echo '    transform.structured.vectorize %folded : !transform.any_op'
    echo '    transform.apply_patterns to %funcs { transform.apply_patterns.canonicalization } : !transform.any_op'
    echo '    transform.yield'
    echo '  }'
    echo '}'
  } > "${OUT}/schedule.mlir"
}

run() {
  local label=$1 tm=$2 tn=$3 tk=$4 optimize=$5
  local ir="${OUT}/k.mlir" asm="${OUT}/k.s" log="${OUT}/k.log"

  if ! "${BIN}/nanodsp-opt" "${OUT}/matmul.mlir" -convert-dsp-to-linalg \
      "${optimize}" -nanodsp-bufferize -nanodsp-promote-local=target=mn1 \
      -o "${ir}" 2> "${log}"; then
    echo "${label},${tm},${tn},${tk},,,,,does not fit"
    return
  fi

  if ! "${BIN}/nanodsp-translate" --mlir-to-mn1 "${ir}" -o "${asm}" 2> "${log}"; then
    echo "${label},${tm},${tn},${tk},,,,,$(grep -o 'error: .*' "${log}" | head -1 | tr ',' ';')"
    return
  fi

  "${MNPU_SIM}" "${asm}" --load 0 "${OUT}/data/a.f32" --load 0x40000 "${OUT}/data/b.f32" \
    --expect-bits 0x80000 "${OUT}/data/c_ref.f32" > "${log}" 2>&1 || true

  local cycles raw dma exact
  cycles=$(awk '$1 == "cycles" {print $2}' "${log}")
  raw=$(awk '$1 == "raw" {print $3}' "${log}")
  dma=$(awk '$1 == "dma-wait" {print $3}' "${log}")
  exact=$(grep -q 'bit-exact' "${log}" && echo yes || echo no)
  local bytes=$(( 4 * (2 * tm * tk + 2 * tk * tn + tm * tn) ))
  echo "${label},${tm},${tn},${tk},$(( bytes / 1024 )),${cycles},${raw},${dma},${exact}"
}

echo "label,tm,tn,tk,local_kib_doubled,cycles,raw_stalls,dma_wait_stalls,bit_exact"
run model 32 128 16 -nanodsp-optimize=target=mn1

for tm in ${TMS}; do
  for tn in ${TNS}; do
    for tk in ${TKS}; do
      write_schedule "${tm}" "${tn}" "${tk}"
      run sweep "${tm}" "${tn}" "${tk}" -nanodsp-optimize=schedule-file="${OUT}/schedule.mlir"
    done
  done
done
