#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

OUT=build/bench/t4
BIN=build/cuda-bench
RUN_MOJO="${RUN_MOJO:-1}"
mkdir -p "${OUT}"

{
  nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem --format=csv
  nvcc --version | tail -n 2
  python3 -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__)"
} | tee "${OUT}/env.txt"

nvcc -std=c++20 -O3 -arch=sm_75 -Xcompiler -ffp-contract=off \
  -o "${BIN}" cuda/bench.cu -lcublas

"${BIN}" --check
"${BIN}" > "${OUT}/cuda-t4.json"
python3 triton/matmul.py > "${OUT}/triton-t4.json" 2> "${OUT}/triton-t4.log" ||
  echo "triton failed, see ${OUT}/triton-t4.log"
cat "${OUT}/triton-t4.log"

ncu --csv --page details \
  --section SpeedOfLight --section Occupancy --section LaunchStats \
  --section MemoryWorkloadAnalysis --section WarpStateStats \
  --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
  "${BIN}" --profile > "${OUT}/ncu-t4.csv" 2> "${OUT}/ncu-t4.log" ||
  echo "ncu failed, see ${OUT}/ncu-t4.log"

if [[ "${RUN_MOJO}" == "1" ]]; then
  if ! command -v pixi > /dev/null; then
    curl -fsSL https://pixi.sh/install.sh | bash
    export PATH="${HOME}/.pixi/bin:${PATH}"
  fi

  pixi run test-gpu
  pixi run bench-gpu
  cp build/bench/results-gpu.json "${OUT}/mojo-t4.json"
fi

tar czf build/t4-results.tar.gz -C "${OUT}" .
echo "wrote build/t4-results.tar.gz"
