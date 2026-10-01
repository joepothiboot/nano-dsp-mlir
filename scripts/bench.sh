#!/bin/bash
# Stage 5 benchmark: the dsp kernels in benchmarks/kernels.mlir compiled by
# nanodsp-opt (untiled, and with the host-neon TargetModel schedule) against
# the scalar C++ reference, plus measured roofline ceilings.
#
#   scripts/bench.sh --check   build everything, check every kernel bit for bit
#                              against reference/nanodsp_ref.h, no timing (CI)
#   scripts/bench.sh           check, then time; writes build/bench/results.json
#
# Environment:
#   NANODSP_OPT   nanodsp-opt to use (default build/bin/nanodsp-opt; ./test.sh)
#   LLVM_BIN      mlir-translate, llc, clang++ (default: Homebrew LLVM)
#   BENCH_ARGS    extra harness arguments, e.g. "--samples 20 --filter matmul"
#   BENCH_IR_OPT  LLVM IR optimization of the kernels before llc (default
#                 -O3, as clang -O3 gives the C++ reference); "none" skips it
#   BENCH_CPU     -mcpu for llc and clang++ (default native)
#
# Every configuration compiles the same kernels file with its functions
# renamed `<name>_<config>` (a sed over the func.func names), so all
# configurations link into one harness binary without symbol clashes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CHECK_ONLY=0
case "${1:-}" in
  --check) CHECK_ONLY=1 ;;
  "") ;;
  *) echo "usage: $0 [--check]" >&2; exit 2 ;;
esac

NANODSP_OPT="${NANODSP_OPT:-build/bin/nanodsp-opt}"
if [[ -z "${LLVM_BIN:-}" ]]; then
  if [[ -x /opt/homebrew/opt/llvm/bin/llc ]]; then
    LLVM_BIN=/opt/homebrew/opt/llvm/bin
  else
    LLVM_BIN="$(brew --prefix llvm)/bin"
  fi
fi
OUT=build/bench
KERNELS=benchmarks/kernels.mlir
CPU="${BENCH_CPU:-native}"
IR_OPT="${BENCH_IR_OPT:--O3}"

if [[ ! -x "${NANODSP_OPT}" ]]; then
  echo "error: ${NANODSP_OPT} not found; run ./test.sh or set NANODSP_OPT" >&2
  exit 1
fi

# Homebrew clang does not find the macOS SDK on its own; the kernel objects
# get the SDK's macOS version so the linker does not warn about a mismatch.
SYSROOT=()
LLC_TRIPLE=()
if [[ "$(uname)" == Darwin ]]; then
  SYSROOT=(-isysroot "$(xcrun --show-sdk-path)")
  LLC_TRIPLE=(-mtriple="$(uname -m)-apple-macosx$(xcrun --show-sdk-version)")
fi
CXXFLAGS=(-std=c++20 -O3 -mcpu="${CPU}" -ffp-contract=off -Wall -Wextra)

mkdir -p "${OUT}"

# kernel_object <config> <nanodsp-opt schedule pass, or "">
kernel_object() {
  local cfg=$1 sched=$2
  sed -E "s/func\.func @([A-Za-z0-9_]+)\(/func.func @\1_${cfg}(/" "${KERNELS}" \
    > "${OUT}/kernels-${cfg}.mlir"
  # shellcheck disable=SC2086  # $sched is empty or one pass
  "${NANODSP_OPT}" "${OUT}/kernels-${cfg}.mlir" -convert-dsp-to-linalg ${sched} \
      -nanodsp-lower-to-llvm \
    | "${LLVM_BIN}/mlir-translate" --mlir-to-llvmir -o "${OUT}/kernels-${cfg}.ll"
  # llc alone runs no IR-level passes: without opt, every scalar `+=` goes
  # through memory (no promotion of the accumulator), which is not what
  # clang -O3 does to the reference. opt adds no fast-math flags, so it can
  # neither reassociate nor contract; the harness checks that anyway.
  local ll="${OUT}/kernels-${cfg}.ll"
  if [[ "${IR_OPT}" != none ]]; then
    "${LLVM_BIN}/opt" "${IR_OPT}" "${LLC_TRIPLE[@]}" -mcpu="${CPU}" \
      "${ll}" -o "${OUT}/kernels-${cfg}.bc"
    ll="${OUT}/kernels-${cfg}.bc"
  fi
  "${LLVM_BIN}/llc" -O3 "${LLC_TRIPLE[@]}" -mcpu="${CPU}" -filetype=obj \
    "${ll}" -o "${OUT}/kernels-${cfg}.o"
}

echo "== compiling kernels (${NANODSP_OPT})"
kernel_object untiled ""
kernel_object scheduled -nanodsp-optimize=target=host-neon

# The bit-exactness argument (docs/05-soundness.md) needs separate fmul and
# fadd; fail early if llc fused anything.
if "${LLVM_BIN}/llvm-objdump" -d "${OUT}"/kernels-*.o | grep -qE '\bfml[as]\b|\bfn?madd\b|\bfn?msub\b'; then
  echo "error: FMA instructions in the kernel objects" >&2
  exit 1
fi

echo "== building harness"
"${LLVM_BIN}/clang++" "${SYSROOT[@]}" "${CXXFLAGS[@]}" \
  benchmarks/harness.cpp "${OUT}"/kernels-untiled.o \
  "${OUT}"/kernels-scheduled.o -o "${OUT}/harness"

echo "== building ceilings"
"${LLVM_BIN}/clang++" "${SYSROOT[@]}" "${CXXFLAGS[@]}" \
  benchmarks/ceilings.cpp -o "${OUT}/ceilings"

if (( CHECK_ONLY )); then
  "${OUT}/harness" --check
  exit 0
fi

echo "== ceilings"
"${OUT}/ceilings" --json "${OUT}/ceilings.json"

echo "== kernels"
# shellcheck disable=SC2086  # BENCH_ARGS is a word list
"${OUT}/harness" --json "${OUT}/benchmarks.json" ${BENCH_ARGS:-}

# --- context ---------------------------------------------------------------
json_str() { # escape for a JSON string
  printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' | tr -d '\n'
}
first_line() { "$@" 2>/dev/null | head -n 1; }

if [[ "$(uname)" == Darwin ]]; then
  cpu=$(sysctl -n machdep.cpu.brand_string)
  ncpu=$(sysctl -n hw.ncpu)
  mem=$(sysctl -n hw.memsize)
  l1d=$(sysctl -n hw.perflevel0.l1dcachesize 2>/dev/null || sysctl -n hw.l1dcachesize)
  l2=$(sysctl -n hw.perflevel0.l2cachesize 2>/dev/null || sysctl -n hw.l2cachesize)
  os="macOS $(sw_vers -productVersion) ($(uname -m))"
else
  cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //' || uname -m)
  ncpu=$(nproc)
  mem=$(( $(grep MemTotal /proc/meminfo | awk '{print $2}') * 1024 ))
  l1d=$(getconf LEVEL1_DCACHE_SIZE 2>/dev/null || echo 0)
  l2=$(getconf LEVEL2_CACHE_SIZE 2>/dev/null || echo 0)
  os="$(uname -sr) ($(uname -m))"
fi
commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
dirty=false
git diff --quiet HEAD 2>/dev/null || dirty=true

{
  printf '{\n  "context": {\n'
  printf '    "date": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '    "cpu": "%s",\n' "$(json_str "${cpu}")"
  printf '    "num_cpus": %s,\n' "${ncpu}"
  printf '    "threads": 1,\n'
  printf '    "memory_bytes": %s,\n' "${mem}"
  printf '    "caches": {"l1d_bytes": %s, "l2_bytes": %s, "note": "%s"},\n' \
    "${l1d}" "${l2}" "performance-core values where the OS reports them"
  printf '    "os": "%s",\n' "$(json_str "${os}")"
  printf '    "commit": "%s",\n' "${commit}"
  printf '    "dirty": %s,\n' "${dirty}"
  printf '    "nanodsp_opt": "%s",\n' "$(json_str "${NANODSP_OPT}")"
  printf '    "llc": "%s",\n' "$(json_str "$(first_line "${LLVM_BIN}/llc" --version | sed 's/^ *//'; "${LLVM_BIN}/llc" --version | sed -n 2p | sed 's/^ *//')")"
  printf '    "clang++": "%s",\n' "$(json_str "$(first_line "${LLVM_BIN}/clang++" --version)")"
  printf '    "kernel_flags": "%s",\n' "nanodsp-opt -convert-dsp-to-linalg [schedule] -nanodsp-lower-to-llvm | mlir-translate --mlir-to-llvmir | opt ${IR_OPT} | llc -O3 -mcpu=${CPU}"
  printf '    "cxx_flags": "%s",\n' "$(json_str "${CXXFLAGS[*]}")"
  printf '    "timing": "%s"\n' "steady_clock; warmup call, calls per sample doubled until >= min-time, best per-call time over samples; results checked bit-exact against nanodsp::ref first"
  printf '  },\n  "ceilings": '
  cat "${OUT}/ceilings.json"
  printf ',\n  "benchmarks": '
  cat "${OUT}/benchmarks.json"
  printf '\n}\n'
} > "${OUT}/results.json"

if command -v python3 > /dev/null; then
  python3 -m json.tool "${OUT}/results.json" > /dev/null
fi
echo "== wrote ${OUT}/results.json"
