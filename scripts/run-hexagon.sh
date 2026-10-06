#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLVM_BIN="${LLVM_BIN:-/opt/homebrew/opt/llvm/bin}"
IMAGE="${HEXAGON_IMAGE:-nanodsp-hexagon}"
QEMU="${QEMU_HEXAGON:-qemu-hexagon}"
OUT=build/hexagon
TARGET=hexagon-unknown-linux-musl
HVX="+hvxv68,+hvx-length128b"
HVX_FP=0
[[ "${1:-}" == "--hvx-fp" ]] && HVX_FP=1

cd "${ROOT}"
mkdir -p "${OUT}"

in_image() {
  docker run --rm --platform linux/amd64 -v "${ROOT}:/work" -w /work \
    "${IMAGE}" sh -c "$1"
}

kernel_object() {
  build/bin/nanodsp-opt "test/Hexagon/$1.mlir" -convert-dsp-to-linalg $3 \
      -nanodsp-lower-to-llvm="${5:-generic-alloc}" \
    | "${LLVM_BIN}/mlir-translate" --mlir-to-llvmir \
    | "${LLVM_BIN}/llc" -O2 -mtriple="${TARGET}" -mcpu=hexagonv68 \
        -mattr="$4" -hexagon-small-data-threshold=0 -filetype=obj \
        -o "${OUT}/$1-$2.o"
}

SCHED=-nanodsp-optimize=target=hexagon-hvx128
kernel_object kernels-f32 scalar "" "${HVX},+hvx-ieee-fp"
kernel_object kernels-i8 scalar "" "${HVX}"
kernel_object kernels-i8 hvx "${SCHED}" "${HVX}"
kernel_object kernels-f32 hvx-ieee "${SCHED}" "${HVX},+hvx-ieee-fp"
kernel_object kernels-f32 hvx-qf32 "${SCHED}" "${HVX}"
LOCAL="generic-alloc local-target=hexagon-hvx128"
kernel_object kernels-local-f32 scalar "${SCHED}" "" "${LOCAL}"
kernel_object kernels-local-i8 hvx "${SCHED}" "${HVX}" "${LOCAL}"

if [[ ! -d "${OUT}/sysroot/usr/include/c++" ]]; then
  mkdir -p "${OUT}/sysroot"
  docker run --rm --platform linux/amd64 "${IMAGE}" \
      tar -C "/opt/hexagon/toolchain/target/${TARGET}" -cf - \
        --exclude='usr/include/linux/netfilter*' usr/include \
    | tar -xf - -C "${OUT}/sysroot"
fi

for variant in harness harness-local; do
  defines=""
  [[ "${variant}" == harness-local ]] && defines=-DNANODSP_LOCAL_KERNELS
  "${LLVM_BIN}/clang++" --target="${TARGET}" --sysroot="${OUT}/sysroot" -mv68 \
    -O2 -std=c++20 -ffp-contract=off ${defines} -c test/Hexagon/harness.cpp \
    -o "${OUT}/${variant}.o"
done

builds=("scalar harness kernels-f32-scalar kernels-i8-scalar"
        "hvx-int harness kernels-f32-scalar kernels-i8-hvx"
        "local harness-local kernels-f32-scalar kernels-i8-hvx
           kernels-local-f32-scalar kernels-local-i8-hvx")
if (( HVX_FP )); then
  builds+=("hvx-ieee harness kernels-f32-hvx-ieee kernels-i8-hvx"
           "hvx-qf32 harness kernels-f32-hvx-qf32 kernels-i8-hvx")
fi

status=0
for b in "${builds[@]}"; do
  read -r name harness kernels <<< "$(echo ${b})"
  objects=""
  for k in ${kernels}; do objects+=" ${OUT}/${k}.o"; done
  echo "== ${name}"
  in_image "clang++ --target=${TARGET} -static ${OUT}/${harness}.o \
      ${objects} -o ${OUT}/harness-${name} &&
    ${QEMU} ${OUT}/harness-${name}" || status=1
done
exit "${status}"
