#!/bin/bash
# Run the nano-dsp-mlir kernels on an emulated Hexagon V68 (with HVX) and
# check every result bit for bit against the scalar C++ reference, computed
# on the same emulated core (test/Hexagon/harness.cpp).
#
#   ./test.sh                                                   # nanodsp-opt
#   docker build --platform linux/amd64 -t nanodsp-hexagon docker/hexagon
#   scripts/run-hexagon.sh
#
# Everything is compiled on the host: the kernels with nanodsp-opt + llc,
# the harness with Homebrew clang against the image's sysroot headers. The
# image only links (musl, libc++, compiler-rt) and runs qemu-hexagon.
#
# Builds, as f32 kernels + int8 kernels:
#   scalar       unscheduled + unscheduled
#   hvx-int      unscheduled + HVX (-nanodsp-optimize=target=hexagon-hvx128)
#   hvx-ieee     HVX, IEEE float (+hvx-ieee-fp) + HVX        [needs --hvx-fp]
#   hvx-qf32     HVX, QFloat (no +hvx-ieee-fp) + HVX         [needs --hvx-fp]
#
# The last two need a qemu-hexagon with the HVX floating-point instructions,
# which QEMU 8.2 (the image's) lacks. Pass --hvx-fp to run them with the
# qemu-hexagon named by $QEMU_HEXAGON (inside the container).
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

# kernel_object <kernels file stem> <build name> <nanodsp-opt schedule> <attrs>
kernel_object() {
  # Relative paths: MLIR pass options split on whitespace, and the checkout
  # path may contain some.
  build/bin/nanodsp-opt "test/Hexagon/$1.mlir" -convert-dsp-to-linalg $3 \
      -nanodsp-lower-to-llvm=generic-alloc \
    | "${LLVM_BIN}/mlir-translate" --mlir-to-llvmir \
    | "${LLVM_BIN}/llc" -O2 -mtriple="${TARGET}" -mcpu=hexagonv68 \
        -mattr="$4" -hexagon-small-data-threshold=0 -filetype=obj \
        -o "${OUT}/$1-$2.o"
  # (Small data is off because the toolchain's lld rejects the GP-relative
  # relocations llc uses for constants by default.)
}

SCHED=-nanodsp-optimize=target=hexagon-hvx128
kernel_object kernels-f32 scalar "" "${HVX},+hvx-ieee-fp"
kernel_object kernels-i8 scalar "" "${HVX}"
kernel_object kernels-i8 hvx "${SCHED}" "${HVX}"
kernel_object kernels-f32 hvx-ieee "${SCHED}" "${HVX},+hvx-ieee-fp"
kernel_object kernels-f32 hvx-qf32 "${SCHED}" "${HVX}"

# Headers for the host-side harness compile, exported from the image once.
# netfilter headers are skipped: some differ only in case, which a
# case-insensitive (macOS) filesystem cannot hold, and nothing uses them.
if [[ ! -d "${OUT}/sysroot/usr/include/c++" ]]; then
  mkdir -p "${OUT}/sysroot"
  docker run --rm --platform linux/amd64 "${IMAGE}" \
      tar -C "/opt/hexagon/toolchain/target/${TARGET}" -cf - \
        --exclude='usr/include/linux/netfilter*' usr/include \
    | tar -xf - -C "${OUT}/sysroot"
fi

# Scalar C++, built like the host reference: -ffp-contract=off, no FMA.
"${LLVM_BIN}/clang++" --target="${TARGET}" --sysroot="${OUT}/sysroot" -mv68 \
  -O2 -std=c++20 -ffp-contract=off -c test/Hexagon/harness.cpp \
  -o "${OUT}/harness.o"

builds=("scalar kernels-f32-scalar kernels-i8-scalar"
        "hvx-int kernels-f32-scalar kernels-i8-hvx")
if (( HVX_FP )); then
  builds+=("hvx-ieee kernels-f32-hvx-ieee kernels-i8-hvx"
           "hvx-qf32 kernels-f32-hvx-qf32 kernels-i8-hvx")
fi

status=0
for b in "${builds[@]}"; do
  read -r name f32 i8 <<< "${b}"
  echo "== ${name}"
  in_image "clang++ --target=${TARGET} -static ${OUT}/harness.o \
      ${OUT}/${f32}.o ${OUT}/${i8}.o -o ${OUT}/harness-${name} &&
    ${QEMU} ${OUT}/harness-${name}" || status=1
done
exit "${status}"
