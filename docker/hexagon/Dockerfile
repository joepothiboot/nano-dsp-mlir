# Hexagon cross toolchain + user-mode emulator for running nano-dsp-mlir
# kernels on an emulated Hexagon core (HVX included).
#
#   docker build --platform linux/amd64 -t nanodsp-hexagon docker/hexagon
#   docker run --rm --platform linux/amd64 -v "$PWD:/work" -w /work \
#     nanodsp-hexagon <command>
#
# linux/amd64 is the configuration QEMU's own Hexagon CI container uses; on
# Apple silicon Docker runs it emulated.
#
# Everything is compiled on the host (nanodsp-opt + llc for the kernels,
# Homebrew clang against this image's sysroot for the harness); the image
# links the objects against musl/libc++/compiler-rt and runs the result under
# qemu-hexagon. See scripts/run-hexagon.sh.
#
# The harness is not compiled in here because the toolchain's clang
# mishandles header lookup in this environment: a lookup that misses in one
# search directory is a hard "cannot open file" error instead of moving on,
# so __has_include(<features.h>) in libc++'s headers fails. Linking does no
# header search and works.
FROM ubuntu:24.04

# qemu-user ships qemu-hexagon (linux-user mode) since QEMU 6.0.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl zstd qemu-user \
 && rm -rf /var/lib/apt/lists/*

# CodeLinaro's prebuilt clang/LLVM for hexagon-unknown-linux-musl, with the
# musl sysroot and libc++. The host build must match the container's arch.
ARG TOOLCHAIN_VERSION=22.1.8
ARG TARGETARCH
RUN set -eu; \
    case "${TARGETARCH:-$(dpkg --print-architecture)}" in \
      arm64) host=aarch64-linux-musl; \
             sha=2d3c926981445ca71daf308ceebff2f7b26c6b32873ef7209c858f0034555c87 ;; \
      amd64) host=x86_64-linux-musl; \
             sha=ec8c319c2938a56c620a1bb40268c69818192ad16f7bc104de62b12097381f43 ;; \
      *) echo "unsupported host arch" >&2; exit 1 ;; \
    esac; \
    name="clang+llvm-${TOOLCHAIN_VERSION}-cross-hexagon-unknown-linux-musl_${host}"; \
    curl -fsSL -o /tmp/toolchain.tar.zst \
      "https://artifacts.codelinaro.org/artifactory/codelinaro-toolchain-for-hexagon/${TOOLCHAIN_VERSION}/${name}.tar.zst"; \
    echo "${sha}  /tmp/toolchain.tar.zst" | sha256sum -c -; \
    mkdir -p /opt/hexagon; \
    tar --zstd -xf /tmp/toolchain.tar.zst -C /opt/hexagon --strip-components=1; \
    rm /tmp/toolchain.tar.zst

# The toolchain binaries are themselves linked against musl (for the host);
# Ubuntu's musl package provides their loader, /lib/ld-musl-<arch>.so.1.
RUN apt-get update \
 && apt-get install -y --no-install-recommends musl \
 && rm -rf /var/lib/apt/lists/*

# The archive unpacks to a per-host directory (e.g. x86_64-linux-musl/);
# give it a stable name. The sysroot is in its target/ subdirectory.
RUN ln -s "$(ls -d /opt/hexagon/*-linux-musl)" /opt/hexagon/toolchain

# The 22.1.8 host archives ship a dangling libclang_rt.builtins-hexagon.a
# symlink: the compiler-rt builtins it points to are missing. Take them from
# the same release's standalone package.
RUN set -eu; \
    deb=libclang-rt-22-builtins-hexagon-cross_22.1.4_all.deb; \
    curl -fsSL -o /tmp/rt.deb \
      "https://artifacts.codelinaro.org/artifactory/codelinaro-toolchain-for-hexagon/${TOOLCHAIN_VERSION}/${deb}"; \
    echo "40239e9aa0cba56f60284933dd8143f92008f46014fa034e3bae971fc20bc354  /tmp/rt.deb" | sha256sum -c -; \
    dpkg-deb -x /tmp/rt.deb /tmp/rt && cd /tmp/rt; \
    dest=/opt/hexagon/toolchain/lib/clang/22/lib/hexagon-unknown-linux-musl; \
    mkdir -p "${dest}"; \
    cp usr/hexagon-unknown-linux-musl/usr/lib/libclang_rt.builtins-hexagon.a \
       "${dest}/libclang_rt.builtins.a"; \
    rm -rf /tmp/rt /tmp/rt.deb
ENV PATH=/opt/hexagon/toolchain/bin:$PATH
