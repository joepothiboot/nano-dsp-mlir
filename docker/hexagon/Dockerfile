FROM ubuntu:24.04

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl zstd qemu-user \
 && rm -rf /var/lib/apt/lists/*

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

RUN apt-get update \
 && apt-get install -y --no-install-recommends musl \
 && rm -rf /var/lib/apt/lists/*

RUN ln -s "$(ls -d /opt/hexagon/*-linux-musl)" /opt/hexagon/toolchain

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
