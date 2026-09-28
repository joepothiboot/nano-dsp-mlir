# nano-dsp-mlir v0.1.0 🎉

First public release of nano-dsp-mlir.

## 📦 Included

- `dsp` MLIR dialect with add, relu, matmul, and conv2d operations.
- DSP-to-Linalg conversion patterns for tensor-based lowering.
- TableGen-generated dialect and conversion pass definitions.
- `nanodsp-opt` optimizer tool.
- CMake/Ninja setup that finds Homebrew LLVM/MLIR on macOS automatically.
- Lit regression suite covering dialect parsing, conversion, and integration.
- Benchmark harness starter for generated MLIR kernels.

## ✅ Validation

- 15 lit tests passing.
- Tested with Homebrew LLVM/MLIR 23.1.1 on Apple Silicon macOS.

## 🔭 Known scope

This is the foundation release. Frontend parsing, schedule tuning, target
probing, runner tooling, and benchmark sweeps are planned follow-up work.
