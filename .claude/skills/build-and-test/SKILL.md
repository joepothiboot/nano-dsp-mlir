---
name: build-and-test
description: Build nanodsp-opt and run the lit test suite. Use after changing any .td, .cpp, CMake or test file.
---

1. Run `./test.sh` from the repo root. It configures with Ninja into `build/`
   and runs the `check-nanodsp` target.
2. If MLIR isn't found, set `MLIR_DIR` to the directory containing
   `MLIRConfig.cmake` (Homebrew: `/opt/homebrew/opt/llvm/lib/cmake/mlir`).
3. To run a single test: `lit -v build/test/<path/to/test.mlir>`.
4. For a failing FileCheck test, run the `RUN:` line by hand with
   `build/bin/nanodsp-opt` and diff against the `CHECK` lines.
