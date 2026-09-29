# Contributing

## Build and test

```bash
./test.sh                # configure, build and run the MLIR lit suite
pixi run test-reference  # C++ reference golden tests
pixi run test-mojo       # Mojo kernel tests
npx prettier --check .   # formatting for non-C++ files
```

`test.sh` finds Homebrew LLVM automatically. Anywhere else, set `MLIR_DIR` to
the directory that contains `MLIRConfig.cmake`. LLVM/MLIR 21 or newer is
required. CI runs all four commands on every pull request.

## Adding an op

Every op has the same semantics in three implementations, and the tests keep
them in agreement. A new op needs all of the following:

1. ODS definition and verifier in `include/nanodsp/Dialect/DSP/IR/DSPOps.td`
2. A lowering to `linalg.generic` (destination-passing style) in
   `lib/Conversion/DSPToLinalg/`
3. Tests: `test/Dialect/` (parse, print, verify), `test/Conversion/` (IR
   structure) and `test/Integration/` (executed, checks output values)
4. A Mojo kernel in `mojo/nanodsp/` with golden and differential tests
5. A scalar C++ version in `reference/nanodsp_ref.h`

Steps 3 to 5 share one set of golden values. Keep multiply and add unfused, and
keep the naive loop nest's reduction order, so results compare bit-exactly.

## Pull requests

- One logical change per PR, with the tests for it in the same PR.
- Builds stay free of warnings, including deprecation warnings.
- If a change moves the project forward on the roadmap, update the status table
  in `README.md` in the same PR.
