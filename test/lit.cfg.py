import os

import lit.formats
from lit.llvm import llvm_config

config.name = "NANO-DSP-MLIR"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.nanodsp_obj_root, "test")
config.excludes = ["CMakeLists.txt", "lit.cfg.py", "lit.site.cfg.py.in"]

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])
llvm_config.use_default_substitutions()

shlibext = config.llvm_shlib_ext
config.substitutions.append(
    ("%mlir_runner_utils",
     os.path.join(config.mlir_lib_dir, "libmlir_runner_utils" + shlibext)))
config.substitutions.append(
    ("%mlir_c_runner_utils",
     os.path.join(config.mlir_lib_dir, "libmlir_c_runner_utils" + shlibext)))

# -----------------------------------------------------------------------------
# Stage 2 TEST ORACLE ONLY.
#
# This is a stock upstream lowering chain used purely so integration tests can
# *execute* and check numbers. nano-dsp-mlir does not own any of these passes.
# Stages 3-5 replace this entirely with the project's own pipeline; this
# substitution then survives only as a differential-testing reference.
# -----------------------------------------------------------------------------
config.substitutions.append((
    "%stock_lower_to_llvm",
    " ".join([
        '-one-shot-bufferize="bufferize-function-boundaries"',
        "-buffer-deallocation-pipeline",
        "-convert-linalg-to-loops",
        "-convert-scf-to-cf",
        "-expand-strided-metadata",
        "-lower-affine",
        "-convert-arith-to-llvm",
        "-finalize-memref-to-llvm",
        "-convert-func-to-llvm",
        "-convert-cf-to-llvm",
        "-reconcile-unrealized-casts",
    ]),
))

tool_dirs = [config.nanodsp_tools_dir, config.llvm_tools_dir]
tools = ["nanodsp-opt", "mlir-opt", "mlir-runner"]
llvm_config.add_tool_substitutions(tools, tool_dirs)