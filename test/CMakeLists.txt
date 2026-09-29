# The path lit sees for test sources. test.sh overrides it with a
# whitespace-free symlink, because lit does not quote %s in RUN lines and CMake
# resolves symlinks in CMAKE_CURRENT_SOURCE_DIR.
set(NANODSP_LIT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH
  "Test source directory as seen by lit")

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(NANODSP_TEST_DEPENDS
  FileCheck count not
  nanodsp-opt
  mlir-opt
  mlir-runner
  mlir_runner_utils
  mlir_c_runner_utils
)

add_lit_testsuite(check-nanodsp "Running nano-dsp-mlir regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${NANODSP_TEST_DEPENDS}
)
set_target_properties(check-nanodsp PROPERTIES FOLDER "Tests")
add_lit_testsuites(NANODSP ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${NANODSP_TEST_DEPENDS})