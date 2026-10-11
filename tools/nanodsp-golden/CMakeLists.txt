add_llvm_executable(nanodsp-golden nanodsp-golden.cpp)
set_target_properties(nanodsp-golden PROPERTIES CXX_STANDARD 20)
target_compile_options(nanodsp-golden PRIVATE -fexceptions -ffp-contract=off)
