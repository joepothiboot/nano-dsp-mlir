// Golden-value checks for the scalar reference: the same inputs and expected
// outputs as test/Integration/ and mojo/tests/, so all three implementations
// are pinned to one set of numbers.
//
// Build and run from the repo root:  pixi run test-reference
#include "nanodsp_ref.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace nanodsp::ref;

static int failures = 0;

static void expect(const char *name, const Tensor &actual,
                   const std::vector<float> &expected) {
  bool ok = actual.data.size() == expected.size();
  for (std::size_t i = 0; ok && i < expected.size(); ++i)
    ok = actual.data[i] == expected[i];
  std::printf("%s %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok)
    ++failures;
}

int main() {
  // test/Integration/end-to-end.mlir
  expect("add", add(Tensor({4}, {1, 2, 3, 4}), Tensor({4}, {1, 1, 1, 1})),
         {2, 3, 4, 5});

  expect("relu", relu(Tensor({6}, {-2, -0.5f, 0, 0.5f, 2, -7})),
         {0, 0, 0, 0.5f, 2, 0});

  Tensor nan_in({1}, std::nanf(""));
  bool nan_ok = std::isnan(relu(nan_in).data[0]);
  std::printf("%s relu propagates NaN\n", nan_ok ? "PASS" : "FAIL");
  failures += !nan_ok;

  // test/Integration/DSPToLinalg/matmul.mlir
  expect("matmul",
         matmul(Tensor({2, 3}, {1, 2, 3, 4, 5, 6}),
                Tensor({3, 2}, {1, 0, 0, 1, 1, 1})),
         {4, 5, 10, 11});

  // test/Integration/DSPToLinalg/conv2d.mlir
  std::vector<float> ramp(16);
  for (int i = 0; i < 16; ++i)
    ramp[i] = float(i + 1);
  expect("conv2d",
         conv2d(Tensor({1, 4, 4, 1}, ramp), Tensor({3, 3, 1, 1}, 1.0f)),
         {54, 63, 90, 99});

  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
