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

static void expect_q(const char *name, const QTensor &actual,
                     const std::vector<std::int8_t> &expected) {
  bool ok = actual.data == expected;
  std::printf("%s %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok)
    ++failures;
}

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

  // test/Integration/DSPToLinalg/qmatmul.mlir, case 1: non-zero zero points,
  // saturation at both ends, and both .5 ties (0.5 -> 1, -100.5 -> -100).
  expect_q("qmatmul zero points + rounding",
           qmatmul(QTensor({2, 3}, {-128, 0, 127, 10, -7, 50}),
                   QTensor({3, 4}, {1, -3, 127, -128, 4, 2, 0, 9, -1, 5, -55,
                                    30}),
                   {/*lhs_zp=*/3, /*rhs_zp=*/-2, /*multiplier=*/1073741824,
                    /*shift=*/3, /*out_zp=*/-5}),
           {-23, 57, -128, 127, -4, 13, -105, 27});

  // Case 2: a multiplier that is not a power of two.
  expect_q("qmatmul fractional multiplier",
           qmatmul(QTensor({2, 4}, {127, -128, 64, -1, -50, 33, -2, 90}),
                   QTensor({4, 2}, {3, -7, -2, 11, 100, -100, -9, 4}),
                   {/*lhs_zp=*/0, /*rhs_zp=*/0, /*multiplier=*/1518500250,
                    /*shift=*/5, /*out_zp=*/1}),
           {127, -128, -26, 29});

  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
