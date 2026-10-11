#include "../../reference/nanodsp_ref.h"

#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

namespace {

using nanodsp::ref::Tensor;

constexpr std::uint64_t kHashMultiplier = 2654435761ULL;
constexpr std::uint64_t kHashModulus = 1000003ULL;
constexpr float kScale = 997.0F;
constexpr float kShift = 500.0F;

Tensor inexact(std::vector<std::size_t> shape, int seed) {
  Tensor t(std::move(shape));

  for (std::size_t i = 0; i < t.data.size(); ++i) {
    std::uint64_t h = ((i + seed) * kHashMultiplier) % kHashModulus;
    t.data[i] = (static_cast<float>(h) / kScale) - kShift;
  }

  return t;
}

void write(const std::string &path, const Tensor &t) {
  std::ofstream f(path, std::ios::binary);

  if (!f)
    throw std::runtime_error("cannot write " + path);

  f.write(reinterpret_cast<const char *>(t.data.data()),
          static_cast<std::streamsize>(t.data.size() * sizeof(float)));
}

std::size_t parse(const char *text) {
  std::size_t used = 0;
  unsigned long value = std::stoul(text, &used);

  if (text[used] != '\0' || value == 0)
    throw std::runtime_error(std::string("bad dimension '") + text + "'");

  return value;
}

int usage() {
  std::fprintf(stderr, "usage: nanodsp-golden matmul M K N OUT_DIR\n"
                       "       nanodsp-golden add ROWS COLS OUT_DIR\n");
  return 1;
}

}

int main(int argc, char **argv) {
  try {
    std::vector<std::string> args(argv + 1, argv + argc);

    if (args.size() == 5 && args[0] == "matmul") {
      std::size_t m = parse(argv[2]);
      std::size_t k = parse(argv[3]);
      std::size_t n = parse(argv[4]);
      Tensor a = inexact({m, k}, 1);
      Tensor b = inexact({k, n}, 2);
      write(args[4] + "/a.f32", a);
      write(args[4] + "/b.f32", b);
      write(args[4] + "/c_ref.f32", nanodsp::ref::matmul(a, b));
      return 0;
    }

    if (args.size() == 4 && args[0] == "add") {
      std::size_t rows = parse(argv[2]);
      std::size_t cols = parse(argv[3]);
      Tensor a = inexact({rows, cols}, 1);
      Tensor b = inexact({rows, cols}, 2);
      write(args[3] + "/a.f32", a);
      write(args[3] + "/b.f32", b);
      write(args[3] + "/c_ref.f32", nanodsp::ref::add(a, b));
      return 0;
    }

    return usage();
  } catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
