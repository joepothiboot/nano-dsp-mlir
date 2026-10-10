#pragma once

#include "constants.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace nanodsp::cuda {

struct Timing {
  double best_ns;
  double median_ns;
  double sample_stddev_ns;
};

inline std::vector<float> inexact(int rows, int cols, int seed) {
  std::vector<float> v(static_cast<std::size_t>(rows) * cols);

  for (std::size_t i = 0; i < v.size(); ++i) {
    std::uint64_t h = ((i + seed) * 2654435761ull) % 1000003ull;
    v[i] = static_cast<float>(h) / 997.0f - 500.0f;
  }

  return v;
}

template <typename Run> double seconds_for(Run &run, long iterations) {
  auto t0 = std::chrono::steady_clock::now();

  for (long i = 0; i < iterations; ++i)
    run();

  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
      .count();
}

template <typename Run> Timing time_samples(Run run) {
  run();
  long iterations = 1;
  double elapsed = seconds_for(run, iterations);

  while (elapsed < kMinSampleSeconds) {
    iterations *= 2;
    elapsed = seconds_for(run, iterations);
  }

  std::vector<double> samples{elapsed / iterations};

  while (static_cast<int>(samples.size()) < kSamples)
    samples.push_back(seconds_for(run, iterations) / iterations);

  double mean = 0.0;

  for (double s : samples)
    mean += s / kSamples;

  double squared = 0.0;

  for (double s : samples)
    squared += (s - mean) * (s - mean);

  std::vector<double> ordered = samples;
  std::ranges::sort(ordered);

  double median = (ordered[kSamples / 2 - 1] + ordered[kSamples / 2]) / 2.0;

  return {ordered.front() * 1e9, median * 1e9,
          std::sqrt(squared / (kSamples - 1)) * 1e9};
}

inline std::string shape_name(int m, int n, int k) {
  return std::to_string(m) + "x" + std::to_string(k) + "x" + std::to_string(n);
}

inline std::string json_row(const std::string &impl, const std::string &device,
                            int m, int n, int k, const Timing &t,
                            const std::string &checked, double bound_used) {
  double ops = 2.0 * m * n * k;
  double bytes = 4.0 * (static_cast<double>(m) * k + static_cast<double>(k) * n + static_cast<double>(m) * n);
  std::string shape = shape_name(m, n, k);
  char buf[1024];

  std::snprintf(buf, sizeof buf,
                "    {\"name\": \"matmul/%s/%s\", \"op\": \"matmul\", "
                "\"shape\": \"%s\", \"impl\": \"%s\", \"config\": \"%s\", "
                "\"real_time\": %.3f, \"median_time\": %.3f, "
                "\"sample_stddev\": %.3f, \"time_unit\": \"ns\", "
                "\"aggregate\": \"min\", \"samples\": %d, \"ops\": %.0f, "
                "\"rate\": %.3f, \"rate_unit\": \"GFLOP/s\", \"bytes\": %.0f, "
                "\"intensity\": %.4f, \"checked\": \"%s\", "
                "\"bound_used\": %.6f}",
                shape.c_str(), impl.c_str(), shape.c_str(), impl.c_str(),
                device.c_str(), t.best_ns, t.median_ns, t.sample_stddev_ns,
                kSamples, ops, ops / t.best_ns, bytes, ops / bytes,
                checked.c_str(), bound_used);

  return buf;
}

}
