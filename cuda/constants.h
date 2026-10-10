#pragma once

namespace nanodsp::cuda {

inline constexpr int kTile = 16;

inline constexpr int kBlockM = 64;
inline constexpr int kBlockN = 64;
inline constexpr int kBlockK = 16;
inline constexpr int kThreadM = 4;
inline constexpr int kThreadN = 4;

inline constexpr int kVecM = 128;
inline constexpr int kVecN = 128;
inline constexpr int kVecK = 8;
inline constexpr int kVecThreadM = 8;
inline constexpr int kVecThreadN = 8;

inline constexpr int kSamples = 10;
inline constexpr double kMinSampleSeconds = 0.05;

inline constexpr int kBenchSizes[] = {256, 512, 1024, 2048};
inline constexpr int kProfileSize = 1024;

}
