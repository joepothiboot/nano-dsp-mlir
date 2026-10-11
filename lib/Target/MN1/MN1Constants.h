#ifndef NANODSP_LIB_TARGET_MN1_MN1CONSTANTS_H
#define NANODSP_LIB_TARGET_MN1_MN1CONSTANTS_H

#include <cstdint>

namespace mlir::nanodsp::mn1 {

inline constexpr int kLanes = 8;
inline constexpr int64_t kElementBytes = 4;
inline constexpr int64_t kVectorBytes = kLanes * kElementBytes;

inline constexpr int kFirstScalarReg = 1;
inline constexpr int kLastScalarReg = 12;
inline constexpr int kScratch0 = 13;
inline constexpr int kScratch1 = 14;
inline constexpr int kScratch2 = 15;
inline constexpr int kNumVectorRegs = 16;

inline constexpr int kNumTags = 8;
inline constexpr int kInternalTag = kNumTags - 1;

inline constexpr uint64_t kSpmBytes = 64 * 1024;
inline constexpr uint64_t kDramBytes = 1024 * 1024;
inline constexpr uint64_t kDramSlotBytes = 0x40000;
inline constexpr int64_t kDmaGranule = 16;
inline constexpr int64_t kZeroBlockBytes = 512;

}

#endif
