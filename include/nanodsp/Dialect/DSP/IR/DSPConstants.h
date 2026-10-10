#ifndef NANODSP_DIALECT_DSP_IR_DSPCONSTANTS_H
#define NANODSP_DIALECT_DSP_IR_DSPCONSTANTS_H

#include <cstdint>
#include <limits>

namespace mlir::nanodsp {

inline constexpr int64_t kInt8Min = INT8_MIN;
inline constexpr int64_t kInt8Max = INT8_MAX;
inline constexpr int64_t kUint8Span = 255;

inline constexpr int64_t kQuantShiftBase = 31;
inline constexpr int64_t kMaxQuantShift = 31;
inline constexpr int64_t kMinQuantMultiplier = int64_t{1} << 30;

inline constexpr int64_t kMaxQuantK =
    std::numeric_limits<int32_t>::max() / (kUint8Span * kUint8Span);

}

#endif
