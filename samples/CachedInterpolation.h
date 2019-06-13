
#pragma once

#include <stdint.h>
// Compute the interpolation indices only once.
struct CachedInterpolation {
  int64_t lower;  // Lower source index used in the interpolation
  int64_t upper;  // Upper source index used in the interpolation
  // 1-D linear iterpolation scale (see:
  // https://en.wikipedia.org/wiki/Bilinear_interpolation)
  float lerp;
};

