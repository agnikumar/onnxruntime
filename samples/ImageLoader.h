// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma  once
#include <stdint.h>
#include <vector>
#include <string>
#include "CachedInterpolation.h"
#include <onnxruntime/core/session/onnxruntime_c_api.h>

inline void compute_interpolation_weights(const int64_t out_size,
                                          const int64_t in_size,
                                          const float scale,
                                          CachedInterpolation* interpolation) {
  interpolation[out_size].lower = 0;
  interpolation[out_size].upper = 0;
  for (int64_t i = out_size - 1; i >= 0; --i) {
    const float in = i * scale;
    interpolation[i].lower = static_cast<int64_t>(in);
    interpolation[i].upper = std::min(interpolation[i].lower + 1, in_size - 1);
    interpolation[i].lerp = in - interpolation[i].lower;
  }
}

/**
 * Computes the bilinear interpolation from the appropriate 4 float points
 * and the linear interpolation weights.
 */
inline float compute_lerp(const float top_left, const float top_right,
                          const float bottom_left, const float bottom_right,
                          const float x_lerp, const float y_lerp) {
  const float top = top_left + (top_right - top_left) * x_lerp;
  const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
  return top + (bottom - top) * y_lerp;
}


void resize_image(const float* input_b_ptr,
                  const int batch_size, const int64_t in_height,
                  const int64_t in_width, const int64_t out_height,
                  const int64_t out_width, const int channels,
                  const std::vector<CachedInterpolation>& xs_vec,
                  const std::vector<CachedInterpolation>& ys,
                  float* output_y_ptr) {
  const int64_t in_row_size = in_width * channels;
  const int64_t in_batch_num_values = in_height * in_row_size;
  const int64_t out_row_size = out_width * channels;

  const CachedInterpolation* xs = xs_vec.data();

  if (channels == 3) {
    for (int b = 0; b < batch_size; ++b) {
      for (int64_t y = 0; y < out_height; ++y) {
        const float* ys_input_lower_ptr = input_b_ptr + ys[y].lower * in_row_size;
        const float* ys_input_upper_ptr = input_b_ptr + ys[y].upper * in_row_size;
        const float ys_lerp = ys[y].lerp;
        for (int64_t x = 0; x < out_width; ++x) {
          const int64_t xs_lower = xs[x].lower;
          const int64_t xs_upper = xs[x].upper;
          const float xs_lerp = xs[x].lerp;

          // Read channel 0.
          const float top_left0(ys_input_lower_ptr[xs_lower + 0]);
          const float top_right0(ys_input_lower_ptr[xs_upper + 0]);
          const float bottom_left0(ys_input_upper_ptr[xs_lower + 0]);
          const float bottom_right0(ys_input_upper_ptr[xs_upper + 0]);

          // Read channel 1.
          const float top_left1(ys_input_lower_ptr[xs_lower + 1]);
          const float top_right1(ys_input_lower_ptr[xs_upper + 1]);
          const float bottom_left1(ys_input_upper_ptr[xs_lower + 1]);
          const float bottom_right1(ys_input_upper_ptr[xs_upper + 1]);

          // Read channel 2.
          const float top_left2(ys_input_lower_ptr[xs_lower + 2]);
          const float top_right2(ys_input_lower_ptr[xs_upper + 2]);
          const float bottom_left2(ys_input_upper_ptr[xs_lower + 2]);
          const float bottom_right2(ys_input_upper_ptr[xs_upper + 2]);

          // Compute output.
          output_y_ptr[x * channels + 0] =
              compute_lerp(top_left0, top_right0, bottom_left0, bottom_right0,
                           xs_lerp, ys_lerp);
          output_y_ptr[x * channels + 1] =
              compute_lerp(top_left1, top_right1, bottom_left1, bottom_right1,
                           xs_lerp, ys_lerp);
          output_y_ptr[x * channels + 2] =
              compute_lerp(top_left2, top_right2, bottom_left2, bottom_right2,
                           xs_lerp, ys_lerp);
        }
        output_y_ptr += out_row_size;
      }
      input_b_ptr += in_batch_num_values;
    }
  } else {
    for (int b = 0; b < batch_size; ++b) {
      for (int64_t y = 0; y < out_height; ++y) {
        const float* ys_input_lower_ptr = input_b_ptr + ys[y].lower * in_row_size;
        const float* ys_input_upper_ptr = input_b_ptr + ys[y].upper * in_row_size;
        const float ys_lerp = ys[y].lerp;
        for (int64_t x = 0; x < out_width; ++x) {
          auto xs_lower = xs[x].lower;
          auto xs_upper = xs[x].upper;
          auto xs_lerp = xs[x].lerp;
          for (int c = 0; c < channels; ++c) {
            const float top_left(ys_input_lower_ptr[xs_lower + c]);
            const float top_right(ys_input_lower_ptr[xs_upper + c]);
            const float bottom_left(ys_input_upper_ptr[xs_lower + c]);
            const float bottom_right(ys_input_upper_ptr[xs_upper + c]);
            output_y_ptr[x * channels + c] =
                compute_lerp(top_left, top_right, bottom_left, bottom_right,
                             xs_lerp, ys_lerp);
          }
        }
        output_y_ptr += out_row_size;
      }
      input_b_ptr += in_batch_num_values;
    }
  }
}


/**
 * CalculateResizeScale determines the float scaling factor.
 * @param in_size
 * @param out_size
 * @param align_corners If true, the centers of the 4 corner pixels of the input and output tensors are aligned,
 *                        preserving the values at the corner pixels
 * @return
 */
inline float CalculateResizeScale(int64_t in_size, int64_t out_size,
                                  bool align_corners) {
  return (align_corners && out_size > 1)
         ? (in_size - 1) / static_cast<float>(out_size - 1)
         : in_size / static_cast<float>(out_size);
}

void LoadJpegFileIntoMemory(const std::basic_string<ORTCHAR_T>& file_name,int out_height, int out_width,float* output_data,size_t output_data_len);

