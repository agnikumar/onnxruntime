// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ImageLoader.h"
#include "CachedInterpolation.h"
#include <jpeglib.h>
#include "local_filesystem.h"
#include "jpeg_mem.h"
#include "Callback.h"

void LoadJpegFileIntoMemory(const std::basic_string<ORTCHAR_T>& file_name,int out_height, int out_width,float* output_data,size_t output_data_len){

  int width;
  int height;
  int channels;

  UncompressFlags flags;
  flags.components = 3;
  // The TensorFlow-chosen default for jpeg decoding is IFAST, sacrificing
  // image quality for speed.
  flags.dct_method = JDCT_IFAST;
  size_t file_len;
  void* file_data;
  Callback c;
  ReadFileAsString(file_name.c_str(),file_data,file_len, c);
  file_data = Uncompress(file_data,file_len,flags,&width,&height,&channels,nullptr);
  if(c.f)
    c.f(c.param);

  if (channels != 3) {
    std::ostringstream oss;
    oss << "input format error, expect 3 channels, got " << channels;
    throw std::runtime_error(oss.str());
  }

  //cast uint8 to float
  std::vector<float> float_file_data(height * width*channels);
  for(size_t i=0;i!=float_file_data.size();++i){
    float_file_data[i] = static_cast<float>(((uint8_t*) file_data)[i])/255;
  }
  delete[] file_data;

  int in_height = height;
  int in_width = width;

  float height_scale = CalculateResizeScale(in_height, out_height, false);
  float width_scale = CalculateResizeScale(in_width, out_width, false);


  std::vector<CachedInterpolation> ys(out_height + 1);
  std::vector<CachedInterpolation> xs(out_width + 1);

  // Compute the cached interpolation weights on the x and y dimensions.
  compute_interpolation_weights(out_height, in_height, height_scale,
                                ys.data());
  compute_interpolation_weights(out_width, in_width, width_scale, xs.data());


  // Scale x interpolation weights to avoid a multiplication during iteration.
  for (int i = 0; i < xs.size(); ++i) {
    xs[i].lower *= channels;
    xs[i].upper *= channels;
  }


  resize_image(float_file_data.data(),1,in_height,in_width,out_height,out_width,channels,xs,ys,output_data);
  for(size_t i =0;i!=output_data_len;++i){
    output_data[i] = (output_data[i] - 0.5) * 2;
  }
}