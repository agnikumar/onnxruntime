// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/tensor/slice.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace cuda {

template<typename Tind, bool dynamic>
class Slice final : public CudaKernel, public SliceBase {
 public:
  Slice(const OpKernelInfo& info) : CudaKernel(info), SliceBase(info, dynamic) {}

  Status ComputeInternal(OpKernelContext* ctx) const override {
    auto input_tensor = ctx->Input<Tensor>(0);
    ORT_ENFORCE(nullptr != input_tensor);
    auto& input_dimensions = input_tensor->Shape().GetDims();

    // Initialize the starts & ends to the actual tensor shape
    const size_t dimension_count = input_dimensions.size();
    std::vector<int64_t> starts(dimension_count, 0);
    std::vector<int64_t> steps(input_dimensions.size(), 1);
    std::vector<int64_t> output_dims(input_dimensions);

    if (dynamic) {
      std::vector<int64_t> input_starts, input_ends, input_axes, input_steps;
      FillVectorsFromInput(ctx, input_starts, input_ends, input_axes, input_steps);
      ORT_RETURN_IF_ERROR(PrepareForCompute(input_starts, input_ends, input_axes,
                                            input_steps, input_dimensions, starts, steps, output_dims));

    } else {
      ORT_RETURN_IF_ERROR(PrepareForCompute(attr_starts_, attr_ends_, attr_axes_,
                                            input_dimensions, starts, output_dims));
    }

    TensorShape output_shape(output_dims);
    auto output_tensor = ctx->Output(0, output_shape);
    int64_t output_size = output_shape.Size();
    if (output_size == 0) {
      return Status::OK();
    }
    int device_id = 0;
    CudaAsyncBuffer<int64_t> starts_buffer(this, device_id, dimension_count);
    gsl::span<int64_t> starts_buffer_span = starts_buffer.CpuSpan();
    for (int i = 0; i < dimension_count; ++i) {
      starts_buffer_span[i] = starts[i];
    }
    starts_buffer.CopyToGpu();

    CudaAsyncBuffer<int64_t> input_strides(this, device_id, dimension_count);
    ORT_ENFORCE(TensorPitches::Calculate(input_strides.CpuSpan(), input_dimensions));
    input_strides.CopyToGpu();

    TensorPitches output_pitches(output_dims);

    CudaAsyncBuffer<fast_divmod> div_strides(this, device_id, dimension_count);
    gsl::span<fast_divmod> div_strides_span = div_strides.CpuSpan();
    for (int i = 0; i < dimension_count; ++i) {
      div_strides_span[i] = fast_divmod(gsl::narrow_cast<int>(output_pitches[i]));
    }
    div_strides.CopyToGpu();

    size_t element_size = input_tensor->DataType()->Size();

    ORT_RETURN_IF_ERROR(SliceImpl(element_size,
                                  gsl::narrow_cast<int32_t>(dimension_count),
                                  starts_buffer.GpuPtr(),
                                  input_strides.GpuPtr(),
                                  div_strides.GpuPtr(),
                                  input_tensor->DataRaw(),
                                  output_tensor->MutableDataRaw(),
                                  output_size));

    return Status::OK();
  }
};

}  // namespace cuda
}  // namespace onnxruntime
