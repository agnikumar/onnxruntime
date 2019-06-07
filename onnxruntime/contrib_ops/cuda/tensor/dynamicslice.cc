// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/slice.h"
//#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/cuda/tensor/slice_impl.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_TYPED_DYNAMICSLICE(TIND)                                                                                                                                                                                                                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                                                                                                                                                                                                        \
      DynamicSlice,                                                                                                                                                                                                                                                     \
      kOnnxDomain,                                                                                                                                                                                                                                                      \
      1,                                                                                                                                                                                                                                                                \
      TIND,                                                                                                                                                                                                                                                             \
      kCudaExecutionProvider,                                                                                                                                                                                                                                           \
      KernelDefBuilder().InputMemoryType<OrtMemTypeCPUInput>(1).InputMemoryType<OrtMemTypeCPUInput>(2).InputMemoryType<OrtMemTypeCPUInput>(3).TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()).TypeConstraint("Tind", DataTypeImpl::GetTensorType<TIND>()), \
      Slice<TIND, true>);

REGISTER_TYPED_DYNAMICSLICE(int32_t)
REGISTER_TYPED_DYNAMICSLICE(int64_t)

}  // namespace cuda
}  // namespace onnxruntime
