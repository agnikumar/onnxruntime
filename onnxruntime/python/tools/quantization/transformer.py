import onnx
from quantize import quantize, QuantizationMode
import numpy as np

# Load the onnx model
#model = onnx.load('path/to/the/model.onnx')
model = onnx.load('C:/Users/t-agkum/quantization_experimentation/resnet50/model.onnx')

# Quantize
#quantized_model = quantize(model, quantization_mode=QuantizationMode.IntegerOps)

# Trial, changing inputs based on calibration
quantized_model = quantize(model, quantization_mode=QuantizationMode.QLinearOps,
                           static=True,
                           input_quantization_params={
                                'input_1': [np.uint8(113), np.float32(0.05)]
                           },
                           output_quantization_params={
                                'output_1': [np.uint8(113), np.float32(0.05)]
                           })

# Save the quantized model
#onnx.save(quantized_model, 'path/to/the/quantized_model.onnx'
onnx.save(quantized_model, 'C:/Users/t-agkum/quantization_experimentation/resnet50/quantized_model.onnx')
