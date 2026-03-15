# ONNX Models Directory

Place your ONNX model files (.onnx) in this directory.

## Recommended Models

### Background Segmentation
- **U²-Net**: Salient object detection
- **MODNet**: Mobile real-time portrait matting
- **DeepLabV3**: Semantic segmentation

### Object Detection
- **YOLOv8**: Real-time object detection
- **YOLOX**: High-performance detector

### Style Transfer
- **Fast Neural Style**: Real-time artistic style transfer

## Model Requirements

- **Format**: ONNX (.onnx file)
- **Input Shape**: Typically [1, 3, H, W] for RGB images (NCHW format)
- **Precision**: FP32 or FP16 (FP16 recommended for RTX GPUs)
- **Memory**: Consider your GPU VRAM (8GB recommended minimum)

## Converting Models to ONNX

If you have a PyTorch or TensorFlow model, you can convert it to ONNX:

### PyTorch Example
```python
import torch
import torch.onnx

model = YourModel()
model.eval()

dummy_input = torch.randn(1, 3, 512, 512)
torch.onnx.export(model, dummy_input, "model.onnx",
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'},
                               'output': {0: 'batch_size'}})
```

## Testing Models

1. Place the .onnx file in this directory
2. Launch OBS Studio
3. Add the ONNX AI Filter to a video source
4. Browse to select your model file
5. Monitor OBS logs for any errors

## Performance Tips

- Use smaller input resolutions (512x512 or 640x640) for better FPS
- FP16 models run ~2x faster on RTX GPUs
- Test with OBS performance monitor (Stats dock)
