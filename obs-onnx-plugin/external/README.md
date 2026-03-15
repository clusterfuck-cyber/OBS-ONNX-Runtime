# External Dependencies

This directory should contain symbolic links or references to external dependencies:

## Required Dependencies

### 1. OBS Studio Source
- **Location**: Should point to `../obs-studio/`
- **Purpose**: Provides libobs headers and libraries
- **How to link**: The CMakeLists.txt references this automatically

### 2. ONNX Runtime
- **Location**: Should point to `../onnxruntime-win-x64-1.23.2/`
- **Purpose**: Provides ONNX Runtime inference engine with CUDA support
- **How to link**: The CMakeLists.txt references this automatically

## Current Setup

Based on your workspace structure:
- OBS Studio: `E:\_DEV\OBSPlugins\obs-studio\`
- ONNX Runtime: `E:\_DEV\OBSPlugins\onnxruntime-win-x64-1.23.2\`

The CMakeLists.txt is configured to find these automatically using relative paths.

## Manual Configuration

If the automatic detection fails, you can override the paths when configuring CMake:

```powershell
cmake -B build -S . `
  -DOBS_STUDIO_ROOT="E:\_DEV\OBSPlugins\obs-studio" `
  -DONNXRUNTIME_ROOT="E:\_DEV\OBSPlugins\onnxruntime-win-x64-1.23.2"
```
