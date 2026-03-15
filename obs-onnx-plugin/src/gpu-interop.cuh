/*
 * GPU Interop - CUDA Kernel Headers
 * C interface for launching CUDA kernels from C++ code
 */

#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Launch pre-processing kernel (BGRA -> RGB planar normalized)
void LaunchPreprocessKernel(
    cudaSurfaceObject_t input_surface,
    float* output_tensor,
    int width, int height,
    cudaStream_t stream);

// Launch post-processing kernel (RGB planar -> BGRA packed)
void LaunchPostprocessKernel(
    float* input_tensor,
    cudaSurfaceObject_t output_surface,
    int width, int height,
    float min_val, float max_val,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
