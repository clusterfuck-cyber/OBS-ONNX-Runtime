/*
 * GPU Interop - CUDA Kernels for Format Conversion
 * Handles conversion between D3D11 textures and ONNX tensor formats
 */

#include <cuda_runtime.h>

// Pre-processing kernel: BGRA packed -> RGB planar (NCHW) normalized
__global__ void PreprocessKernel(
    cudaSurfaceObject_t input_surface,
    float* output_tensor,
    int width, int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Read packed BGRA pixel (uchar4)
        uchar4 pixel;
        surf2Dread(&pixel, input_surface, x * sizeof(uchar4), y);
        
        // Calculate planar offsets (NCHW format)
        int pixel_index = y * width + x;
        int channel_stride = width * height;
        
        // Normalize (0-255 -> 0.0-1.0) and convert BGRA to RGB planar
        // pixel.x = B, pixel.y = G, pixel.z = R, pixel.w = A
        output_tensor[pixel_index] = (float)pixel.z / 255.0f;                    // R channel
        output_tensor[pixel_index + channel_stride] = (float)pixel.y / 255.0f;   // G channel
        output_tensor[pixel_index + channel_stride * 2] = (float)pixel.x / 255.0f; // B channel
    }
}

// Post-processing kernel: RGB planar (NCHW) -> BGRA packed with denormalization
__global__ void PostprocessKernel(
    float* input_tensor,
    cudaSurfaceObject_t output_surface,
    int width, int height,
    float min_val, float max_val)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Calculate planar offset (NCHW format)
        int pixel_index = y * width + x;
        int channel_stride = width * height;
        
        // Read RGB values from planar tensor
        float r = input_tensor[pixel_index];                      // R channel
        float g = input_tensor[pixel_index + channel_stride];     // G channel
        float b = input_tensor[pixel_index + channel_stride * 2]; // B channel
        
        // Denormalize from [min_val, max_val] to [0, 255]
        // Common ranges: [0,1], [-1,1], or [0,255]
        float range = max_val - min_val;
        r = ((r - min_val) / range) * 255.0f;
        g = ((g - min_val) / range) * 255.0f;
        b = ((b - min_val) / range) * 255.0f;
        
        // Clamp to valid range
        r = fminf(fmaxf(r, 0.0f), 255.0f);
        g = fminf(fmaxf(g, 0.0f), 255.0f);
        b = fminf(fmaxf(b, 0.0f), 255.0f);
        
        // Pack as BGRA (note order: B, G, R, A)
        uchar4 pixel;
        pixel.x = (unsigned char)b; // B
        pixel.y = (unsigned char)g; // G
        pixel.z = (unsigned char)r; // R
        pixel.w = 255;              // A (fully opaque)
        
        // Write to output surface
        surf2Dwrite(pixel, output_surface, x * sizeof(uchar4), y);
    }
}

// C interface for launching kernels
extern "C" {
    
void LaunchPreprocessKernel(
    cudaSurfaceObject_t input_surface,
    float* output_tensor,
    int width, int height,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    PreprocessKernel<<<grid, block, 0, stream>>>(input_surface, output_tensor, width, height);
}

void LaunchPostprocessKernel(
    float* input_tensor,
    cudaSurfaceObject_t output_surface,
    int width, int height,
    float min_val, float max_val,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    PostprocessKernel<<<grid, block, 0, stream>>>(
        input_tensor, output_surface, width, height, min_val, max_val);
}

} // extern "C"
