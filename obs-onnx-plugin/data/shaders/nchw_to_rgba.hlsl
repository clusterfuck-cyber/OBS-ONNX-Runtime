/*
 * NCHW to RGBA Postprocessing Compute Shader
 * Converts planar NCHW tensor output back to packed RGBA texture
 * 
 * Input:  NCHW float tensor (1 x 3 x Height x Width, planar layout)
 * Output: RGBA texture (Width x Height, 8-bit per channel, interleaved)
 */

// Input tensor (planar NCHW format)
StructuredBuffer<float> InputTensor : register(t0);

// Output texture (RGBA format)
RWTexture2D<float4> OutputTexture : register(u0);

// Constants
cbuffer PostprocessingConstants : register(b0)
{
    uint Width;           // Image width
    uint Height;          // Image height
    float MeanR;          // Denormalization mean for R channel
    float MeanG;          // Denormalization mean for G channel
    float MeanB;          // Denormalization mean for B channel
    float StdR;           // Denormalization std dev for R channel
    float StdG;           // Denormalization std dev for G channel
    float StdB;           // Denormalization std dev for B channel
};

// Compute shader: Process one pixel per thread
[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint x = dispatchThreadID.x;
    uint y = dispatchThreadID.y;
    
    // Bounds check
    if (x >= Width || y >= Height)
        return;
    
    // Calculate planar indices (NCHW layout)
    uint linear_index = y * Width + x;
    uint plane_size = Height * Width;
    
    // Read from planar input buffer
    float norm_r = InputTensor[0 * plane_size + linear_index];
    float norm_g = InputTensor[1 * plane_size + linear_index];
    float norm_b = InputTensor[2 * plane_size + linear_index];
    
    // Denormalize: value = (norm_value * std) + mean
    float r = (norm_r * StdR) + MeanR;
    float g = (norm_g * StdG) + MeanG;
    float b = (norm_b * StdB) + MeanB;
    
    // Clamp to valid range [0, 255]
    r = clamp(r, 0.0f, 255.0f);
    g = clamp(g, 0.0f, 255.0f);
    b = clamp(b, 0.0f, 255.0f);
    
    // Convert to 0-1 range for texture
    float4 pixel;
    pixel.r = r / 255.0f;
    pixel.g = g / 255.0f;
    pixel.b = b / 255.0f;
    pixel.a = 1.0f;  // Opaque alpha
    
    // Write to RGBA texture
    OutputTexture[uint2(x, y)] = pixel;
}
