/*
 * RGBA to NCHW Preprocessing Compute Shader
 * Converts packed RGBA texture to planar NCHW tensor format with normalization
 * 
 * Based on Section 5.2 "The HLSL Compute Shader Solution"
 * 
 * Input:  RGBA texture (Width x Height, 8-bit per channel, interleaved)
 * Output: NCHW float tensor (1 x 3 x Height x Width, planar layout)
 * 
 * Memory Layout Transformation:
 *   RGBA: R1 G1 B1 A1 R2 G2 B2 A2 R3 G3 B3 A3 ...
 *   NCHW: R1 R2 R3 ... | G1 G2 G3 ... | B1 B2 B3 ...
 */

// Input texture (from D3D11 shared resource)
Texture2D<float4> InputTexture : register(t0);

// Output buffer (planar NCHW format)
RWStructuredBuffer<float> OutputTensor : register(u0);

// Constants
cbuffer PreprocessingConstants : register(b0)
{
    uint Width;           // Image width
    uint Height;          // Image height
    float MeanR;          // Normalization mean for R channel
    float MeanG;          // Normalization mean for G channel
    float MeanB;          // Normalization mean for B channel
    float StdR;           // Normalization std dev for R channel
    float StdG;           // Normalization std dev for G channel
    float StdB;           // Normalization std dev for B channel
};

// Compute shader: Process one pixel per thread
// Thread group size: 8x8 (optimal for most GPUs)
[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint x = dispatchThreadID.x;
    uint y = dispatchThreadID.y;
    
    // Bounds check
    if (x >= Width || y >= Height)
        return;
    
    // Read pixel from RGBA texture (normalized 0.0-1.0)
    float4 pixel = InputTexture[uint2(x, y)];
    
    // Convert from 0-1 range to 0-255 range
    float r = pixel.r * 255.0f;
    float g = pixel.g * 255.0f;
    float b = pixel.b * 255.0f;
    
    // Apply normalization: (value - mean) / std
    float norm_r = (r - MeanR) / StdR;
    float norm_g = (g - MeanG) / StdG;
    float norm_b = (b - MeanB) / StdB;
    
    // Calculate planar indices (NCHW layout)
    // Tensor shape: [1, 3, Height, Width]
    // Strides: [3*H*W, H*W, W, 1]
    
    uint linear_index = y * Width + x;  // Position in H*W plane
    uint plane_size = Height * Width;   // Size of one channel plane
    
    // Write to planar output buffer
    // Channel 0 (R): [0 to H*W-1]
    OutputTensor[0 * plane_size + linear_index] = norm_r;
    
    // Channel 1 (G): [H*W to 2*H*W-1]
    OutputTensor[1 * plane_size + linear_index] = norm_g;
    
    // Channel 2 (B): [2*H*W to 3*H*W-1]
    OutputTensor[2 * plane_size + linear_index] = norm_b;
}
