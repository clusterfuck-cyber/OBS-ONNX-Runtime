/*
 * ONNX Postprocessing Compute Shader
 * Performs: Denormalization + CHW→HWC + RGB→RGBA
 * 
 * Input:  RGB tensor buffer (NCHW planar format, normalized)
 * Output: RGBA texture (packed format, [0-255])
 */

// Input tensor buffer (NCHW format)
Buffer<float> InputTensor : register(t0);

// Output texture (RGBA packed)
RWTexture2D<float4> OutputTexture : register(u0);

cbuffer PostprocessParams : register(b0)
{
    uint outputWidth;     // Final output texture width (e.g. 640)
    uint outputHeight;    // Final output texture height (e.g. 360)
    uint inputWidth;      // Model output width (e.g. 224)
    uint inputHeight;     // Model output height (e.g. 224)
    float denormScale;    // Scale factor for denormalization
    float denormBias;     // Bias for denormalization
    uint inputStride;     // Elements per channel (inputWidth * inputHeight)
    uint padding;
};

[numthreads(8, 8, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 coord = dispatchThreadID.xy;
    
    // Bounds check
    if (coord.x >= outputWidth || coord.y >= outputHeight)
        return;
    
    // Map output coordinates to input coordinates (nearest-neighbor)
    uint srcX = (coord.x * inputWidth) / outputWidth;
    uint srcY = (coord.y * inputHeight) / outputHeight;
    
    // Clamp to valid range
    srcX = clamp(srcX, 0, inputWidth - 1);
    srcY = clamp(srcY, 0, inputHeight - 1);
    
    // Calculate flat index in NCHW buffer
    uint idx = srcY * inputWidth + srcX;
    
    // Read RGB channels from planar NCHW layout
    float r = InputTensor[0 * inputStride + idx];
    float g = InputTensor[1 * inputStride + idx];
    float b = InputTensor[2 * inputStride + idx];
    
    // Denormalize from model output range to [0,1]
    float3 rgb = float3(r, g, b);
    rgb = (rgb - denormBias) / denormScale;
    
    // Write RGBA to output texture
    OutputTexture[coord] = float4(saturate(rgb), 1.0);
}
