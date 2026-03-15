/*
 * ONNX Preprocessing Compute Shader
 * Performs: Resize + RGBA→RGB + HWC→CHW + Normalization
 * 
 * Input:  RGBA texture (any size, packed format)
 * Output: RGB tensor buffer (NCHW planar format, normalized [-1, 1])
 */

// Input texture (from OBS/webcam)
Texture2D<float4> InputTexture : register(t0);
SamplerState BilinearSampler : register(s0);

// Output tensor buffer (NCHW format: [Batch, Channel, Height, Width])
RWBuffer<float> OutputTensor : register(u0);

cbuffer PreprocessParams : register(b0)
{
    uint2 inputSize;      // Source texture dimensions (e.g., 640x360)
    uint2 outputSize;     // Target tensor dimensions (e.g., 224x224)
    float2 normalize;     // [scale, bias] for normalization (e.g., [2/255, -1] for [-1,1])
    uint outputStride;    // Elements per channel (width * height)
};

[numthreads(8, 8, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 outputCoord = dispatchThreadID.xy;
    
    // Bounds check
    if (outputCoord.x >= outputSize.x || outputCoord.y >= outputSize.y)
        return;
    
    // Calculate normalized UV coordinates for bilinear sampling
    float2 uv = (float2(outputCoord) + 0.5f) / float2(outputSize);
    
    // Sample input texture with bilinear filtering (automatic resize)
    float4 rgba = InputTexture.SampleLevel(BilinearSampler, uv, 0);
    
    // Apply normalization: pixel = (value / 255.0) * scale + bias
    // For [-1, 1] range: scale=2.0, bias=-1.0
    // For [0, 1] range:  scale=1.0, bias=0.0
    float3 rgb = rgba.rgb * normalize.x + normalize.y;
    
    // Calculate output buffer offsets for NCHW layout
    // NCHW: [Batch, Channel, Height, Width]
    // Batch=0, so layout is: [R channel][G channel][B channel]
    uint pixelIndex = outputCoord.y * outputSize.x + outputCoord.x;
    
    uint redOffset   = 0 * outputStride + pixelIndex;  // R channel start
    uint greenOffset = 1 * outputStride + pixelIndex;  // G channel start
    uint blueOffset  = 2 * outputStride + pixelIndex;  // B channel start
    
    // Write planar RGB (separate channels)
    OutputTensor[redOffset]   = rgb.r;
    OutputTensor[greenOffset] = rgb.g;
    OutputTensor[blueOffset]  = rgb.b;
}
