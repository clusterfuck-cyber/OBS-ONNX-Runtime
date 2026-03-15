/*
 * AI Engine - ONNX Runtime Wrapper with DirectML
 * Manages ONNX Runtime session and zero-copy inference execution
 * Architecture based on "Advanced Engineering Report: Architecting High-Performance AI Plugins for OBS Studio"
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <wrl/client.h>
#include <d3d11.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <DirectML.h>

class AiEngine {
public:
    AiEngine(const std::wstring& model_path, ID3D11Device* d3d11_device = nullptr);
    ~AiEngine();
    
    // Run inference with D3D12 resources (GPU-only path)
    bool RunInferenceZeroCopy(ID3D12Resource* input_resource, ID3D12Resource* output_resource);
    
    // Get model input/output dimensions
    std::vector<int64_t> GetInputShape() const;
    std::vector<int64_t> GetOutputShape() const;
    
    bool IsInitialized() const { return initialized_; }
    
    // Get D3D12 device for resource sharing
    ID3D12Device* GetD3D12Device() const { return d3d12_device_.Get(); }
    ID3D12CommandQueue* GetCommandQueue() const { return command_queue_.Get(); }
    
    // Device recovery after TDR
    bool RecoverFromDeviceLost();
    
private:
    // Initialize D3D12 device and DirectML
    bool InitializeD3D12Device();
    
    // Initialize ONNX Runtime session with DirectML EP
    bool InitializeSession(const std::wstring& model_path);
    
    // D3D12 device for explicit DirectML control (Section 3.2)
    Microsoft::WRL::ComPtr<ID3D12Device> d3d12_device_;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> command_queue_;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> command_allocator_;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> command_list_;
    Microsoft::WRL::ComPtr<IDMLDevice> dml_device_;
    
    // Dedicated preprocessing command infrastructure (separate from inference)
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> preprocess_command_allocator_;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> preprocess_command_list_;
    UINT64 preprocess_fence_value_;
    
    // Dedicated inference/copy command infrastructure
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> inference_command_allocator_;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> inference_command_list_;
    UINT64 inference_fence_value_;
    
    // Dedicated postprocessing command infrastructure
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> postprocess_command_allocator_;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> postprocess_command_list_;
    UINT64 postprocess_fence_value_;
    
    // Preprocessing compute shader resources
    Microsoft::WRL::ComPtr<ID3D12RootSignature> preprocess_root_signature_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> preprocess_pipeline_;
    Microsoft::WRL::ComPtr<ID3D12Resource> preprocess_tensor_buffer_;  // Intermediate NCHW buffer
    Microsoft::WRL::ComPtr<ID3D12Resource> preprocess_constant_buffer_;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> preprocess_descriptor_heap_;
    
    // Constant buffer data for preprocessing shader
    struct PreprocessParams {
        uint32_t inputWidth;
        uint32_t inputHeight;
        uint32_t outputWidth;
        uint32_t outputHeight;
        float normalizeScale;   // For [-1,1]: 2.0, for [0,1]: 1.0
        float normalizeBias;    // For [-1,1]: -1.0, for [0,1]: 0.0
        uint32_t outputStride;  // width * height (elements per channel)
        uint32_t padding;
    };
    
    // Postprocessing compute shader resources
    Microsoft::WRL::ComPtr<ID3D12RootSignature> postprocess_root_signature_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> postprocess_pipeline_;
    Microsoft::WRL::ComPtr<ID3D12Resource> postprocess_tensor_buffer_;  // Intermediate NCHW buffer from inference
    Microsoft::WRL::ComPtr<ID3D12Resource> postprocess_constant_buffer_;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> postprocess_descriptor_heap_;
    Microsoft::WRL::ComPtr<ID3D12Resource> postprocess_output_texture_;  // RGBA texture for shader output
    
    // Helper functions for preprocessing pipeline
    bool InitializePreprocessPipeline();
    bool InitializePostprocessPipeline();
    
    // Synchronization primitives (Section 3.3)
    Microsoft::WRL::ComPtr<ID3D12Fence> fence_;
    UINT64 fence_value_;
    HANDLE fence_event_;
    
    // Inference synchronization fence
    Microsoft::WRL::ComPtr<ID3D12Fence> inference_fence_;
    HANDLE inference_fence_event_;
    
    // ONNX Runtime components
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    
    // Store actual string data (not just pointers!)
    std::vector<std::string> input_name_strings_;
    std::vector<std::string> output_name_strings_;
    
    // Pointers to the strings above (for ONNX Runtime API)
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;
    
    // Throttled logging state
    uint64_t last_warning_time_;
    
    size_t input_tensor_size_;
    size_t output_tensor_size_;
    
    // Persistent staging buffers for CPU roundtrip (created once, reused every frame)
    Microsoft::WRL::ComPtr<ID3D12Resource> staging_readback_;
    Microsoft::WRL::ComPtr<ID3D12Resource> staging_upload_;
    std::vector<float> cpu_input_buffer_;
    std::vector<float> cpu_output_buffer_;
    
    bool initialized_;
    
    // Cached model path for recovery
    std::wstring model_path_;
    
    // Adapter LUID from OBS D3D11 device (to ensure D3D12 uses same GPU)
    LUID adapter_luid_;
};
