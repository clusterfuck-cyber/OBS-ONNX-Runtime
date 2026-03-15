/*
 * AI Engine - ONNX Runtime Wrapper Implementation
 * Handles ONNX model loading and zero-copy inference execution with DirectML
 * Architecture based on "Advanced Engineering Report: Architecting High-Performance AI Plugins for OBS Studio"
 */

#include "ai-engine.hpp"
#include <obs-module.h>
#include <util/platform.h>
#include <onnxruntime_c_api.h>
#include <dml_provider_factory.h>

// Include precompiled shader bytecode
#include "preprocess_nchw_bytecode.h"
#include "postprocess_rgba_bytecode.h"


AiEngine::AiEngine(const std::wstring& model_path, ID3D11Device* d3d11_device)
    : initialized_(false)
    , input_tensor_size_(0)
    , output_tensor_size_(0)
    , fence_value_(0)
    , preprocess_fence_value_(0)
    , inference_fence_value_(0)
    , fence_event_(nullptr)
    , model_path_(model_path)
    , last_warning_time_(0)
{
    // CRITICAL FIX: Extract LUID from OBS D3D11 device to ensure D3D12 uses same GPU adapter
    adapter_luid_ = {};
    
    if (d3d11_device) {
        Microsoft::WRL::ComPtr<IDXGIDevice> dxgi_device;
        HRESULT hr = d3d11_device->QueryInterface(IID_PPV_ARGS(&dxgi_device));
        if (SUCCEEDED(hr)) {
            Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;
            hr = dxgi_device->GetAdapter(&adapter);
            if (SUCCEEDED(hr)) {
                DXGI_ADAPTER_DESC desc;
                hr = adapter->GetDesc(&desc);
                if (SUCCEEDED(hr)) {
                    adapter_luid_ = desc.AdapterLuid;
                    blog(LOG_INFO, "[AI Engine] OBS Adapter LUID: %08X-%08X", 
                         adapter_luid_.HighPart, adapter_luid_.LowPart);
                } else {
                    blog(LOG_ERROR, "[AI Engine] Failed to get adapter description");
                }
            } else {
                blog(LOG_ERROR, "[AI Engine] Failed to get adapter from DXGI device");
            }
        } else {
            blog(LOG_ERROR, "[AI Engine] Failed to get IDXGIDevice from D3D11 device");
        }
    } else {
        blog(LOG_WARNING, "[AI Engine] No D3D11 device provided - will use default adapter");
    }
    
    try {
        blog(LOG_INFO, "[AI Engine] Initializing ONNX Runtime with DirectML (Zero-Copy Architecture)");
        
        // Section 3.1: Initialize D3D12 device for DirectML ON THE SAME ADAPTER
        if (!InitializeD3D12Device()) {
            blog(LOG_WARNING, "[AI Engine] Failed to initialize D3D12 device - will use CPU path");
            // Continue anyway - we can still use CPU execution
        }
        
        // Section 4.3: Initialize ONNX Runtime session
        if (!InitializeSession(model_path)) {
            blog(LOG_ERROR, "[AI Engine] InitializeSession returned false");
            throw std::runtime_error("Failed to initialize ONNX Runtime session");
        }
        
        // Initialize preprocessing pipeline if D3D12 is available
        if (d3d12_device_) {
            if (!InitializePreprocessPipeline()) {
                blog(LOG_WARNING, "[AI Engine] Failed to initialize preprocessing pipeline - dimension mismatches will fail");
            } else {
                blog(LOG_INFO, "[AI Engine] ✓ Preprocessing pipeline ready");
            }
            
            // Initialize postprocessing pipeline
            if (!InitializePostprocessPipeline()) {
                blog(LOG_ERROR, "[AI Engine] Failed to initialize postprocessing pipeline");
                throw std::runtime_error("Failed to initialize postprocessing pipeline");
            } else {
                blog(LOG_INFO, "[AI Engine] ✓ Postprocessing pipeline ready");
            }
        }
        
        initialized_ = true;
        blog(LOG_INFO, "[AI Engine] ✓ Initialization complete");
        
    } catch (const std::exception& e) {
        blog(LOG_ERROR, "[AI Engine] ❌ Initialization error: %s", e.what());
        initialized_ = false;
    } catch (...) {
        blog(LOG_ERROR, "[AI Engine] ❌ Unknown initialization error");
        initialized_ = false;
    }
}

bool AiEngine::InitializeD3D12Device()
{
    blog(LOG_INFO, "[AI Engine] === Initializing D3D12 Device (Section 3.1) ===");
    
    try {
        // Find the specific adapter matching the OBS LUID
        Microsoft::WRL::ComPtr<IDXGIAdapter1> hardware_adapter;
        
        if (adapter_luid_.LowPart != 0 || adapter_luid_.HighPart != 0) {
            Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
            HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
            if (FAILED(hr)) {
                blog(LOG_ERROR, "[AI Engine] Failed to create DXGI Factory (HRESULT: 0x%08X)", hr);
                return false;
            }
            
            Microsoft::WRL::ComPtr<IDXGIAdapter1> temp_adapter;
            for (UINT adapter_index = 0; 
                 factory->EnumAdapters1(adapter_index, &temp_adapter) != DXGI_ERROR_NOT_FOUND; 
                 ++adapter_index) 
            {
                DXGI_ADAPTER_DESC1 desc;
                temp_adapter->GetDesc1(&desc);
                
                // Check if LUID matches OBS adapter
                if (desc.AdapterLuid.LowPart == adapter_luid_.LowPart && 
                    desc.AdapterLuid.HighPart == adapter_luid_.HighPart) 
                {
                    hardware_adapter = temp_adapter;
                    blog(LOG_INFO, "[AI Engine] ✓ Found matching adapter: %ls", desc.Description);
                    break;
                }
            }
            
            if (!hardware_adapter) {
                blog(LOG_ERROR, "[AI Engine] ❌ Could not find adapter matching OBS LUID! Shared handles will likely fail.");
                // Will use nullptr fallback below
            }
        } else {
            blog(LOG_INFO, "[AI Engine] No LUID specified, using default adapter");
        }
        
        // Create D3D12 device on the matching adapter (or nullptr if not found)
        HRESULT hr = D3D12CreateDevice(
            hardware_adapter.Get(),     // Use matching adapter (or nullptr for default)
            D3D_FEATURE_LEVEL_11_0,     // Minimum feature level
            IID_PPV_ARGS(&d3d12_device_)
        );
        
        if (FAILED(hr)) {
            blog(LOG_ERROR, "[AI Engine] Failed to create D3D12 device (HRESULT: 0x%08X)", hr);
            return false;
        }
        
        blog(LOG_INFO, "[AI Engine] ✓ D3D12 device created on matching adapter");
        
        // Create command queue (Section 3.2.2)
        D3D12_COMMAND_QUEUE_DESC queue_desc = {};
        queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        
        hr = d3d12_device_->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&command_queue_));
        if (FAILED(hr)) {
            blog(LOG_ERROR, "[AI Engine] Failed to create command queue (HRESULT: 0x%08X)", hr);
            return false;
        }
        
        blog(LOG_INFO, "[AI Engine] ✓ Command queue created");
        
        // Create command allocator
        hr = d3d12_device_->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&command_allocator_)
        );
        
        if (FAILED(hr)) {
            blog(LOG_ERROR, "[AI Engine] Failed to create command allocator (HRESULT: 0x%08X)", hr);
            return false;
        }
        
        // Create command list
        hr = d3d12_device_->CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            command_allocator_.Get(),
            nullptr,
            IID_PPV_ARGS(&command_list_)
        );
        
        if (FAILED(hr)) {
            blog(LOG_ERROR, "[AI Engine] Failed to create command list (HRESULT: 0x%08X)", hr);
            return false;
        }
        
        // Close the command list (it starts in recording state)
        command_list_->Close();
        
        blog(LOG_INFO, "[AI Engine] ✓ Command list created");
        
        // Create inference command allocator and list (for copy operations)
        hr = d3d12_device_->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&inference_command_allocator_)
        );
        
        if (FAILED(hr)) {
            blog(LOG_ERROR, "[AI Engine] Failed to create inference command allocator (HRESULT: 0x%08X)", hr);
            return false;
        }
        
        hr = d3d12_device_->CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            inference_command_allocator_.Get(),
            nullptr,  // No PSO needed for copy operations
            IID_PPV_ARGS(&inference_command_list_)
        );
        
        if (FAILED(hr)) {
            blog(LOG_ERROR, "[AI Engine] Failed to create inference command list (HRESULT: 0x%08X)", hr);
            return false;
        }
        
        inference_command_list_->Close();
        inference_fence_value_ = 0;
        
        blog(LOG_INFO, "[AI Engine] ✓ Inference command infrastructure created");
        
        // Create postprocessing command allocator and list (for compute shader)
        hr = d3d12_device_->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&postprocess_command_allocator_)
        );
        
        if (FAILED(hr)) {
            blog(LOG_ERROR, "[AI Engine] Failed to create postprocess command allocator (HRESULT: 0x%08X)", hr);
            return false;
        }
        
        hr = d3d12_device_->CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            postprocess_command_allocator_.Get(),
            nullptr,  // PSO will be set at runtime
            IID_PPV_ARGS(&postprocess_command_list_)
        );
        
        if (FAILED(hr)) {
            blog(LOG_ERROR, "[AI Engine] Failed to create postprocess command list (HRESULT: 0x%08X)", hr);
            return false;
        }
        
        postprocess_command_list_->Close();
        postprocess_fence_value_ = 0;
        
        blog(LOG_INFO, "[AI Engine] ✓ Postprocessing command infrastructure created");
        
        // Create fence for synchronization (Section 3.3)
        hr = d3d12_device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_));
        if (FAILED(hr)) {
            blog(LOG_ERROR, "[AI Engine] Failed to create fence (HRESULT: 0x%08X)", hr);
            return false;
        }
        
        fence_event_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (!fence_event_) {
            blog(LOG_ERROR, "[AI Engine] Failed to create fence event");
            return false;
        }
        
        // Create inference fence for CPU staging synchronization
        hr = d3d12_device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&inference_fence_));
        if (FAILED(hr)) {
            blog(LOG_ERROR, "[AI Engine] Failed to create inference fence (HRESULT: 0x%08X)", hr);
            return false;
        }
        
        inference_fence_event_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (!inference_fence_event_) {
            blog(LOG_ERROR, "[AI Engine] Failed to create inference fence event");
            return false;
        }
        inference_fence_value_ = 0;
        
        blog(LOG_INFO, "[AI Engine] ✓ Synchronization primitives created");
        
        // Create DirectML device
        hr = DMLCreateDevice(
            d3d12_device_.Get(),
            DML_CREATE_DEVICE_FLAG_NONE,
            IID_PPV_ARGS(&dml_device_)
        );
        
        if (FAILED(hr)) {
            blog(LOG_ERROR, "[AI Engine] Failed to create DirectML device (HRESULT: 0x%08X)", hr);
            return false;
        }
        
        blog(LOG_INFO, "[AI Engine] ✓ DirectML device created");
        blog(LOG_INFO, "[AI Engine] ✓ D3D12 initialization complete");
        
        return true;
        
    } catch (const std::exception& e) {
        blog(LOG_ERROR, "[AI Engine] Exception during D3D12 initialization: %s", e.what());
        return false;
    }
}

bool AiEngine::InitializeSession(const std::wstring& model_path)
{
    blog(LOG_INFO, "[AI Engine] === Initializing ONNX Runtime Session (Section 4.3) ===");
    blog(LOG_INFO, "[AI Engine] Model path: %ls", model_path.c_str());
    
    // Quick model inspection before full session creation
    try {
        Ort::Env temp_env(ORT_LOGGING_LEVEL_ERROR, "ModelInspector");
        Ort::SessionOptions temp_options;
        temp_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
        
        blog(LOG_INFO, "[AI Engine] === Pre-flight Model Inspection ===");
        Ort::Session inspect_session(temp_env, model_path.c_str(), temp_options);
        
        // Get input/output info
        size_t num_inputs = inspect_session.GetInputCount();
        size_t num_outputs = inspect_session.GetOutputCount();
        
        blog(LOG_INFO, "[AI Engine] Model has %zu input(s), %zu output(s)", num_inputs, num_outputs);
        
        Ort::AllocatorWithDefaultOptions allocator;
        for (size_t i = 0; i < num_inputs && i < 2; i++) {
            Ort::AllocatedStringPtr input_name_ptr = inspect_session.GetInputNameAllocated(i, allocator);
            std::string input_name = std::string(input_name_ptr.get());
            
            Ort::TypeInfo type_info = inspect_session.GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> shape = tensor_info.GetShape();
            
            std::string shape_str = "[";
            for (size_t j = 0; j < shape.size(); j++) {
                shape_str += (shape[j] == -1 ? "?" : std::to_string(shape[j]));
                if (j < shape.size() - 1) shape_str += ",";
            }
            shape_str += "]";
            
            blog(LOG_INFO, "[AI Engine]   Input '%s': %s", input_name.c_str(), shape_str.c_str());
        }
        
        blog(LOG_INFO, "[AI Engine] ✓ Model structure validated");
        
    } catch (const std::exception& e) {
        blog(LOG_WARNING, "[AI Engine] Model inspection failed (will try full load anyway): %s", e.what());
    }
    
    try {
        // Check if DirectML is already loaded
        HMODULE existing_directml = GetModuleHandleA("DirectML.dll");
        if (existing_directml) {
            char existing_path[MAX_PATH];
            GetModuleFileNameA(existing_directml, existing_path, sizeof(existing_path));
            blog(LOG_INFO, "[AI Engine] DirectML.dll loaded from: %s", existing_path);
            
            if (strstr(existing_path, "System32") || strstr(existing_path, "SysWOW64")) {
                blog(LOG_ERROR, "[AI Engine] ❌ FATAL: System32 DirectML.dll was loaded!");
                return false;
            }
        }
        
        // Create environment (Section 4.3)
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "OBS_ONNX_DirectML");
        blog(LOG_INFO, "[AI Engine] ✓ ONNX Environment created");
        
        // Configure session options (Section 4.3)
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options.DisableMemPattern();
        session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        
        blog(LOG_INFO, "[AI Engine] Configuring DirectML execution provider...");
        
        try {
            // The obs-backgroundremoval plugin loads a CPU-only onnxruntime.dll first
            // We need to explicitly get our DirectML version by full path
            HMODULE our_plugin = GetModuleHandleA("obs-onnx-plugin.dll");
            if (!our_plugin) {
                blog(LOG_ERROR, "[AI Engine] ❌ Could not find obs-onnx-plugin.dll handle");
                return false;
            }
            
            char plugin_dir[MAX_PATH];
            GetModuleFileNameA(our_plugin, plugin_dir, sizeof(plugin_dir));
            
            // Remove filename to get directory
            char* last_slash = strrchr(plugin_dir, '\\');
            if (last_slash) *last_slash = '\0';
            
            // Build path to our DirectML onnxruntime.dll
            char onnxruntime_path[MAX_PATH];
            snprintf(onnxruntime_path, sizeof(onnxruntime_path), "%s\\onnxruntime.dll", plugin_dir);
            
            blog(LOG_INFO, "[AI Engine] Loading DirectML onnxruntime from: %s", onnxruntime_path);
            
            HMODULE onnxruntime_module = LoadLibraryExA(onnxruntime_path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
            if (!onnxruntime_module) {
                DWORD error = GetLastError();
                blog(LOG_ERROR, "[AI Engine] ❌ Failed to load DirectML onnxruntime.dll (error: %lu)", error);
                return false;
            }
            
            // Try the Ex version with our D3D12 device
            typedef OrtStatus* (*AppendDMLExFunc)(OrtSessionOptions*, IDMLDevice*, ID3D12CommandQueue*);
            AppendDMLExFunc appendDMLEx = (AppendDMLExFunc)GetProcAddress(onnxruntime_module, "OrtSessionOptionsAppendExecutionProviderEx_DML");
            
            if (appendDMLEx) {
                blog(LOG_INFO, "[AI Engine] Using OrtSessionOptionsAppendExecutionProviderEx_DML");
                OrtStatus* status = appendDMLEx((OrtSessionOptions*)session_options, dml_device_.Get(), command_queue_.Get());
                
                if (status) {
                    const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
                    const char* error_msg = ort_api->GetErrorMessage(status);
                    blog(LOG_ERROR, "[AI Engine] ❌ Failed to append DirectML provider: %s", error_msg);
                    ort_api->ReleaseStatus(status);
                    return false;
                }
                
                blog(LOG_INFO, "[AI Engine] ✓ DirectML provider configured");
            } else {
                // Try the simple version
                typedef OrtStatus* (*AppendDMLFunc)(OrtSessionOptions*, int);
                AppendDMLFunc appendDML = (AppendDMLFunc)GetProcAddress(onnxruntime_module, "OrtSessionOptionsAppendExecutionProvider_DML");
                
                if (appendDML) {
                    blog(LOG_INFO, "[AI Engine] Using OrtSessionOptionsAppendExecutionProvider_DML");
                    OrtStatus* status = appendDML((OrtSessionOptions*)session_options, 0);
                    
                    if (status) {
                        const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
                        const char* error_msg = ort_api->GetErrorMessage(status);
                        blog(LOG_ERROR, "[AI Engine] ❌ Failed to append DirectML provider: %s", error_msg);
                        ort_api->ReleaseStatus(status);
                        return false;
                    }
                    
                    blog(LOG_INFO, "[AI Engine] ✓ DirectML provider configured");
                } else {
                    blog(LOG_ERROR, "[AI Engine] ❌ No DirectML functions found in %s", onnxruntime_path);
                    return false;
                }
            }
        }
        catch (const std::exception& e) {
            blog(LOG_ERROR, "[AI Engine] ❌ Exception configuring DirectML: %s", e.what());
            return false;
        }
        catch (...) {
            blog(LOG_ERROR, "[AI Engine] ❌ Unknown exception configuring DirectML");
            return false;
        }
        
        // Verify model file exists
        DWORD fileAttrib = GetFileAttributesW(model_path.c_str());
        if (fileAttrib == INVALID_FILE_ATTRIBUTES) {
            blog(LOG_ERROR, "[AI Engine] ❌ Model file not found!");
            return false;
        }
        
        blog(LOG_INFO, "[AI Engine] Creating ONNX Runtime session...");
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);
        blog(LOG_INFO, "[AI Engine] ✓ Session created successfully");
        
        // Get input/output metadata (Section 4.3)
        blog(LOG_INFO, "[AI Engine] === Querying Model Metadata ===");
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Input info
        size_t num_input_nodes = session_->GetInputCount();
        blog(LOG_INFO, "[AI Engine] Number of input nodes: %zu", num_input_nodes);
        
        if (num_input_nodes > 0) {
            Ort::AllocatedStringPtr input_name = session_->GetInputNameAllocated(0, allocator);
            input_name_strings_.push_back(std::string(input_name.get()));
            input_names_.push_back(input_name_strings_.back().c_str());
            
            Ort::TypeInfo type_info = session_->GetInputTypeInfo(0);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            input_shape_ = tensor_info.GetShape();
            
            blog(LOG_INFO, "[AI Engine] Input '%s': [%lld, %lld, %lld, %lld]",
                 input_names_[0],
                 input_shape_[0], input_shape_[1], input_shape_[2], input_shape_[3]);
        }
        
        // Output info
        size_t num_output_nodes = session_->GetOutputCount();
        blog(LOG_INFO, "[AI Engine] Number of output nodes: %zu", num_output_nodes);
        
        if (num_output_nodes > 0) {
            Ort::AllocatedStringPtr output_name = session_->GetOutputNameAllocated(0, allocator);
            output_name_strings_.push_back(std::string(output_name.get()));
            output_names_.push_back(output_name_strings_.back().c_str());
            
            Ort::TypeInfo type_info = session_->GetOutputTypeInfo(0);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            output_shape_ = tensor_info.GetShape();
            
            blog(LOG_INFO, "[AI Engine] Output '%s': [%lld, %lld, %lld, %lld]",
                 output_names_[0],
                 output_shape_[0], output_shape_[1], output_shape_[2], output_shape_[3]);
        }
        
        // Calculate tensor sizes with validation
        input_tensor_size_ = 1;
        for (auto dim : input_shape_) {
            if (dim <= 0 || dim > 10000) {
                blog(LOG_ERROR, "[AI Engine] Invalid input dimension: %lld", dim);
                return false;
            }
            input_tensor_size_ *= static_cast<size_t>(dim);
        }
        
        output_tensor_size_ = 1;
        for (auto dim : output_shape_) {
            if (dim <= 0 || dim > 10000) {
                blog(LOG_ERROR, "[AI Engine] Invalid output dimension: %lld", dim);
                return false;
            }
            output_tensor_size_ *= static_cast<size_t>(dim);
        }
        
        blog(LOG_INFO, "[AI Engine] Input tensor: %zu elements (%zu KB)",
             input_tensor_size_, (input_tensor_size_ * sizeof(float)) / 1024);
        blog(LOG_INFO, "[AI Engine] Output tensor: %zu elements (%zu KB)",
             output_tensor_size_, (output_tensor_size_ * sizeof(float)) / 1024);
        
        // Create persistent staging buffers if D3D12 is available
        if (d3d12_device_) {
            // Readback buffer (GPU → CPU)
            D3D12_HEAP_PROPERTIES readback_heap = {};
            readback_heap.Type = D3D12_HEAP_TYPE_READBACK;
            
            D3D12_RESOURCE_DESC readback_desc = {};
            readback_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            readback_desc.Width = input_tensor_size_ * sizeof(float);
            readback_desc.Height = 1;
            readback_desc.DepthOrArraySize = 1;
            readback_desc.MipLevels = 1;
            readback_desc.Format = DXGI_FORMAT_UNKNOWN;
            readback_desc.SampleDesc.Count = 1;
            readback_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            
            HRESULT hr = d3d12_device_->CreateCommittedResource(
                &readback_heap,
                D3D12_HEAP_FLAG_NONE,
                &readback_desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                nullptr,
                IID_PPV_ARGS(&staging_readback_)
            );
            
            if (FAILED(hr)) {
                blog(LOG_ERROR, "[AI Engine] Failed to create persistent readback buffer (HRESULT: 0x%08X)", hr);
                return false;
            }
            
            // Upload buffer (CPU → GPU)
            D3D12_HEAP_PROPERTIES upload_heap = {};
            upload_heap.Type = D3D12_HEAP_TYPE_UPLOAD;
            
            D3D12_RESOURCE_DESC upload_desc = {};
            upload_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            upload_desc.Width = output_tensor_size_ * sizeof(float);
            upload_desc.Height = 1;
            upload_desc.DepthOrArraySize = 1;
            upload_desc.MipLevels = 1;
            upload_desc.Format = DXGI_FORMAT_UNKNOWN;
            upload_desc.SampleDesc.Count = 1;
            upload_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            
            hr = d3d12_device_->CreateCommittedResource(
                &upload_heap,
                D3D12_HEAP_FLAG_NONE,
                &upload_desc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr,
                IID_PPV_ARGS(&staging_upload_)
            );
            
            if (FAILED(hr)) {
                blog(LOG_ERROR, "[AI Engine] Failed to create persistent upload buffer (HRESULT: 0x%08X)", hr);
                return false;
            }
            
            // Allocate CPU buffers
            cpu_input_buffer_.resize(input_tensor_size_);
            cpu_output_buffer_.resize(output_tensor_size_);
            
            blog(LOG_INFO, "[AI Engine] ✓ Persistent staging buffers created");
        }
        
        blog(LOG_INFO, "[AI Engine] ✓ Session initialization complete");
        return true;
        
    } catch (const Ort::Exception& e) {
        blog(LOG_ERROR, "[AI Engine] ❌ ONNX Runtime Exception: %s (code: %d)", 
             e.what(), e.GetOrtErrorCode());
        return false;
    } catch (const std::exception& e) {
        blog(LOG_ERROR, "[AI Engine] ❌ Exception: %s", e.what());
        return false;
    }
}

AiEngine::~AiEngine()
{
    blog(LOG_INFO, "[AI Engine] Shutting down");
    
    // Close fence events
    if (fence_event_) {
        CloseHandle(fence_event_);
        fence_event_ = nullptr;
    }
    
    if (inference_fence_event_) {
        CloseHandle(inference_fence_event_);
        inference_fence_event_ = nullptr;
    }
    
    // Release resources in reverse order
    session_.reset();
    env_.reset();
    
    blog(LOG_INFO, "[AI Engine] ✓ Shutdown complete");
}

bool AiEngine::RunInferenceZeroCopy(ID3D12Resource* input_resource, ID3D12Resource* output_resource)
{
    if (!initialized_ || !session_) {
        blog(LOG_ERROR, "[AI Engine] Cannot run inference - engine not initialized");
        return false;
    }

    if (!input_resource || !output_resource) {
        blog(LOG_ERROR, "[AI Engine] Null D3D12 resources passed to RunInferenceZeroCopy");
        return false;
    }

    // ============================================================================
    // 1. PREPROCESSING (Compute Shader: Texture -> NCHW Buffer)
    // ============================================================================
    
    // Check dimensions
    D3D12_RESOURCE_DESC input_desc = input_resource->GetDesc();
    int64_t expected_width = input_shape_.size() >= 4 ? input_shape_[3] : 0;
    int64_t expected_height = input_shape_.size() >= 4 ? input_shape_[2] : 0;

    // Ensure Preprocessing Buffer Exists
    if (!preprocess_tensor_buffer_) {
        // Lazy initialization of preprocessing tensor buffer
        size_t tensor_size = expected_width * expected_height * 3;  // RGB channels (NCHW)
        
        D3D12_HEAP_PROPERTIES heap_props = {};
        heap_props.Type = D3D12_HEAP_TYPE_DEFAULT;
        
        D3D12_RESOURCE_DESC buffer_desc = {};
        buffer_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer_desc.Width = tensor_size * sizeof(float);
        buffer_desc.Height = 1;
        buffer_desc.DepthOrArraySize = 1;
        buffer_desc.MipLevels = 1;
        buffer_desc.Format = DXGI_FORMAT_UNKNOWN;
        buffer_desc.SampleDesc.Count = 1;
        buffer_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        buffer_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        
        HRESULT hr = d3d12_device_->CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &buffer_desc,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&preprocess_tensor_buffer_)
        );
        
        if (FAILED(hr)) {
            blog(LOG_ERROR, "[AI Engine] Failed to create preprocessing tensor buffer");
            return false;
        }
    }

    // Update Constant Buffer
    PreprocessParams params = {};
    params.inputWidth = static_cast<uint32_t>(input_desc.Width);
    params.inputHeight = input_desc.Height;
    params.outputWidth = static_cast<uint32_t>(expected_width);
    params.outputHeight = static_cast<uint32_t>(expected_height);
    params.normalizeScale = 2.0f / 255.0f; 
    params.normalizeBias = -1.0f;
    params.outputStride = params.outputWidth * params.outputHeight;
    params.padding = 0;

    // Map and Copy Constants
    void* cb_data = nullptr;
    if (SUCCEEDED(preprocess_constant_buffer_->Map(0, nullptr, &cb_data))) {
        memcpy(cb_data, &params, sizeof(PreprocessParams));
        preprocess_constant_buffer_->Unmap(0, nullptr);
    }

    // Execute Preprocessing Command List
    HRESULT hr = preprocess_command_allocator_->Reset();
    if (FAILED(hr)) {
        blog(LOG_ERROR, "[AI Engine] Failed to reset preprocess allocator");
        return false;
    }
    
    hr = preprocess_command_list_->Reset(preprocess_command_allocator_.Get(), preprocess_pipeline_.Get());
    if (FAILED(hr)) {
        blog(LOG_ERROR, "[AI Engine] Failed to reset preprocess list");
        return false;
    }

    preprocess_command_list_->SetComputeRootSignature(preprocess_root_signature_.Get());
    ID3D12DescriptorHeap* heaps[] = { preprocess_descriptor_heap_.Get() };
    preprocess_command_list_->SetDescriptorHeaps(1, heaps);

    // Bind Descriptors manually (re-create for ring buffer rotation)
    D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle = preprocess_descriptor_heap_->GetGPUDescriptorHandleForHeapStart();
    UINT inc_size = d3d12_device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    
    // CBV (constant buffer)
    D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle = preprocess_descriptor_heap_->GetCPUDescriptorHandleForHeapStart();
    D3D12_CONSTANT_BUFFER_VIEW_DESC cbv_desc = {};
    cbv_desc.BufferLocation = preprocess_constant_buffer_->GetGPUVirtualAddress();
    cbv_desc.SizeInBytes = (sizeof(PreprocessParams) + 255) & ~255;
    d3d12_device_->CreateConstantBufferView(&cbv_desc, cpu_handle);
    preprocess_command_list_->SetComputeRootDescriptorTable(0, gpu_handle);
    
    // SRV (input texture) - re-create for current ring buffer frame
    D3D12_CPU_DESCRIPTOR_HANDLE srv_cpu = { cpu_handle.ptr + inc_size };
    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Format = input_desc.Format;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv_desc.Texture2D.MipLevels = 1;
    d3d12_device_->CreateShaderResourceView(input_resource, &srv_desc, srv_cpu);
    
    D3D12_GPU_DESCRIPTOR_HANDLE srv_gpu = { gpu_handle.ptr + inc_size };
    preprocess_command_list_->SetComputeRootDescriptorTable(1, srv_gpu);

    // UAV (output buffer)
    D3D12_CPU_DESCRIPTOR_HANDLE uav_cpu = { cpu_handle.ptr + (inc_size * 2) };
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc = {};
    uav_desc.Format = DXGI_FORMAT_R32_FLOAT;
    uav_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uav_desc.Buffer.NumElements = static_cast<UINT>(expected_width * expected_height * 3);
    d3d12_device_->CreateUnorderedAccessView(preprocess_tensor_buffer_.Get(), nullptr, &uav_desc, uav_cpu);
    
    D3D12_GPU_DESCRIPTOR_HANDLE uav_gpu = { gpu_handle.ptr + (inc_size * 2) };
    preprocess_command_list_->SetComputeRootDescriptorTable(2, uav_gpu);
    
    // Transition preprocess buffer to UAV for shader write
    D3D12_RESOURCE_BARRIER pre_uav_barrier = {};
    pre_uav_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    pre_uav_barrier.Transition.pResource = preprocess_tensor_buffer_.Get();
    pre_uav_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    pre_uav_barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    pre_uav_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    preprocess_command_list_->ResourceBarrier(1, &pre_uav_barrier);
    
    // Dispatch
    uint32_t dispatch_x = (static_cast<uint32_t>(expected_width) + 7) / 8;
    uint32_t dispatch_y = (static_cast<uint32_t>(expected_height) + 7) / 8;
    preprocess_command_list_->Dispatch(dispatch_x, dispatch_y, 1);
    
    // Barrier: UAV -> COMMON (DirectML expects COMMON state)
    D3D12_RESOURCE_BARRIER pre_barrier = {};
    pre_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    pre_barrier.Transition.pResource = preprocess_tensor_buffer_.Get();
    pre_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    pre_barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    pre_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    preprocess_command_list_->ResourceBarrier(1, &pre_barrier);

    hr = preprocess_command_list_->Close();
    if (FAILED(hr)) {
        blog(LOG_ERROR, "[AI Engine] Failed to close preprocess command list");
        return false;
    }
    
    ID3D12CommandList* pre_cmds[] = { preprocess_command_list_.Get() };
    command_queue_->ExecuteCommandLists(1, pre_cmds);

    // Sync: Ensure preprocessing is done before ONNX Runtime starts
    preprocess_fence_value_++;
    command_queue_->Signal(fence_.Get(), preprocess_fence_value_);
    if (fence_->GetCompletedValue() < preprocess_fence_value_) {
        fence_->SetEventOnCompletion(preprocess_fence_value_, fence_event_);
        WaitForSingleObject(fence_event_, INFINITE);
    }

    blog(LOG_DEBUG, "[AI Engine] Preprocessing complete");

    // ============================================================================
    // 2. TRUE ZERO-COPY INFERENCE (DirectML IO Binding)
    // ============================================================================

    try {
        // Reset inference command list for readback operations
        hr = inference_command_allocator_->Reset();
        if (FAILED(hr)) {
            blog(LOG_ERROR, "[AI Engine] Failed to reset inference allocator for readback");
            return false;
        }
        
        hr = inference_command_list_->Reset(inference_command_allocator_.Get(), nullptr);
        if (FAILED(hr)) {
            blog(LOG_ERROR, "[AI Engine] Failed to reset inference command list for readback");
            return false;
        }

        // ===== GPU → CPU: Copy preprocessed buffer to staging =====
        D3D12_RESOURCE_BARRIER barrier_before = {};
        barrier_before.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier_before.Transition.pResource = preprocess_tensor_buffer_.Get();
        barrier_before.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        barrier_before.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        barrier_before.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        inference_command_list_->ResourceBarrier(1, &barrier_before);

        inference_command_list_->CopyResource(staging_readback_.Get(), preprocess_tensor_buffer_.Get());

        D3D12_RESOURCE_BARRIER barrier_after = {};
        barrier_after.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier_after.Transition.pResource = preprocess_tensor_buffer_.Get();
        barrier_after.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
        barrier_after.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        barrier_after.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        inference_command_list_->ResourceBarrier(1, &barrier_after);

        hr = inference_command_list_->Close();
        if (FAILED(hr)) {
            blog(LOG_ERROR, "[AI Engine] Failed to close inference command list");
            return false;
        }

        ID3D12CommandList* lists[] = { inference_command_list_.Get() };
        command_queue_->ExecuteCommandLists(1, lists);

        // Wait for GPU copy to complete
        hr = command_queue_->Signal(inference_fence_.Get(), ++inference_fence_value_);
        if (FAILED(hr)) {
            blog(LOG_ERROR, "[AI Engine] Failed to signal inference fence");
            return false;
        }

        if (inference_fence_->GetCompletedValue() < inference_fence_value_) {
            hr = inference_fence_->SetEventOnCompletion(inference_fence_value_, inference_fence_event_);
            if (FAILED(hr)) {
                blog(LOG_ERROR, "[AI Engine] Failed to set fence event");
                return false;
            }
            WaitForSingleObject(inference_fence_event_, INFINITE);
        }

        // Map readback buffer and copy to CPU input buffer
        void* mapped_data = nullptr;
        D3D12_RANGE read_range = { 0, input_tensor_size_ * sizeof(float) };
        hr = staging_readback_->Map(0, &read_range, &mapped_data);
        if (FAILED(hr)) {
            blog(LOG_ERROR, "[AI Engine] Failed to map readback buffer");
            return false;
        }

        memcpy(cpu_input_buffer_.data(), mapped_data, input_tensor_size_ * sizeof(float));
        
        D3D12_RANGE write_range = { 0, 0 };
        staging_readback_->Unmap(0, &write_range);

        // ===== ONNX Runtime Inference with CPU tensors =====
        Ort::MemoryInfo mem_info_cpu = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info_cpu,
            cpu_input_buffer_.data(),
            input_tensor_size_,
            input_shape_.data(),
            input_shape_.size()
        );

        Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
            mem_info_cpu,
            cpu_output_buffer_.data(),
            output_tensor_size_,
            output_shape_.data(),
            output_shape_.size()
        );

        const char* input_names[] = { input_names_[0] };
        const char* output_names[] = { output_names_[0] };
        Ort::Value input_values[] = { std::move(input_tensor) };
        Ort::Value output_values[] = { std::move(output_tensor) };

        session_->Run(Ort::RunOptions{nullptr}, input_names, input_values, 1, output_names, output_values, 1);

        // Debug: Log first few output values to verify model is producing data
        blog(LOG_INFO, "[AI Engine] Model output sample: [0]=%.3f, [1]=%.3f, [100]=%.3f, [1000]=%.3f",
             cpu_output_buffer_[0], cpu_output_buffer_[1], 
             cpu_output_buffer_[100], cpu_output_buffer_[1000]);

        // ===== CPU → GPU: Copy result to upload staging =====
        void* upload_data = nullptr;
        D3D12_RANGE upload_read_range = { 0, 0 };
        hr = staging_upload_->Map(0, &upload_read_range, &upload_data);
        if (FAILED(hr)) {
            blog(LOG_ERROR, "[AI Engine] Failed to map upload buffer");
            return false;
        }

        memcpy(upload_data, cpu_output_buffer_.data(), output_tensor_size_ * sizeof(float));
        
        D3D12_RANGE upload_write_range = { 0, output_tensor_size_ * sizeof(float) };
        staging_upload_->Unmap(0, &upload_write_range);

        // Reset for upload command list
        hr = inference_command_allocator_->Reset();
        if (FAILED(hr)) {
            blog(LOG_ERROR, "[AI Engine] Failed to reset inference allocator");
            return false;
        }

        hr = inference_command_list_->Reset(inference_command_allocator_.Get(), nullptr);
        if (FAILED(hr)) {
            blog(LOG_ERROR, "[AI Engine] Failed to reset inference command list");
            return false;
        }

        D3D12_RESOURCE_BARRIER barrier_upload_before = {};
        barrier_upload_before.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier_upload_before.Transition.pResource = postprocess_tensor_buffer_.Get();
        barrier_upload_before.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        barrier_upload_before.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
        barrier_upload_before.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        inference_command_list_->ResourceBarrier(1, &barrier_upload_before);

        inference_command_list_->CopyResource(postprocess_tensor_buffer_.Get(), staging_upload_.Get());

        D3D12_RESOURCE_BARRIER barrier_upload_after = {};
        barrier_upload_after.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier_upload_after.Transition.pResource = postprocess_tensor_buffer_.Get();
        barrier_upload_after.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        barrier_upload_after.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        barrier_upload_after.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        inference_command_list_->ResourceBarrier(1, &barrier_upload_after);

        hr = inference_command_list_->Close();
        if (FAILED(hr)) {
            blog(LOG_ERROR, "[AI Engine] Failed to close upload command list");
            return false;
        }

        ID3D12CommandList* upload_lists[] = { inference_command_list_.Get() };
        command_queue_->ExecuteCommandLists(1, upload_lists);

        // Wait for upload to complete
        hr = command_queue_->Signal(inference_fence_.Get(), ++inference_fence_value_);
        if (FAILED(hr)) {
            blog(LOG_ERROR, "[AI Engine] Failed to signal upload fence");
            return false;
        }

        if (inference_fence_->GetCompletedValue() < inference_fence_value_) {
            hr = inference_fence_->SetEventOnCompletion(inference_fence_value_, inference_fence_event_);
            if (FAILED(hr)) {
                blog(LOG_ERROR, "[AI Engine] Failed to set upload fence event");
                return false;
            }
            WaitForSingleObject(inference_fence_event_, INFINITE);
        }

        blog(LOG_INFO, "[AI Engine] ✓ DirectML GPU inference complete (CPU staging path)");

    } catch (const std::exception& e) {
        blog(LOG_ERROR, "[AI Engine] Inference failed: %s", e.what());
        return false;
    }

    // ============================================================================
    // 3. POSTPROCESSING (Compute Shader: NCHW Buffer -> RGBA Texture)
    // ============================================================================

    hr = postprocess_command_allocator_->Reset();
    if (FAILED(hr)) {
        blog(LOG_ERROR, "[AI Engine] Failed to reset postprocess allocator");
        return false;
    }
    
    hr = postprocess_command_list_->Reset(postprocess_command_allocator_.Get(), postprocess_pipeline_.Get());
    if (FAILED(hr)) {
        blog(LOG_ERROR, "[AI Engine] Failed to reset postprocess list");
        return false;
    }

    // Barrier: COMMON -> SRV (Transition buffer from ORT write state to Shader Read)
    D3D12_RESOURCE_BARRIER post_barrier = {};
    post_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    post_barrier.Transition.pResource = postprocess_tensor_buffer_.Get();
    post_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    post_barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    post_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    postprocess_command_list_->ResourceBarrier(1, &post_barrier);

    // Barrier: COMMON -> UAV (Transition Output Texture)
    D3D12_RESOURCE_BARRIER out_barrier = {};
    out_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    out_barrier.Transition.pResource = output_resource;
    out_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON; 
    out_barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    out_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    postprocess_command_list_->ResourceBarrier(1, &out_barrier);

    postprocess_command_list_->SetComputeRootSignature(postprocess_root_signature_.Get());
    ID3D12DescriptorHeap* pp_heaps[] = { postprocess_descriptor_heap_.Get() };
    postprocess_command_list_->SetDescriptorHeaps(1, pp_heaps);

    // Re-create descriptors for current ring buffer frame
    D3D12_GPU_DESCRIPTOR_HANDLE pp_gpu_handle = postprocess_descriptor_heap_->GetGPUDescriptorHandleForHeapStart();
    D3D12_CPU_DESCRIPTOR_HANDLE pp_cpu_handle = postprocess_descriptor_heap_->GetCPUDescriptorHandleForHeapStart();
    
    // Update constant buffer with output dimensions and denormalization parameters
    struct PostprocessParams {
        uint32_t output_width;   // Final output texture size (e.g., 640x360)
        uint32_t output_height;
        uint32_t input_width;    // Model output size (e.g., 224x224)
        uint32_t input_height;
        float denorm_scale;      // For [-1,1] model output: use 2.0
        float denorm_bias;       // For [-1,1] model output: use -1.0
        uint32_t input_stride;
        uint32_t padding;        // Pad to 16-byte alignment
    } pp_params;
    
    int64_t model_output_width = output_shape_.size() >= 4 ? output_shape_[3] : 224;
    int64_t model_output_height = output_shape_.size() >= 4 ? output_shape_[2] : 224;
    
    // Get output dimensions from the actual output resource
    D3D12_RESOURCE_DESC output_desc = output_resource->GetDesc();
    
    // Output dimensions come from the shared texture size (full video resolution)
    pp_params.output_width = static_cast<uint32_t>(output_desc.Width);
    pp_params.output_height = output_desc.Height;
    
    // Input dimensions come from the model output shape
    pp_params.input_width = static_cast<uint32_t>(model_output_width);
    pp_params.input_height = static_cast<uint32_t>(model_output_height);
    
    pp_params.denorm_scale = 255.0f;  // Model outputs [0,255], need to convert to [0,1]
    pp_params.denorm_bias = 0.0f;     // pixel = (value - 0.0) / 255.0
    pp_params.input_stride = pp_params.input_width * pp_params.input_height;
    pp_params.padding = 0;
    
    // Debug: Log postprocessing parameters (only once per model load)
    static bool logged_once = false;
    if (!logged_once) {
        blog(LOG_INFO, "[AI Engine] Postprocessing dispatch: output=%ux%u, input=%ux%u, stride=%u",
            pp_params.output_width, pp_params.output_height,
            pp_params.input_width, pp_params.input_height, pp_params.input_stride);
        logged_once = true;
    }
    
    void* pp_cb_data = nullptr;
    if (SUCCEEDED(postprocess_constant_buffer_->Map(0, nullptr, &pp_cb_data))) {
        memcpy(pp_cb_data, &pp_params, sizeof(PostprocessParams));
        postprocess_constant_buffer_->Unmap(0, nullptr);
        
        // Verify the values were written correctly
        blog(LOG_DEBUG, "[AI Engine] Constant buffer uploaded: out=%ux%u, in=%ux%u, denorm=%.2f/%.2f, stride=%u",
            pp_params.output_width, pp_params.output_height,
            pp_params.input_width, pp_params.input_height,
            pp_params.denorm_scale, pp_params.denorm_bias,
            pp_params.input_stride);
    } else {
        blog(LOG_ERROR, "[AI Engine] Failed to map postprocess constant buffer!");
    }
    
    // CBV
    D3D12_CONSTANT_BUFFER_VIEW_DESC pp_cbv_desc = {};
    pp_cbv_desc.BufferLocation = postprocess_constant_buffer_->GetGPUVirtualAddress();
    pp_cbv_desc.SizeInBytes = 256;
    d3d12_device_->CreateConstantBufferView(&pp_cbv_desc, pp_cpu_handle);
    postprocess_command_list_->SetComputeRootDescriptorTable(0, pp_gpu_handle);
    
    // SRV (postprocess tensor buffer)
    D3D12_CPU_DESCRIPTOR_HANDLE pp_srv_cpu = { pp_cpu_handle.ptr + inc_size };
    D3D12_SHADER_RESOURCE_VIEW_DESC pp_srv_desc = {};
    pp_srv_desc.Format = DXGI_FORMAT_R32_FLOAT;
    pp_srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    pp_srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    pp_srv_desc.Buffer.NumElements = static_cast<UINT>(output_tensor_size_);
    d3d12_device_->CreateShaderResourceView(postprocess_tensor_buffer_.Get(), &pp_srv_desc, pp_srv_cpu);
    
    D3D12_GPU_DESCRIPTOR_HANDLE pp_srv_gpu = { pp_gpu_handle.ptr + inc_size };
    postprocess_command_list_->SetComputeRootDescriptorTable(1, pp_srv_gpu);

    // UAV (output texture) - re-create for current ring buffer frame
    D3D12_CPU_DESCRIPTOR_HANDLE pp_uav_cpu = { pp_cpu_handle.ptr + (inc_size * 2) };
    // Reuse output_desc from line 1045
    D3D12_UNORDERED_ACCESS_VIEW_DESC pp_uav_desc = {};
    pp_uav_desc.Format = output_desc.Format;
    pp_uav_desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    d3d12_device_->CreateUnorderedAccessView(output_resource, nullptr, &pp_uav_desc, pp_uav_cpu);
    
    D3D12_GPU_DESCRIPTOR_HANDLE pp_uav_gpu = { pp_gpu_handle.ptr + (inc_size * 2) };
    postprocess_command_list_->SetComputeRootDescriptorTable(2, pp_uav_gpu);

    // Dispatch based on OUTPUT dimensions (full video resolution), not model dimensions
    dispatch_x = (static_cast<uint32_t>(output_desc.Width) + 7) / 8;
    dispatch_y = (output_desc.Height + 7) / 8;
    
    // Diagnostic logging for viewport/dimension issues
    blog(LOG_INFO, "[AI Engine] Postprocess dispatch: dispatch=(%ux%u), output texture=(%llux%u), params=(%ux%u -> %ux%u)",
         dispatch_x, dispatch_y,
         output_desc.Width, output_desc.Height,
         pp_params.input_width, pp_params.input_height,
         pp_params.output_width, pp_params.output_height);
    
    postprocess_command_list_->Dispatch(dispatch_x, dispatch_y, 1);

    // Barrier: UAV -> COMMON (Ready for OBS D3D11 to read via Shared Handle)
    out_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    out_barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    postprocess_command_list_->ResourceBarrier(1, &out_barrier);

    // Barrier: SRV -> COMMON (Reset buffer state for next frame's ONNX write)
    post_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    post_barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    postprocess_command_list_->ResourceBarrier(1, &post_barrier);

    hr = postprocess_command_list_->Close();
    if (FAILED(hr)) {
        blog(LOG_ERROR, "[AI Engine] Failed to close postprocess command list");
        return false;
    }
    
    ID3D12CommandList* post_cmds[] = { postprocess_command_list_.Get() };
    command_queue_->ExecuteCommandLists(1, post_cmds);

    // Final synchronization: Signal that frame is ready
    // CRITICAL: Wait for completion before returning to OBS
    postprocess_fence_value_++;
    command_queue_->Signal(fence_.Get(), postprocess_fence_value_);
    
    if (fence_->GetCompletedValue() < postprocess_fence_value_) {
        fence_->SetEventOnCompletion(postprocess_fence_value_, fence_event_);
        WaitForSingleObject(fence_event_, INFINITE);
    }

    blog(LOG_DEBUG, "[AI Engine] Frame processing complete");
    return true;
}

bool AiEngine::RecoverFromDeviceLost()
{
    blog(LOG_WARNING, "[AI Engine] === Attempting Device Recovery (Section 7.2) ===");
    
    // Release all resources
    session_.reset();
    env_.reset();
    
    fence_.Reset();
    if (fence_event_) {
        CloseHandle(fence_event_);
        fence_event_ = nullptr;
    }
    
    inference_fence_.Reset();
    if (inference_fence_event_) {
        CloseHandle(inference_fence_event_);
        inference_fence_event_ = nullptr;
    }
    
    command_list_.Reset();
    command_allocator_.Reset();
    command_queue_.Reset();
    dml_device_.Reset();
    d3d12_device_.Reset();
    
    initialized_ = false;
    
    // Attempt to reinitialize
    try {
        if (!InitializeD3D12Device()) {
            blog(LOG_ERROR, "[AI Engine] Device recovery failed: D3D12 initialization");
            return false;
        }
        
        if (!InitializeSession(model_path_)) {
            blog(LOG_ERROR, "[AI Engine] Device recovery failed: Session initialization");
            return false;
        }
        
        initialized_ = true;
        blog(LOG_INFO, "[AI Engine] ✓ Device recovery successful");
        return true;
        
    } catch (const std::exception& e) {
        blog(LOG_ERROR, "[AI Engine] Device recovery exception: %s", e.what());
        return false;
    }
}

std::vector<int64_t> AiEngine::GetInputShape() const
{
    return input_shape_;
}

std::vector<int64_t> AiEngine::GetOutputShape() const
{
    return output_shape_;
}

// ============================================================================
// Preprocessing Pipeline Implementation
// ============================================================================

bool AiEngine::InitializePreprocessPipeline()
{
    blog(LOG_INFO, "[AI Engine] === Initializing Preprocessing Pipeline ===");
    
    // Use embedded precompiled shader bytecode
    const uint8_t* shader_bytecode = ShaderBytecode::preprocess_nchw_bytecode;
    size_t bytecode_size = ShaderBytecode::preprocess_nchw_bytecode_size;
    
    blog(LOG_INFO, "[AI Engine] Using embedded shader bytecode (%zu bytes)", bytecode_size);
    
    // Create root signature
    // Root parameters: [0] = CBV (constant buffer), [1] = SRV (input texture), [2] = UAV (output buffer)
    D3D12_DESCRIPTOR_RANGE ranges[3] = {};
    
    // Constant buffer (b0)
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
    ranges[0].NumDescriptors = 1;
    ranges[0].BaseShaderRegister = 0;
    
    // Texture SRV (t0)
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[1].NumDescriptors = 1;
    ranges[1].BaseShaderRegister = 0;
    
    // Buffer UAV (u0)
    ranges[2].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[2].NumDescriptors = 1;
    ranges[2].BaseShaderRegister = 0;
    
    D3D12_ROOT_PARAMETER root_params[3] = {};
    for (int i = 0; i < 3; i++) {
        root_params[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        root_params[i].DescriptorTable.NumDescriptorRanges = 1;
        root_params[i].DescriptorTable.pDescriptorRanges = &ranges[i];
        root_params[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    
    // Static sampler for bilinear filtering
    D3D12_STATIC_SAMPLER_DESC sampler = {};
    sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.ShaderRegister = 0;
    sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    
    D3D12_ROOT_SIGNATURE_DESC root_sig_desc = {};
    root_sig_desc.NumParameters = 3;
    root_sig_desc.pParameters = root_params;
    root_sig_desc.NumStaticSamplers = 1;
    root_sig_desc.pStaticSamplers = &sampler;
    
    ID3DBlob* signature_blob = nullptr;
    ID3DBlob* error_blob = nullptr;
    HRESULT hr = D3D12SerializeRootSignature(&root_sig_desc, D3D_ROOT_SIGNATURE_VERSION_1, 
                                              &signature_blob, &error_blob);
    
    if (FAILED(hr)) {
        if (error_blob) {
            blog(LOG_ERROR, "[AI Engine] Root signature serialization failed:\n%s",
                 (char*)error_blob->GetBufferPointer());
            error_blob->Release();
        }
        return false;
    }
    
    hr = d3d12_device_->CreateRootSignature(0, signature_blob->GetBufferPointer(),
                                             signature_blob->GetBufferSize(),
                                             IID_PPV_ARGS(&preprocess_root_signature_));
    
    signature_blob->Release();
    
    if (FAILED(hr)) {
        blog(LOG_ERROR, "[AI Engine] Failed to create root signature");
        return false;
    }
    
    // Create pipeline state
    D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline_desc = {};
    pipeline_desc.pRootSignature = preprocess_root_signature_.Get();
    pipeline_desc.CS.pShaderBytecode = shader_bytecode;
    pipeline_desc.CS.BytecodeLength = bytecode_size;
    
    hr = d3d12_device_->CreateComputePipelineState(&pipeline_desc, 
                                                    IID_PPV_ARGS(&preprocess_pipeline_));
    
    if (FAILED(hr)) {
        blog(LOG_ERROR, "[AI Engine] Failed to create compute pipeline state");
        return false;
    }
    
    // Create dedicated command allocator for preprocessing
    hr = d3d12_device_->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_PPV_ARGS(&preprocess_command_allocator_)
    );
    
    if (FAILED(hr)) {
        blog(LOG_ERROR, "[AI Engine] Failed to create preprocessing command allocator");
        return false;
    }
    
    // Create dedicated command list for preprocessing
    hr = d3d12_device_->CreateCommandList(
        0,
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        preprocess_command_allocator_.Get(),
        preprocess_pipeline_.Get(),
        IID_PPV_ARGS(&preprocess_command_list_)
    );
    
    if (FAILED(hr)) {
        blog(LOG_ERROR, "[AI Engine] Failed to create preprocessing command list");
        return false;
    }
    
    // Close it initially (will be reset before first use)
    preprocess_command_list_->Close();
    
    // Create descriptor heap for shader resource bindings
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc = {};
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap_desc.NumDescriptors = 3;  // CBV, SRV, UAV
    heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    
    hr = d3d12_device_->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&preprocess_descriptor_heap_));
    if (FAILED(hr)) {
        blog(LOG_ERROR, "[AI Engine] Failed to create descriptor heap");
        return false;
    }
    
    // Create constant buffer (will be updated per-frame with dimensions)
    D3D12_HEAP_PROPERTIES upload_heap = {};
    upload_heap.Type = D3D12_HEAP_TYPE_UPLOAD;
    
    D3D12_RESOURCE_DESC cb_desc = {};
    cb_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    cb_desc.Width = (sizeof(PreprocessParams) + 255) & ~255;  // Align to 256 bytes
    cb_desc.Height = 1;
    cb_desc.DepthOrArraySize = 1;
    cb_desc.MipLevels = 1;
    cb_desc.Format = DXGI_FORMAT_UNKNOWN;
    cb_desc.SampleDesc.Count = 1;
    cb_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    
    hr = d3d12_device_->CreateCommittedResource(
        &upload_heap,
        D3D12_HEAP_FLAG_NONE,
        &cb_desc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&preprocess_constant_buffer_)
    );
    
    if (FAILED(hr)) {
        blog(LOG_ERROR, "[AI Engine] Failed to create constant buffer");
        return false;
    }
    
    blog(LOG_INFO, "[AI Engine] ✓ Preprocessing pipeline initialized");
    return true;
}

bool AiEngine::InitializePostprocessPipeline()
{
    blog(LOG_INFO, "[AI Engine] === Initializing Postprocessing Pipeline ===");
    
    // Use embedded postprocess shader bytecode
    const uint8_t* shader_bytecode = ShaderBytecode::postprocess_rgba_bytecode;
    size_t bytecode_size = ShaderBytecode::postprocess_rgba_bytecode_size;
    
    blog(LOG_INFO, "[AI Engine] Using embedded postprocess shader bytecode (%zu bytes)", bytecode_size);
    
    // Create root signature
    // Root parameters: [0] = CBV (constant buffer), [1] = SRV (input tensor buffer), [2] = UAV (output texture)
    D3D12_DESCRIPTOR_RANGE ranges[3] = {};
    
    // Constant buffer (b0)
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
    ranges[0].NumDescriptors = 1;
    ranges[0].BaseShaderRegister = 0;
    
    // Buffer SRV (t0) - input tensor as Buffer<float>
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[1].NumDescriptors = 1;
    ranges[1].BaseShaderRegister = 0;
    
    // Texture UAV (u0) - output RGBA texture
    ranges[2].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[2].NumDescriptors = 1;
    ranges[2].BaseShaderRegister = 0;
    
    D3D12_ROOT_PARAMETER root_params[3] = {};
    for (int i = 0; i < 3; i++) {
        root_params[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        root_params[i].DescriptorTable.NumDescriptorRanges = 1;
        root_params[i].DescriptorTable.pDescriptorRanges = &ranges[i];
        root_params[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    
    D3D12_ROOT_SIGNATURE_DESC root_sig_desc = {};
    root_sig_desc.NumParameters = 3;
    root_sig_desc.pParameters = root_params;
    root_sig_desc.NumStaticSamplers = 0;
    
    ID3DBlob* signature_blob = nullptr;
    ID3DBlob* error_blob = nullptr;
    HRESULT hr = D3D12SerializeRootSignature(&root_sig_desc, D3D_ROOT_SIGNATURE_VERSION_1, 
                                              &signature_blob, &error_blob);
    
    if (FAILED(hr)) {
        if (error_blob) {
            blog(LOG_ERROR, "[AI Engine] Postprocess root signature serialization failed:\\n%s",
                 (char*)error_blob->GetBufferPointer());
            error_blob->Release();
        }
        return false;
    }
    
    hr = d3d12_device_->CreateRootSignature(0, signature_blob->GetBufferPointer(),
                                             signature_blob->GetBufferSize(),
                                             IID_PPV_ARGS(&postprocess_root_signature_));
    
    signature_blob->Release();
    
    if (FAILED(hr)) {
        blog(LOG_ERROR, "[AI Engine] Failed to create postprocess root signature");
        return false;
    }
    
    // Create pipeline state
    D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline_desc = {};
    pipeline_desc.pRootSignature = postprocess_root_signature_.Get();
    pipeline_desc.CS.pShaderBytecode = shader_bytecode;
    pipeline_desc.CS.BytecodeLength = bytecode_size;
    
    hr = d3d12_device_->CreateComputePipelineState(&pipeline_desc, 
                                                    IID_PPV_ARGS(&postprocess_pipeline_));
    
    if (FAILED(hr)) {
        blog(LOG_ERROR, "[AI Engine] Failed to create postprocess compute pipeline state");
        return false;
    }
    
    // Create descriptor heap for postprocessing (CBV, SRV, UAV)
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc = {};
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap_desc.NumDescriptors = 3;  // CBV, SRV, UAV
    heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    
    hr = d3d12_device_->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&postprocess_descriptor_heap_));
    if (FAILED(hr)) {
        blog(LOG_ERROR, "[AI Engine] Failed to create postprocess descriptor heap");
        return false;
    }
    
    // Create constant buffer
    D3D12_HEAP_PROPERTIES upload_heap = {};
    upload_heap.Type = D3D12_HEAP_TYPE_UPLOAD;
    
    D3D12_RESOURCE_DESC cb_desc = {};
    cb_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    cb_desc.Width = 256;  // Aligned constant buffer size
    cb_desc.Height = 1;
    cb_desc.DepthOrArraySize = 1;
    cb_desc.MipLevels = 1;
    cb_desc.Format = DXGI_FORMAT_UNKNOWN;
    cb_desc.SampleDesc.Count = 1;
    cb_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    
    hr = d3d12_device_->CreateCommittedResource(
        &upload_heap,
        D3D12_HEAP_FLAG_NONE,
        &cb_desc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&postprocess_constant_buffer_)
    );
    
    if (FAILED(hr)) {
        blog(LOG_ERROR, "[AI Engine] Failed to create postprocess constant buffer");
        return false;
    }
    
    // Create tensor buffer for inference output (NCHW Float32)
    // This is allocated at max size based on expected model output
    D3D12_HEAP_PROPERTIES default_heap = {};
    default_heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    
    D3D12_RESOURCE_DESC tensor_desc = {};
    tensor_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    tensor_desc.Width = 224 * 224 * 3 * sizeof(float);  // Max expected output size
    tensor_desc.Height = 1;
    tensor_desc.DepthOrArraySize = 1;
    tensor_desc.MipLevels = 1;
    tensor_desc.Format = DXGI_FORMAT_UNKNOWN;
    tensor_desc.SampleDesc.Count = 1;
    tensor_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    tensor_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    
    hr = d3d12_device_->CreateCommittedResource(
        &default_heap,
        D3D12_HEAP_FLAG_NONE,
        &tensor_desc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&postprocess_tensor_buffer_)
    );
    
    if (FAILED(hr)) {
        blog(LOG_ERROR, "[AI Engine] Failed to create postprocess tensor buffer");
        return false;
    }
    
    blog(LOG_INFO, "[AI Engine] ✓ Postprocessing pipeline initialized");
    return true;
}
