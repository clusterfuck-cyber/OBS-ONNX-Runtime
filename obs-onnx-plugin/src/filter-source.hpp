/*
 * OBS ONNX Filter Source - Header (DirectML Zero-Copy Implementation)
 * Defines the filter source structure and interface
 * Architecture based on "Advanced Engineering Report: Architecting High-Performance AI Plugins for OBS Studio"
 */

#pragma once

#include <obs-module.h>
#include <graphics/graphics.h>
#include <d3d11.h>
#include <d3d11_1.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <memory>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>

// Forward declaration
class AiEngine;

// Context object for ring buffer (Section 6.3)
struct FrameContext {
    // D3D11 shared resources with keyed mutex (Section 3.2)
    ID3D11Texture2D* d3d11_shared_input;
    ID3D11Texture2D* d3d11_shared_output;
    IDXGIKeyedMutex* input_mutex;
    IDXGIKeyedMutex* output_mutex;
    
    // NT Handles for cross-device sharing
    HANDLE shared_input_handle;
    HANDLE shared_output_handle;
    
    // D3D12 imported resources for DirectML
    ID3D12Resource* d3d12_input_resource;
    ID3D12Resource* d3d12_output_resource;
    
    // Preprocessed tensor buffer (NCHW format)
    ID3D12Resource* preprocessed_tensor;
    
    // State flags
    bool is_processing;
    bool has_valid_output;
    
    FrameContext() 
        : d3d11_shared_input(nullptr)
        , d3d11_shared_output(nullptr)
        , input_mutex(nullptr)
        , output_mutex(nullptr)
        , shared_input_handle(nullptr)
        , shared_output_handle(nullptr)
        , d3d12_input_resource(nullptr)
        , d3d12_output_resource(nullptr)
        , preprocessed_tensor(nullptr)
        , is_processing(false)
        , has_valid_output(false)
    {}
};

struct onnx_filter_data {
    // OBS Context
    obs_source_t *context;
    
    // Source dimensions
    uint32_t width;
    uint32_t height;
    
    // AI Engine (owns D3D12 device and ONNX session)
    std::unique_ptr<AiEngine> ai_engine;
    
    // D3D11 device from OBS (Device A - Main Thread)
    ID3D11Device* d3d11_device;
    ID3D11DeviceContext* d3d11_context;
    
    // Separate D3D11 device for worker thread (Device B - Thread-Safe)
    ID3D11Device* worker_device;
    ID3D11DeviceContext* worker_context;
    
    // Worker's view of shared textures
    struct WorkerFrameContext {
        ID3D11Texture2D* input_texture;
        ID3D11Texture2D* output_texture;
        IDXGIKeyedMutex* input_mutex;
        IDXGIKeyedMutex* output_mutex;
        
        WorkerFrameContext()
            : input_texture(nullptr)
            , output_texture(nullptr)
            , input_mutex(nullptr)
            , output_mutex(nullptr)
        {}
    };
    WorkerFrameContext worker_contexts[3];  // RING_BUFFER_SIZE
    
    // Ring buffer for asynchronous processing (Section 6.3)
    static const int RING_BUFFER_SIZE = 3;
    FrameContext frame_contexts[RING_BUFFER_SIZE];
    int current_input_index;
    int current_output_index;
    
    // Asynchronous worker thread (Section 6.2)
    std::unique_ptr<std::thread> worker_thread;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::atomic<bool> worker_running;
    std::queue<int> pending_frames;  // Indices into frame_contexts
    
    // Compute shader for preprocessing (Section 5.2)
    ID3D12RootSignature* root_signature;
    ID3D12PipelineState* preprocessing_pso;
    ID3D12Resource* preprocessing_upload_buffer;
    
    // Graphics Resources
    gs_texrender_t *texrender;
    gs_texture_t *output_texture;  // Processed output texture (full size)
    gs_texture_t *ai_texture;      // AI model output (224x224)
    
    // State flags
    bool initialized;
    bool backend_supported;
    bool device_lost;
    bool has_rendered_once;  // Track if we've ever produced output
    
    // Model path
    std::string model_path;
    std::wstring pending_model_path;
    
    // Statistics
    std::atomic<uint64_t> frames_processed;
    std::atomic<uint64_t> frames_dropped;
    
    onnx_filter_data()
        : context(nullptr)
        , width(0)
        , height(0)
        , d3d11_device(nullptr)
        , d3d11_context(nullptr)
        , worker_device(nullptr)
        , worker_context(nullptr)
        , current_input_index(0)
        , current_output_index(0)
        , root_signature(nullptr)
        , preprocessing_pso(nullptr)
        , preprocessing_upload_buffer(nullptr)
        , texrender(nullptr)
        , output_texture(nullptr)
        , ai_texture(nullptr)
        , initialized(false)
        , backend_supported(true)
        , device_lost(false)
        , has_rendered_once(false)
        , frames_processed(0)
        , frames_dropped(0)
        , worker_running(false)
    {}
};

// OBS Source Info callbacks
extern struct obs_source_info onnx_filter_info;

// Filter lifecycle functions
void *onnx_filter_create(obs_data_t *settings, obs_source_t *source);
void onnx_filter_destroy(void *data);
void onnx_filter_update(void *data, obs_data_t *settings);
void onnx_filter_video_render(void *data, gs_effect_t *effect);
void onnx_filter_video_tick(void *data, float seconds);
obs_properties_t *onnx_filter_properties(void *data);
void onnx_filter_defaults(obs_data_t *settings);
const char *onnx_filter_get_name(void *unused);
uint32_t onnx_filter_get_width(void *data);
uint32_t onnx_filter_get_height(void *data);

// Worker thread function (Section 6.2)
void worker_thread_func(onnx_filter_data* filter);
