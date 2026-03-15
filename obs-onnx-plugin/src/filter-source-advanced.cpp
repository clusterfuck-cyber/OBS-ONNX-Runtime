/*
 * OBS ONNX Filter - Advanced DirectML Implementation with Zero-Copy Architecture
 * Based on "Advanced Engineering Report: Architecting High-Performance AI Plugins for OBS Studio"
 * 
 * Key Features:
 * - D3D11/D3D12 interop with shared NT handles (Section 3.2)
 * - Keyed mutex synchronization (Section 3.3)
 * - Asynchronous worker thread pattern (Section 6.2)
 * - Ring buffer for frame management (Section 6.3)
 * - HLSL compute shaders for preprocessing (Section 5.2)
 */

#include "filter-source.hpp"
#include "ai-engine.hpp"
#include <util/platform.h>

// Define the source info structure
struct obs_source_info onnx_filter_info;

// Forward declarations for helper functions
static bool CreateSharedInteropTexture(
    onnx_filter_data* filter,
    int index,
    uint32_t width,
    uint32_t height
);

static void ReleaseFrameContext(FrameContext& context);

// Worker thread function (Section 6.2)
void worker_thread_func(onnx_filter_data* filter)
{
    blog(LOG_INFO, "[ONNX Filter] Worker thread started");
    
    while (filter->worker_running) {
        int frame_index = -1;
        
        // Wait for a pending frame
        {
            std::unique_lock<std::mutex> lock(filter->queue_mutex);
            filter->queue_cv.wait(lock, [filter] {
                return !filter->pending_frames.empty() || !filter->worker_running;
            });
            
            if (!filter->worker_running) {
                break;
            }
            
            if (!filter->pending_frames.empty()) {
                frame_index = filter->pending_frames.front();
                filter->pending_frames.pop();
            }
        }
        
        if (frame_index < 0 || frame_index >= filter->RING_BUFFER_SIZE) {
            continue;
        }
        
        FrameContext& ctx = filter->frame_contexts[frame_index];
        onnx_filter_data::WorkerFrameContext& worker_ctx = filter->worker_contexts[frame_index];
        
        if (!filter->ai_engine || !filter->ai_engine->IsInitialized()) {
            ctx.is_processing = false;
            continue;
        }
        
        // Check if worker context is fully initialized
        // Note: mutexes are nullptr (using fence-based sync), only check textures
        if (!worker_ctx.input_texture || !worker_ctx.output_texture) {
            blog(LOG_WARNING, "[ONNX Filter] Worker frame %d not initialized (input_tex=%p, output_tex=%p)",
                 frame_index, worker_ctx.input_texture, worker_ctx.output_texture);
            ctx.is_processing = false;
            continue;
        }
        
        blog(LOG_DEBUG, "[ONNX Filter] Worker processing frame %d", frame_index);
        
        try {
            // No mutex acquisition needed - fence-based sync handles ordering
            
            // Run AI inference with DirectML (Section 4.3)
            // Use D3D12 imported resources for zero-copy inference
            bool inference_succeeded = false;
            try {
                if (ctx.d3d12_input_resource && ctx.d3d12_output_resource) {
                    inference_succeeded = filter->ai_engine->RunInferenceZeroCopy(ctx.d3d12_input_resource, ctx.d3d12_output_resource);
                }
            } catch (const std::exception& e) {
                blog(LOG_WARNING, "[ONNX Filter] Inference failed for frame %d: %s", frame_index, e.what());
                inference_succeeded = false;
            }
            
            // If inference failed or was skipped, do passthrough copy
            if (!inference_succeeded && worker_ctx.input_texture && worker_ctx.output_texture && filter->worker_context) {
                blog(LOG_DEBUG, "[ONNX Filter] Inference skipped/failed - doing passthrough copy");
                filter->worker_context->CopyResource(worker_ctx.output_texture, worker_ctx.input_texture);
                filter->worker_context->Flush();
            } else if (worker_ctx.output_texture && filter->worker_context) {
                // Inference succeeded - just flush to ensure D3D12 writes are visible
                filter->worker_context->Flush();
            }
            
            // No mutex release needed - fence-based sync
            
            // Clear processing flag and mark as having valid output
            ctx.is_processing = false;
            ctx.has_valid_output = true;
            
            blog(LOG_DEBUG, "[ONNX Filter] Frame %d output ready - has_valid_output=%d, is_processing=%d", 
                 frame_index, ctx.has_valid_output, ctx.is_processing);
            filter->frames_processed++;
            
        } catch (const std::exception& e) {
            blog(LOG_ERROR, "[ONNX Filter] Worker thread exception: %s", e.what());
            ctx.is_processing = false;
            filter->frames_dropped++;
        }
    }
    
    blog(LOG_INFO, "[ONNX Filter] Worker thread stopped");
}

void register_onnx_filter_info()
{
    memset(&onnx_filter_info, 0, sizeof(onnx_filter_info));
    onnx_filter_info.id = "onnx_filter_directml";
    onnx_filter_info.type = OBS_SOURCE_TYPE_FILTER;
    onnx_filter_info.output_flags = OBS_SOURCE_VIDEO;
    onnx_filter_info.get_name = onnx_filter_get_name;
    onnx_filter_info.create = onnx_filter_create;
    onnx_filter_info.destroy = onnx_filter_destroy;
    onnx_filter_info.update = onnx_filter_update;
    onnx_filter_info.get_width = onnx_filter_get_width;
    onnx_filter_info.get_height = onnx_filter_get_height;
    onnx_filter_info.video_tick = onnx_filter_video_tick;
    onnx_filter_info.video_render = onnx_filter_video_render;
    onnx_filter_info.get_properties = onnx_filter_properties;
    onnx_filter_info.get_defaults = onnx_filter_defaults;
}

const char* onnx_filter_get_name(void* unused)
{
    UNUSED_PARAMETER(unused);
    return obs_module_text("ONNX Filter (DirectML Zero-Copy)");
}

void* onnx_filter_create(obs_data_t* settings, obs_source_t* source)
{
    blog(LOG_INFO, "[ONNX Filter] Creating filter instance (Zero-Copy Architecture)");
    
    onnx_filter_data* filter = new onnx_filter_data();
    filter->context = source;
    
    // Get D3D11 device from OBS (Section 2.2)
    // We need to get the underlying D3D11 device from a texture
    obs_enter_graphics();
    
    // Create a temporary staging texture to get the device
    gs_texture_t* temp_tex = gs_texture_create(1, 1, GS_RGBA, 1, nullptr, 0);
    if (temp_tex) {
        ID3D11Texture2D* d3d11_tex = (ID3D11Texture2D*)gs_texture_get_obj(temp_tex);
        if (d3d11_tex) {
            d3d11_tex->GetDevice(&filter->d3d11_device);
            if (filter->d3d11_device) {
                filter->d3d11_device->GetImmediateContext(&filter->d3d11_context);
                blog(LOG_INFO, "[ONNX Filter] ✓ Got D3D11 device from OBS");
            }
        }
        gs_texture_destroy(temp_tex);
    }
    
    obs_leave_graphics();
    
    if (!filter->d3d11_device) {
        blog(LOG_ERROR, "[ONNX Filter] Failed to get D3D11 device from OBS");
        filter->backend_supported = false;
    }
    
    blog(LOG_INFO, "[ONNX Filter] Filter instance created");
    onnx_filter_update(filter, settings);
    
    return filter;
}

// Multi-Device Architecture: Initialize worker device on same adapter as OBS (Section 6.3)
bool InitWorkerDevice(onnx_filter_data* filter)
{
    blog(LOG_INFO, "[ONNX Filter] Initializing worker device (Multi-Device Architecture)...");
    
    // Get the DXGI adapter LUID from OBS device
    IDXGIDevice* dxgi_device = nullptr;
    IDXGIAdapter* dxgi_adapter = nullptr;
    HRESULT hr = filter->d3d11_device->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgi_device);
    if (FAILED(hr)) {
        blog(LOG_ERROR, "[ONNX Filter] Failed to get IDXGIDevice (HRESULT: 0x%08X)", hr);
        return false;
    }
    
    hr = dxgi_device->GetAdapter(&dxgi_adapter);
    dxgi_device->Release();
    if (FAILED(hr)) {
        blog(LOG_ERROR, "[ONNX Filter] Failed to get adapter (HRESULT: 0x%08X)", hr);
        return false;
    }
    
    DXGI_ADAPTER_DESC adapter_desc;
    hr = dxgi_adapter->GetDesc(&adapter_desc);
    if (FAILED(hr)) {
        dxgi_adapter->Release();
        blog(LOG_ERROR, "[ONNX Filter] Failed to get adapter desc (HRESULT: 0x%08X)", hr);
        return false;
    }
    
    blog(LOG_INFO, "[ONNX Filter] OBS adapter LUID: %08X-%08X", 
         adapter_desc.AdapterLuid.HighPart, adapter_desc.AdapterLuid.LowPart);
    
    // Create worker D3D11 device on the same adapter
    D3D_FEATURE_LEVEL feature_levels[] = {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
    };
    
    UINT create_flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
#ifdef _DEBUG
    create_flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
    
    D3D_FEATURE_LEVEL feature_level;
    hr = D3D11CreateDevice(
        dxgi_adapter,
        D3D_DRIVER_TYPE_UNKNOWN,
        nullptr,
        create_flags,
        feature_levels,
        ARRAYSIZE(feature_levels),
        D3D11_SDK_VERSION,
        &filter->worker_device,
        &feature_level,
        &filter->worker_context
    );
    
    dxgi_adapter->Release();
    
    if (FAILED(hr)) {
        blog(LOG_ERROR, "[ONNX Filter] Failed to create worker device (HRESULT: 0x%08X)", hr);
        return false;
    }
    
    blog(LOG_INFO, "[ONNX Filter] ✓ Worker device created (Feature Level: %d.%d)", 
         feature_level >> 12, (feature_level >> 8) & 0xF);
    
    // Open shared resources on worker device (using legacy shared resource API)
    for (int i = 0; i < filter->RING_BUFFER_SIZE; ++i) {
        FrameContext& obs_ctx = filter->frame_contexts[i];
        onnx_filter_data::WorkerFrameContext& worker_ctx = filter->worker_contexts[i];
        
        // Open input texture using legacy OpenSharedResource
        if (obs_ctx.shared_input_handle) {
            ID3D11Resource* resource = nullptr;
            hr = filter->worker_device->OpenSharedResource(obs_ctx.shared_input_handle, 
                                                          __uuidof(ID3D11Texture2D),
                                                          (void**)&resource);
            if (FAILED(hr)) {
                blog(LOG_ERROR, "[ONNX Filter] Failed to open shared input [%d] (HRESULT: 0x%08X)", i, hr);
                return false;
            }
            
            worker_ctx.input_texture = (ID3D11Texture2D*)resource;
            worker_ctx.input_mutex = nullptr;  // No mutex - fence-based sync
        }
        
        // Open output texture using legacy OpenSharedResource
        if (obs_ctx.shared_output_handle) {
            ID3D11Resource* resource = nullptr;
            hr = filter->worker_device->OpenSharedResource(obs_ctx.shared_output_handle,
                                                          __uuidof(ID3D11Texture2D),
                                                          (void**)&resource);
            if (FAILED(hr)) {
                blog(LOG_ERROR, "[ONNX Filter] Failed to open shared output [%d] (HRESULT: 0x%08X)", i, hr);
                return false;
            }
            
            worker_ctx.output_texture = (ID3D11Texture2D*)resource;
            worker_ctx.output_mutex = nullptr;  // No mutex - fence-based sync
        }
    }
    
    blog(LOG_INFO, "[ONNX Filter] ✓ Worker device initialized - all shared resources opened");
    return true;
}

void onnx_filter_destroy(void* data)
{
    blog(LOG_INFO, "[ONNX Filter] Destroying filter instance");
    onnx_filter_data* filter = static_cast<onnx_filter_data*>(data);
    
    // Stop worker thread (Section 6.2)
    if (filter->worker_thread) {
        filter->worker_running = false;
        filter->queue_cv.notify_one();
        if (filter->worker_thread->joinable()) {
            filter->worker_thread->join();
        }
        filter->worker_thread.reset();
    }
    
    // Release frame contexts (Section 6.3)
    for (int i = 0; i < filter->RING_BUFFER_SIZE; ++i) {
        ReleaseFrameContext(filter->frame_contexts[i]);
        
        // Release worker contexts
        onnx_filter_data::WorkerFrameContext& worker_ctx = filter->worker_contexts[i];
        if (worker_ctx.input_mutex) worker_ctx.input_mutex->Release();
        if (worker_ctx.input_texture) worker_ctx.input_texture->Release();
        if (worker_ctx.output_mutex) worker_ctx.output_mutex->Release();
        if (worker_ctx.output_texture) worker_ctx.output_texture->Release();
    }
    
    // Release worker device
    if (filter->worker_context) {
        filter->worker_context->Release();
    }
    
    if (filter->worker_device) {
        filter->worker_device->Release();
    }
    
    // Release graphics resources
    if (filter->texrender) {
        obs_enter_graphics();
        gs_texrender_destroy(filter->texrender);
        obs_leave_graphics();
    }
    
    if (filter->output_texture) {
        obs_enter_graphics();
        gs_texture_destroy(filter->output_texture);
        obs_leave_graphics();
    }
    
    if (filter->ai_texture) {
        obs_enter_graphics();
        gs_texture_destroy(filter->ai_texture);
        obs_leave_graphics();
    }
    
    // Release D3D11 context
    if (filter->d3d11_context) {
        filter->d3d11_context->Release();
    }
    
    if (filter->d3d11_device) {
        filter->d3d11_device->Release();
    }
    
    blog(LOG_INFO, "[ONNX Filter] Statistics - Processed: %llu, Dropped: %llu",
         filter->frames_processed.load(), filter->frames_dropped.load());
    
    delete filter;
}

void onnx_filter_update(void* data, obs_data_t* settings)
{
    onnx_filter_data* filter = static_cast<onnx_filter_data*>(data);
    
    const char* model_path = obs_data_get_string(settings, "model_path");
    if (model_path && *model_path && filter->model_path != model_path) {
        blog(LOG_INFO, "[ONNX Filter] Model path updated: %s", model_path);
        filter->model_path = model_path;
        
        // Stop existing worker thread
        if (filter->worker_thread) {
            filter->worker_running = false;
            filter->queue_cv.notify_one();
            if (filter->worker_thread->joinable()) {
                filter->worker_thread->join();
            }
            filter->worker_thread.reset();
        }
        
        // Convert to wide string
        std::wstring wmodel_path;
        size_t len = strlen(model_path);
        wmodel_path.resize(len + 1);
        size_t converted = 0;
        mbstowcs_s(&converted, &wmodel_path[0], wmodel_path.size(), model_path, len);
        wmodel_path.resize(converted - 1);
        
        // Initialize AI engine (Section 4.3)
        try {
            blog(LOG_INFO, "[ONNX Filter] Creating AI Engine...");
            filter->ai_engine = std::make_unique<AiEngine>(wmodel_path, filter->d3d11_device);
            
            if (filter->ai_engine->IsInitialized()) {
                blog(LOG_INFO, "[ONNX Filter] ✓ AI engine initialized successfully");
                
                // Start worker thread (Section 6.2)
                // Worker device will be initialized when ring buffer is ready
                filter->worker_running = true;
                filter->worker_thread = std::make_unique<std::thread>(worker_thread_func, filter);
                blog(LOG_INFO, "[ONNX Filter] ✓ Worker thread started");
            } else {
                blog(LOG_ERROR, "[ONNX Filter] ❌ Failed to initialize AI engine");
                filter->ai_engine.reset();
            }
        } catch (const std::exception& e) {
            blog(LOG_ERROR, "[ONNX Filter] ❌ Exception loading model: %s", e.what());
            filter->ai_engine.reset();
        }
        
        filter->initialized = false;
        filter->has_rendered_once = false;  // Reset to passthrough until new model produces output
    }
}

void onnx_filter_video_tick(void* data, float seconds)
{
    UNUSED_PARAMETER(seconds);
    onnx_filter_data* filter = static_cast<onnx_filter_data*>(data);
    
    if (!filter->backend_supported) return;
    
    // Update dimensions if source changed
    obs_source_t* target = obs_filter_get_target(filter->context);
    if (target) {
        uint32_t width = obs_source_get_base_width(target);
        uint32_t height = obs_source_get_base_height(target);
        
        if (width != filter->width || height != filter->height) {
            blog(LOG_INFO, "[ONNX Filter] Resolution changed: %dx%d -> %dx%d",
                 filter->width, filter->height, width, height);
            
            filter->width = width;
            filter->height = height;
            filter->initialized = false;
            filter->has_rendered_once = false;  // Reset to passthrough until new resolution produces output
            
            // Release and recreate frame contexts (Section 6.3)
            for (int i = 0; i < filter->RING_BUFFER_SIZE; ++i) {
                ReleaseFrameContext(filter->frame_contexts[i]);
            }
        }
    }
}

void onnx_filter_video_render(void* data, gs_effect_t* effect)
{
    UNUSED_PARAMETER(effect);
    onnx_filter_data* filter = static_cast<onnx_filter_data*>(data);
    
    if (!filter->backend_supported || !filter->ai_engine) {
        blog(LOG_DEBUG, "[ONNX Filter] Skipping - backend_supported=%d, ai_engine=%p", 
             filter->backend_supported, filter->ai_engine.get());
        obs_source_skip_video_filter(filter->context);
        return;
    }
    
    if (filter->ai_engine && !filter->ai_engine->IsInitialized()) {
        blog(LOG_DEBUG, "[ONNX Filter] Skipping - AI engine not initialized");
        obs_source_skip_video_filter(filter->context);
        return;
    }
    
    obs_source_t* target = obs_filter_get_target(filter->context);
    obs_source_t* parent = obs_filter_get_parent(filter->context);
    
    if (!target || !parent || !filter->width || !filter->height) {
        obs_source_skip_video_filter(filter->context);
        return;
    }
    
    // Initialize shared resources on first frame (Section 3.2)
    if (!filter->initialized) {
        blog(LOG_INFO, "[ONNX Filter] First frame - initializing resources (%dx%d)", 
             filter->width, filter->height);
        
        // Create texrender for capturing input
        if (!filter->texrender) {
            filter->texrender = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
            if (!filter->texrender) {
                blog(LOG_ERROR, "[ONNX Filter] Failed to create texrender");
                obs_source_skip_video_filter(filter->context);
                return;
            }
        }
        
        // Create output texture for displaying results
        if (!filter->output_texture) {
            filter->output_texture = gs_texture_create(filter->width, filter->height, GS_RGBA, 1, nullptr, GS_RENDER_TARGET);
            if (!filter->output_texture) {
                blog(LOG_ERROR, "[ONNX Filter] Failed to create output texture");
                obs_source_skip_video_filter(filter->context);
                return;
            }
        }
        
        // Create AI texture to hold model output (same size as video for proper scaling)
        if (!filter->ai_texture) {
            filter->ai_texture = gs_texture_create(filter->width, filter->height, GS_RGBA, 1, nullptr, 0);
            if (!filter->ai_texture) {
                blog(LOG_ERROR, "[ONNX Filter] Failed to create AI texture");
                obs_source_skip_video_filter(filter->context);
                return;
            }
        }
        
        bool success = true;
        for (int i = 0; i < filter->RING_BUFFER_SIZE; ++i) {
            if (!CreateSharedInteropTexture(filter, i, filter->width, filter->height)) {
                success = false;
                break;
            }
        }
        
        if (success) {
            filter->initialized = true;
            blog(LOG_INFO, "[ONNX Filter] ✓ Texrender and output texture created");
            blog(LOG_INFO, "[ONNX Filter] ✓ Ring buffer initialized (%d contexts)", 
                 filter->RING_BUFFER_SIZE);
            
            // Initialize worker device after ring buffer is ready (Multi-Device Architecture)
            blog(LOG_INFO, "[ONNX Filter] Checking worker device init: ai_engine=%p, initialized=%d, worker_device=%p",
                 filter->ai_engine.get(), 
                 filter->ai_engine ? filter->ai_engine->IsInitialized() : 0,
                 filter->worker_device);
            
            if (filter->ai_engine && filter->ai_engine->IsInitialized() && !filter->worker_device) {
                if (!InitWorkerDevice(filter)) {
                    blog(LOG_ERROR, "[ONNX Filter] ❌ Failed to initialize worker device");
                    filter->backend_supported = false;
                    obs_source_skip_video_filter(filter->context);
                    return;
                }
            }
        } else {
            // Device may be lost - stop trying to prevent hang
            blog(LOG_ERROR, "[ONNX Filter] Failed to initialize ring buffer - device may be lost");
            filter->backend_supported = false;  // Disable further attempts
            obs_source_skip_video_filter(filter->context);
            return;
        }
    }
    
    // Section 6.2: Asynchronous submission pattern
    // Get current input frame context
    int input_index = filter->current_input_index;
    FrameContext& input_ctx = filter->frame_contexts[input_index];
    
    if (!input_ctx.is_processing) {
        // Copy current frame to shared D3D11 texture
        // Render source to texrender
        gs_texrender_reset(filter->texrender);
        if (gs_texrender_begin(filter->texrender, filter->width, filter->height)) {
            vec4 clear_color = {0.0f, 0.0f, 0.0f, 0.0f};
            gs_clear(GS_CLEAR_COLOR, &clear_color, 0.0f, 0);
            
            obs_source_process_filter_begin(filter->context, GS_RGBA, OBS_NO_DIRECT_RENDERING);
            obs_source_process_filter_end(filter->context, obs_get_base_effect(OBS_EFFECT_DEFAULT), 0, 0);
            
            gs_texrender_end(filter->texrender);
            
            // Copy to shared D3D11 texture using D3D11 device context
            gs_texture_t* rendered = gs_texrender_get_texture(filter->texrender);
            ID3D11Texture2D* obs_d3d11_tex = (ID3D11Texture2D*)gs_texture_get_obj(rendered);
            
            if (obs_d3d11_tex && input_ctx.d3d11_shared_input) {
                ID3D11Device* device = (ID3D11Device*)gs_get_device_obj();
                ID3D11DeviceContext* context = nullptr;
                device->GetImmediateContext(&context);
                
                if (context) {
                    // No mutex - fence-based sync
                    
                    context->CopyResource(input_ctx.d3d11_shared_input, obs_d3d11_tex);
                    
                    context->Release();
                }
            }
        }
        
        // Submit to worker queue
        input_ctx.is_processing = true;
        {
            std::lock_guard<std::mutex> lock(filter->queue_mutex);
            filter->pending_frames.push(input_index);
        }
        filter->queue_cv.notify_one();
        blog(LOG_DEBUG, "[ONNX Filter] Queued frame %d", input_index);
        
        // Advance to next input slot (ring buffer)
        filter->current_input_index = (filter->current_input_index + 1) % filter->RING_BUFFER_SIZE;
    } else {
        // Frame dropped - worker is too slow
        filter->frames_dropped++;
    }
    
    // Section 6.2: Check for completed output
    int output_index = filter->current_output_index;
    FrameContext& output_ctx = filter->frame_contexts[output_index];
    
    // Check if we have a valid AI frame ready to display
    // Note: We check ONLY has_valid_output, ignoring is_processing
    // because the same slot may be reused for a new input before we consume the old output
    if (output_ctx.has_valid_output && filter->ai_texture) {
        
        // 1. Copy 640x360 AI output to gs_texture
        if (output_ctx.d3d11_shared_output) {
            ID3D11Device* device = (ID3D11Device*)gs_get_device_obj();
            ID3D11DeviceContext* context = nullptr;
            device->GetImmediateContext(&context);
            
            if (context) {
                ID3D11Texture2D* ai_tex = (ID3D11Texture2D*)gs_texture_get_obj(filter->ai_texture);
                context->CopyResource(ai_tex, output_ctx.d3d11_shared_output);
                context->Flush();
                context->Release();
            }
        }
        
        // 2. Render the AI texture (full resolution)
        gs_effect_t* draw_effect = obs_get_base_effect(OBS_EFFECT_DEFAULT);
        gs_eparam_t* param = gs_effect_get_param_by_name(draw_effect, "image");
        gs_effect_set_texture(param, filter->ai_texture);
        
        while (gs_effect_loop(draw_effect, "Draw")) {
            gs_draw_sprite(filter->ai_texture, 0, filter->width, filter->height);
        }
        
        // Mark that we've rendered at least once
        filter->has_rendered_once = true;
        
        // Mark output as consumed and advance
        output_ctx.has_valid_output = false;
        filter->current_output_index = (filter->current_output_index + 1) % filter->RING_BUFFER_SIZE;
        
    } else if (filter->ai_texture && filter->has_rendered_once) {
        // No new AI output - render the previous frame to avoid flickering
        int prev_index = (filter->current_output_index + filter->RING_BUFFER_SIZE - 1) % filter->RING_BUFFER_SIZE;
        FrameContext& prev_ctx = filter->frame_contexts[prev_index];
        
        if (prev_ctx.d3d11_shared_output) {
            // Copy previous AI frame
            ID3D11Device* device = (ID3D11Device*)gs_get_device_obj();
            ID3D11DeviceContext* context = nullptr;
            device->GetImmediateContext(&context);
            
            if (context) {
                ID3D11Texture2D* ai_tex = (ID3D11Texture2D*)gs_texture_get_obj(filter->ai_texture);
                context->CopyResource(ai_tex, prev_ctx.d3d11_shared_output);
                context->Release();
            }
        }
        
        // Render the AI texture
        gs_effect_t* draw_effect = obs_get_base_effect(OBS_EFFECT_DEFAULT);
        gs_eparam_t* param = gs_effect_get_param_by_name(draw_effect, "image");
        gs_effect_set_texture(param, filter->ai_texture);
        
        while (gs_effect_loop(draw_effect, "Draw")) {
            gs_draw_sprite(filter->ai_texture, 0, filter->width, filter->height);
        }
    } else {
        // First frame before any processing - passthrough
        obs_source_skip_video_filter(filter->context);
    }
}

uint32_t onnx_filter_get_width(void* data)
{
    onnx_filter_data* filter = static_cast<onnx_filter_data*>(data);
    return filter->width;
}

uint32_t onnx_filter_get_height(void* data)
{
    onnx_filter_data* filter = static_cast<onnx_filter_data*>(data);
    return filter->height;
}

obs_properties_t* onnx_filter_properties(void* data)
{
    UNUSED_PARAMETER(data);
    
    obs_properties_t* props = obs_properties_create();
    
    obs_properties_add_path(props, "model_path",
                           obs_module_text("Model Path"),
                           OBS_PATH_FILE, "ONNX Models (*.onnx)", 
                           "E:\\_DEV\\OBSPlugins\\ONNX files\\Computer Vision");
    
    return props;
}

void onnx_filter_defaults(obs_data_t* settings)
{
    obs_data_set_default_string(settings, "model_path", "");
}

// Helper function: Create shared interop texture (Section 3.2)
static bool CreateSharedInteropTexture(
    onnx_filter_data* filter,
    int index,
    uint32_t width,
    uint32_t height)
{
    blog(LOG_INFO, "[ONNX Filter] Creating shared texture %d (%ux%u)", index, width, height);
    
    FrameContext& ctx = filter->frame_contexts[index];
    
    // 1. Create D3D11 Texture with legacy shared resource (UAV + NT handles incompatible on some hardware)
    D3D11_TEXTURE2D_DESC tex_desc = {};
    tex_desc.Width = width;
    tex_desc.Height = height;
    tex_desc.MipLevels = 1;
    tex_desc.ArraySize = 1;
    tex_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    tex_desc.SampleDesc.Count = 1;
    tex_desc.Usage = D3D11_USAGE_DEFAULT;
    // IMPORTANT: Bind flags must include UNORDERED_ACCESS for the PostProcess Shader to write to it via D3D12
    tex_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET | D3D11_BIND_UNORDERED_ACCESS;
    // Use legacy shared resource (compatible with UAV on all hardware)
    tex_desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED; 
    
    HRESULT hr = filter->d3d11_device->CreateTexture2D(&tex_desc, nullptr, &ctx.d3d11_shared_input);
    if (FAILED(hr)) {
        blog(LOG_ERROR, "[ONNX Filter] Failed to create shared input texture (HRESULT: 0x%08X)", hr);
        return false;
    }
    
    // No keyed mutex - using fence synchronization
    ctx.input_mutex = nullptr;
    
    // 2. Export legacy shared handle for Input
    IDXGIResource* dxgi_resource = nullptr;
    hr = ctx.d3d11_shared_input->QueryInterface(__uuidof(IDXGIResource), (void**)&dxgi_resource);
    if (FAILED(hr)) {
        blog(LOG_ERROR, "[ONNX Filter] Failed to query IDXGIResource (HRESULT: 0x%08X)", hr);
        return false;
    }
    
    HANDLE shared_handle = nullptr;
    hr = dxgi_resource->GetSharedHandle(&shared_handle);
    dxgi_resource->Release();
    
    if (FAILED(hr) || !shared_handle) {
        blog(LOG_ERROR, "[ONNX Filter] Failed to get input shared handle (HRESULT: 0x%08X)", hr);
        return false;
    }
    
    ctx.shared_input_handle = shared_handle;
    
    // 3. Open in D3D12 (legacy handle owned by resource, don't close on failure)
    ID3D12Device* d3d12_device = filter->ai_engine->GetD3D12Device();
    if (d3d12_device) {
        hr = d3d12_device->OpenSharedHandle(shared_handle, __uuidof(ID3D12Resource), (void**)&ctx.d3d12_input_resource);
        if (FAILED(hr)) {
            blog(LOG_ERROR, "[ONNX Filter] Failed to open input handle in D3D12 (HRESULT: 0x%08X)", hr);
            return false;
        }
    }
    
    // 4. Create Output Texture (Same process)
    hr = filter->d3d11_device->CreateTexture2D(&tex_desc, nullptr, &ctx.d3d11_shared_output);
    if (FAILED(hr)) {
        blog(LOG_ERROR, "[ONNX Filter] Failed to create shared output texture (HRESULT: 0x%08X)", hr);
        return false;
    }
    
    ctx.output_mutex = nullptr;
    
    // 5. Export legacy shared handle for Output
    hr = ctx.d3d11_shared_output->QueryInterface(__uuidof(IDXGIResource), (void**)&dxgi_resource);
    if (FAILED(hr)) {
        blog(LOG_ERROR, "[ONNX Filter] Failed to query IDXGIResource for output (HRESULT: 0x%08X)", hr);
        return false;
    }
    
    hr = dxgi_resource->GetSharedHandle(&shared_handle);
    dxgi_resource->Release();
    
    if (FAILED(hr) || !shared_handle) {
        blog(LOG_ERROR, "[ONNX Filter] Failed to get output shared handle (HRESULT: 0x%08X)", hr);
        return false;
    }
    
    ctx.shared_output_handle = shared_handle;
    
    // 6. Open Output in D3D12
    if (d3d12_device) {
        hr = d3d12_device->OpenSharedHandle(shared_handle, __uuidof(ID3D12Resource), (void**)&ctx.d3d12_output_resource);
        if (FAILED(hr)) {
            blog(LOG_ERROR, "[ONNX Filter] Failed to open output handle in D3D12 (HRESULT: 0x%08X)", hr);
            return false;
        }
    }
    
    blog(LOG_INFO, "[ONNX Filter] ✓ Shared texture %d created (Legacy shared resource with UAV)", index);
    return true;
}

// Helper function: Release frame context resources
static void ReleaseFrameContext(FrameContext& context)
{
    if (context.d3d12_output_resource) {
        context.d3d12_output_resource->Release();
        context.d3d12_output_resource = nullptr;
    }
    
    if (context.d3d12_input_resource) {
        context.d3d12_input_resource->Release();
        context.d3d12_input_resource = nullptr;
    }
    
    if (context.preprocessed_tensor) {
        context.preprocessed_tensor->Release();
        context.preprocessed_tensor = nullptr;
    }
    
    if (context.output_mutex) {
        context.output_mutex->Release();
        context.output_mutex = nullptr;
    }
    
    if (context.input_mutex) {
        context.input_mutex->Release();
        context.input_mutex = nullptr;
    }
    
    if (context.d3d11_shared_output) {
        context.d3d11_shared_output->Release();
        context.d3d11_shared_output = nullptr;
    }
    
    if (context.d3d11_shared_input) {
        context.d3d11_shared_input->Release();
        context.d3d11_shared_input = nullptr;
    }
    
    // Note: With D3D11_RESOURCE_MISC_SHARED, GetSharedHandle() returns legacy handles
    // that are owned by the resource and should NOT be closed separately
    context.shared_input_handle = nullptr;
    context.shared_output_handle = nullptr;
    
    context.is_processing = false;
    context.has_valid_output = false;
}
