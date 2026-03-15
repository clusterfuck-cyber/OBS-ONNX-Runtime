/*
 * OBS ONNX Filter Source - DirectML Implementation (Minimal Pass-Through)
 * This is a temporary implementation until proper DirectML texture handling is added
 */

#include "filter-source.hpp"
#include "ai-engine.hpp"
#include <util/platform.h>

// Define the source info structure
struct obs_source_info onnx_filter_info;

void register_onnx_filter_info() {
    memset(&onnx_filter_info, 0, sizeof(onnx_filter_info));
    onnx_filter_info.id = "onnx_filter";
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

const char *onnx_filter_get_name(void *unused)
{
    UNUSED_PARAMETER(unused);
    return obs_module_text("ONNXFilter");
}

void *onnx_filter_create(obs_data_t *settings, obs_source_t *source)
{
    blog(LOG_INFO, "[ONNX Filter] Creating filter instance (DirectML mode)");
    
    onnx_filter_data *filter = new onnx_filter_data();
    filter->context = source;
    filter->width = 0;
    filter->height = 0;
    filter->initialized = false;
    filter->backend_supported = true; // DirectML always supported on Windows
    
    filter->texrender = nullptr;
    filter->output_texture = nullptr;
    filter->effect = nullptr;
    
    // DirectML: No CUDA resources
    filter->d3d11_input_tex = nullptr;
    filter->d3d11_output_tex = nullptr;
    filter->cuda_input_res = nullptr;
    filter->cuda_output_res = nullptr;
    filter->cuda_tensor_input_buffer = nullptr;
    filter->cuda_tensor_output_buffer = nullptr;
    filter->stream = 0;
    
    filter->ai_engine = nullptr;
    
    blog(LOG_INFO, "[ONNX Filter] Filter instance created");
    
    onnx_filter_update(filter, settings);
    
    return filter;
}

void onnx_filter_destroy(void *data)
{
    blog(LOG_INFO, "[ONNX Filter] Destroying filter instance");
    onnx_filter_data *filter = static_cast<onnx_filter_data*>(data);
    
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
    
    // DirectML: No CUDA cleanup needed
    
    delete filter;
}

void onnx_filter_update(void *data, obs_data_t *settings)
{
    onnx_filter_data *filter = static_cast<onnx_filter_data*>(data);
    
    const char *model_path = obs_data_get_string(settings, "model_path");
    if (model_path && *model_path) {
        blog(LOG_INFO, "[ONNX Filter] Model path: %s", model_path);
        
        filter->initialized = false;
        
        // Convert to wide string for ONNX Runtime
        std::wstring wmodel_path;
        size_t len = strlen(model_path);
        wmodel_path.resize(len);
        size_t converted = 0;
        mbstowcs_s(&converted, &wmodel_path[0], wmodel_path.size() + 1, model_path, len);
        
        // Initialize AI engine
        blog(LOG_INFO, "[ONNX Filter] === Initializing AI Engine ===");
        blog(LOG_INFO, "[ONNX Filter] Model path (wide): %ls", wmodel_path.c_str());
        try {
            blog(LOG_INFO, "[ONNX Filter] Creating AiEngine instance...");
            filter->ai_engine = std::make_unique<AiEngine>(wmodel_path);
            
            if (filter->ai_engine->IsInitialized()) {
                blog(LOG_INFO, "[ONNX Filter] ✓ AI engine initialized successfully with DirectML");
                auto input_shape = filter->ai_engine->GetInputShape();
                auto output_shape = filter->ai_engine->GetOutputShape();
                blog(LOG_INFO, "[ONNX Filter] Input shape: [%lld,%lld,%lld,%lld]",
                     input_shape[0], input_shape[1], input_shape[2], input_shape[3]);
                blog(LOG_INFO, "[ONNX Filter] Output shape: [%lld,%lld,%lld,%lld]",
                     output_shape[0], output_shape[1], output_shape[2], output_shape[3]);
            } else {
                blog(LOG_ERROR, "[ONNX Filter] ❌ Failed to initialize AI engine");
                filter->ai_engine.reset();
            }
        } catch (const std::exception &e) {
            blog(LOG_ERROR, "[ONNX Filter] ❌ Exception loading model: %s", e.what());
            blog(LOG_ERROR, "[ONNX Filter] Exception type: %s", typeid(e).name());
            filter->ai_engine.reset();
        } catch (...) {
            blog(LOG_ERROR, "[ONNX Filter] ❌ Unknown exception loading model");
            filter->ai_engine.reset();
        }
    }
}

void onnx_filter_video_tick(void *data, float seconds)
{
    UNUSED_PARAMETER(seconds);
    onnx_filter_data *filter = static_cast<onnx_filter_data*>(data);
    
    if (!filter->backend_supported) return;
    
    // Update dimensions if source changed
    obs_source_t *target = obs_filter_get_target(filter->context);
    if (target) {
        uint32_t width = obs_source_get_base_width(target);
        uint32_t height = obs_source_get_base_height(target);
        
        if (width != filter->width || height != filter->height) {
            blog(LOG_INFO, "[ONNX Filter] Resolution changed: %dx%d -> %dx%d",
                 filter->width, filter->height, width, height);
            
            filter->width = width;
            filter->height = height;
            filter->initialized = false;
        }
    }
}

void onnx_filter_video_render(void *data, gs_effect_t *effect)
{
    UNUSED_PARAMETER(effect);
    onnx_filter_data *filter = static_cast<onnx_filter_data*>(data);
    
    if (!filter->backend_supported) {
        obs_source_skip_video_filter(filter->context);
        return;
    }
    
    obs_source_t *target = obs_filter_get_target(filter->context);
    obs_source_t *parent = obs_filter_get_parent(filter->context);
    
    if (!target || !parent || !filter->width || !filter->height) {
        obs_source_skip_video_filter(filter->context);
        return;
    }
    
    // DirectML: Pass-through for now (GPU interop not implemented yet)
    // TODO: Implement DirectML texture handling:
    //  1. Copy D3D11 texture to system memory
    //  2. Preprocess and pass to ONNX Runtime
    //  3. DirectML runs inference internally
    //  4. Copy result back to D3D11 texture
    obs_source_process_filter_begin(filter->context, GS_RGBA, OBS_ALLOW_DIRECT_RENDERING);
    obs_source_process_filter_end(filter->context, obs_get_base_effect(OBS_EFFECT_DEFAULT), 0, 0);
}

uint32_t onnx_filter_get_width(void *data)
{
    onnx_filter_data *filter = static_cast<onnx_filter_data*>(data);
    return filter->width;
}

uint32_t onnx_filter_get_height(void *data)
{
    onnx_filter_data *filter = static_cast<onnx_filter_data*>(data);
    return filter->height;
}

obs_properties_t *onnx_filter_properties(void *data)
{
    UNUSED_PARAMETER(data);
    
    obs_properties_t *props = obs_properties_create();
    
    obs_properties_add_path(props, "model_path",
                           obs_module_text("ModelPath"),
                           OBS_PATH_FILE, "ONNX Models (*.onnx)", 
                           "E:\\_DEV\\OBSPlugins\\ONNX files\\Computer Vision");
    
    return props;
}

void onnx_filter_defaults(obs_data_t *settings)
{
    obs_data_set_default_string(settings, "model_path", "");
}
