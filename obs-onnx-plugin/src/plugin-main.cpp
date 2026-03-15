/*
 * OBS ONNX Runtime Plugin - Main Entry Point
 * Provides AI-powered video filtering using ONNX Runtime with DirectML acceleration
 */

#include <obs-module.h>
#include <util/platform.h>
#include <Windows.h>

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("obs-onnx-plugin", "en-US")

// Forward declarations
extern struct obs_source_info onnx_filter_info;
extern void register_onnx_filter_info();

bool obs_module_load(void)
{
    blog(LOG_INFO, "[ONNX Plugin] Loading ONNX Runtime plugin v0.1.0");
    blog(LOG_INFO, "[ONNX Plugin] Using DirectML execution provider (Windows GPU)");
    
    // === CRITICAL: Fix System32 DirectML.dll shadowing ===
    // Windows loads System32 DirectML.dll instead of our bundled v1.15.1
    // Solution: Set DLL search path and pre-load DirectML before onnxruntime.dll loads
    
    // 1. Get plugin directory
    char plugin_path[MAX_PATH];
    HMODULE hModule = NULL;
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | 
                       GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       (LPCSTR)&obs_module_load, &hModule);
    
    if (GetModuleFileNameA(hModule, plugin_path, MAX_PATH)) {
        // Strip filename to get directory
        char* last_slash = strrchr(plugin_path, '\\');
        if (last_slash) *last_slash = '\0';
        
        blog(LOG_INFO, "[ONNX Plugin] Plugin directory: %s", plugin_path);
        
        // 2. Set DLL search directory (prepends to search path)
        SetDllDirectoryA(plugin_path);
        blog(LOG_INFO, "[ONNX Plugin] Set DLL search path to plugin directory");
        
        // 2.5 Pre-load D3D12.dll (required by DirectML)
        blog(LOG_INFO, "[ONNX Plugin] === Pre-loading D3D12.dll (DirectML dependency) ===");
        HMODULE hD3D12 = LoadLibraryA("d3d12.dll");
        if (hD3D12) {
            char d3d12_path[MAX_PATH];
            GetModuleFileNameA(hD3D12, d3d12_path, MAX_PATH);
            blog(LOG_INFO, "[ONNX Plugin] ✓ Pre-loaded d3d12.dll");
            blog(LOG_INFO, "[ONNX Plugin]   Loaded from: %s", d3d12_path);
        } else {
            DWORD error = GetLastError();
            blog(LOG_ERROR, "[ONNX Plugin] ✗ Failed to pre-load d3d12.dll (Error %lu)", error);
            blog(LOG_ERROR, "[ONNX Plugin] This is required by DirectML");
            return false;
        }
        
        // 3. Pre-load DirectML.dll with FULL PATH using LoadLibraryEx
        // CRITICAL: Use LOAD_WITH_ALTERED_SEARCH_PATH to force our bundled v1.15.1
        // System has DirectML v1.15.5 in System32 which may be incompatible
        char dml_path[MAX_PATH];
        snprintf(dml_path, MAX_PATH, "%s\\DirectML.dll", plugin_path);
        
        blog(LOG_INFO, "[ONNX Plugin] === Pre-loading DirectML.dll ===");
        blog(LOG_INFO, "[ONNX Plugin] Full path: %s", dml_path);
        blog(LOG_INFO, "[ONNX Plugin] Using LoadLibraryExA with LOAD_WITH_ALTERED_SEARCH_PATH");
        
        // Use LoadLibraryEx with LOAD_WITH_ALTERED_SEARCH_PATH to bypass System32
        HMODULE hDml = LoadLibraryExA(dml_path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (hDml) {
            // Verify it's our version, not System32
            char loaded_path[MAX_PATH];
            GetModuleFileNameA(hDml, loaded_path, MAX_PATH);
            blog(LOG_INFO, "[ONNX Plugin] ✓ Pre-loaded bundled DirectML.dll (v1.15.1)");
            blog(LOG_INFO, "[ONNX Plugin]   Loaded from: %s", loaded_path);
            
            // Check if wrong version loaded
            if (strstr(loaded_path, "System32") || strstr(loaded_path, "SysWOW64")) {
                blog(LOG_ERROR, "[ONNX Plugin] ❌ WRONG DLL! System32 DirectML loaded despite explicit path!");
                blog(LOG_ERROR, "[ONNX Plugin] This will cause version mismatch crashes!");
                // Continue anyway to show better error message
            } else if (!strstr(loaded_path, "obs-plugins") && !strstr(loaded_path, plugin_path)) {
                blog(LOG_WARNING, "[ONNX Plugin] ⚠️ DirectML loaded from unexpected location");
            }
        } else {
            DWORD error = GetLastError();
            blog(LOG_ERROR, "[ONNX Plugin] ✗ Failed to pre-load DirectML.dll (Error %lu)", error);
            blog(LOG_ERROR, "[ONNX Plugin]   Path: %s", dml_path);
            
            // Translate common error codes
            switch (error) {
                case 126: blog(LOG_ERROR, "[ONNX Plugin]   ERROR_MOD_NOT_FOUND: Module not found"); break;
                case 127: blog(LOG_ERROR, "[ONNX Plugin]   ERROR_PROC_NOT_FOUND: Procedure not found"); break;
                case 998: blog(LOG_ERROR, "[ONNX Plugin]   ERROR_NOACCESS: Invalid access to memory"); break;
                case 5:   blog(LOG_ERROR, "[ONNX Plugin]   ERROR_ACCESS_DENIED: Access denied"); break;
                default:  blog(LOG_ERROR, "[ONNX Plugin]   Unknown error code"); break;
            }
            return false;
        }
        
        // 4. Pre-load onnxruntime.dll to ensure it uses our DirectML
        char ort_path[MAX_PATH];
        snprintf(ort_path, MAX_PATH, "%s\\onnxruntime.dll", plugin_path);
        
        // Also use LoadLibraryEx for onnxruntime to ensure consistent loading
        HMODULE hOrt = LoadLibraryExA(ort_path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (hOrt) {
            char loaded_ort_path[MAX_PATH];
            GetModuleFileNameA(hOrt, loaded_ort_path, MAX_PATH);
            blog(LOG_INFO, "[ONNX Plugin] ✓ Pre-loaded onnxruntime.dll (DirectML variant)");
            blog(LOG_INFO, "[ONNX Plugin]   Loaded from: %s", loaded_ort_path);
        } else {
            DWORD error = GetLastError();
            blog(LOG_ERROR, "[ONNX Plugin] ✗ Failed to pre-load onnxruntime.dll (Error %lu)", error);
            blog(LOG_ERROR, "[ONNX Plugin]   Path: %s", ort_path);
            
            // Translate common error codes
            switch (error) {
                case 126: blog(LOG_ERROR, "[ONNX Plugin]   ERROR_MOD_NOT_FOUND: Module not found (check dependencies)"); break;
                case 127: blog(LOG_ERROR, "[ONNX Plugin]   ERROR_PROC_NOT_FOUND: Procedure not found"); break;
                case 998: blog(LOG_ERROR, "[ONNX Plugin]   ERROR_NOACCESS: Invalid access to memory"); break;
                default:  blog(LOG_ERROR, "[ONNX Plugin]   Unknown error code"); break;
            }
            return false;
        }
    }
    
    blog(LOG_INFO, "[ONNX Plugin] DirectML works on NVIDIA, AMD, and Intel GPUs");
    
    // Initialize and register the filter source
    register_onnx_filter_info();
    obs_register_source(&onnx_filter_info);
    
    blog(LOG_INFO, "[ONNX Plugin] Plugin loaded successfully");
    return true;
}

void obs_module_unload(void)
{
    blog(LOG_INFO, "[ONNX Plugin] Unloading ONNX Runtime plugin");
}

MODULE_EXPORT const char *obs_module_description(void)
{
    return "AI-powered video filter using ONNX Runtime with DirectML acceleration";
}

MODULE_EXPORT const char *obs_module_name(void)
{
    return "OBS ONNX Runtime Plugin (DirectML)";
}
