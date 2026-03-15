cmake_minimum_required(VERSION 3.24)

# Option to enable CUDA support (OFF by default for DirectML-only builds)
option(USE_CUDA "Enable CUDA support for GPU interop" OFF)

if(USE_CUDA)
    project(obs-onnx-plugin VERSION 0.1.0 LANGUAGES C CXX CUDA)
else()
    project(obs-onnx-plugin VERSION 0.1.0 LANGUAGES C CXX)
endif()

# C++ Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA Standards (only if CUDA is enabled)
if(USE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # CUDA Architectures: Target architectures supported by CUDA 13.x
    # Turing (RTX 20-series), Ampere (RTX 30-series), Ada Lovelace (RTX 40-series)
    set(CMAKE_CUDA_ARCHITECTURES "75;86;89")
endif()

# Build configuration
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Output Directories ---
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# --- Dependencies: OBS Studio (LibObs) ---
set(OBS_SOURCE_DIR "${CMAKE_SOURCE_DIR}/../obs-studio" CACHE PATH "Path to OBS Studio source")
set(OBS_BUILD_DIR "${OBS_SOURCE_DIR}/build64" CACHE PATH "Path to OBS Studio build directory")
set(LIBOBS_INCLUDE_DIR "${OBS_SOURCE_DIR}/libobs" CACHE PATH "Path to libobs include directory")
set(LIBOBS_LIB_DIR "${OBS_BUILD_DIR}/libobs/Release" CACHE PATH "Path to libobs library directory")

if(NOT EXISTS "${LIBOBS_INCLUDE_DIR}")
    message(FATAL_ERROR "LibObs include directory not found at: ${LIBOBS_INCLUDE_DIR}
Please set OBS_SOURCE_DIR to point to your obs-studio source directory")
endif()

if(NOT EXISTS "${OBS_BUILD_DIR}/config/obsconfig.h")
    message(FATAL_ERROR "OBS build configuration not found at: ${OBS_BUILD_DIR}/config/obsconfig.h
Please build OBS Studio first: cmake --build ${OBS_BUILD_DIR} --config Release --target libobs")
endif()

# Create imported target for libobs
add_library(libobs SHARED IMPORTED)
set_target_properties(libobs PROPERTIES
    IMPORTED_LOCATION "${LIBOBS_LIB_DIR}/obs.dll"
    IMPORTED_IMPLIB "${LIBOBS_LIB_DIR}/obs.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${LIBOBS_INCLUDE_DIR};${OBS_BUILD_DIR}/config"
)

# --- Dependencies: ONNX Runtime with DirectML ---
set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/../onnxruntime-win-x64-directml-1.19.0" CACHE PATH "Path to ONNX Runtime DirectML directory")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_DIR}/lib" CACHE PATH "Path to ONNX Runtime libraries")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include" CACHE PATH "Path to ONNX Runtime includes")

# --- Dependencies: DirectML SDK ---
set(DIRECTML_DIR "${CMAKE_SOURCE_DIR}/../DirectML-1.15.1" CACHE PATH "Path to DirectML SDK directory")
set(DIRECTML_LIB_DIR "${DIRECTML_DIR}/bin/x64-win" CACHE PATH "Path to DirectML libraries")
set(DIRECTML_INCLUDE_DIR "${DIRECTML_DIR}/include" CACHE PATH "Path to DirectML includes")

if(NOT EXISTS "${ONNXRUNTIME_LIB_DIR}/onnxruntime.lib")
    message(FATAL_ERROR "ONNX Runtime library not found: ${ONNXRUNTIME_LIB_DIR}/onnxruntime.lib
Please extract the ONNX Runtime DirectML package to: ${ONNXRUNTIME_DIR}")
endif()

# Create imported target for ONNX Runtime
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIB_DIR}/onnxruntime.dll"
    IMPORTED_IMPLIB "${ONNXRUNTIME_LIB_DIR}/onnxruntime.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
)

# --- Dependencies: CUDA Toolkit (optional) ---
if(USE_CUDA)
    find_package(CUDAToolkit REQUIRED)

    if(NOT CUDAToolkit_FOUND)
        message(FATAL_ERROR "CUDA Toolkit not found. Please install CUDA 11.x or 12.x")
    endif()

    message(STATUS "Found CUDA ${CUDAToolkit_VERSION} at ${CUDAToolkit_BIN_DIR}")
else()
    message(STATUS "CUDA support disabled - using DirectML only")
endif()

# --- Plugin Library Definition ---
set(PLUGIN_SOURCES
    src/plugin-main.cpp
    src/filter-source-advanced.cpp
    src/ai-engine.cpp
)

set(PLUGIN_HEADERS
    src/filter-source.hpp
    src/ai-engine.hpp
)

# HLSL Compute Shaders (Section 5.2)
set(HLSL_SHADERS
    data/shaders/rgba_to_nchw.hlsl
    data/shaders/nchw_to_rgba.hlsl
)

# Add CUDA sources only if CUDA is enabled
if(USE_CUDA)
    list(APPEND PLUGIN_SOURCES src/gpu-interop.cu)
    list(APPEND PLUGIN_HEADERS src/gpu-interop.cuh)
endif()

# Create the plugin as a MODULE (DLL loaded at runtime)
add_library(${PROJECT_NAME} MODULE
    ${PLUGIN_SOURCES}
    ${PLUGIN_HEADERS}
)

# --- Compile HLSL Shaders ---
# Find FXC (always available on Windows SDK)
find_program(FXC_EXECUTABLE fxc.exe 
    HINTS 
    "C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64"
    "C:/Program Files (x86)/Windows Kits/10/bin/10.0.22621.0/x64"
    "C:/Program Files (x86)/Windows Kits/10/bin/10.0.22000.0/x64"
    "C:/Program Files (x86)/Windows Kits/10/bin/x64"
)

if(FXC_EXECUTABLE)
    message(STATUS "Found FXC: ${FXC_EXECUTABLE}")
    
    # Define shader files to compile
    set(SHADER_FILES
        "data/shaders/preprocess_nchw.hlsl"
        "data/shaders/postprocess_rgba.hlsl"
    )
    
    # Generate shader bytecode header file
    set(SHADER_HEADER_FILE "${CMAKE_BINARY_DIR}/generated/shader_bytecode.h")
    file(WRITE ${SHADER_HEADER_FILE} "// Auto-generated shader bytecode\n#pragma once\n#include <cstdint>\n\n")
    
    foreach(HLSL_FILE ${SHADER_FILES})
        get_filename_component(HLSL_NAME ${HLSL_FILE} NAME_WE)
        set(CSO_FILE "${CMAKE_BINARY_DIR}/shaders/${HLSL_NAME}.cso")
        
        # Compile HLSL to bytecode
        add_custom_command(
            OUTPUT ${CSO_FILE}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/shaders"
            COMMAND ${FXC_EXECUTABLE} /T cs_5_0 /E CSMain /O3 /Fo ${CSO_FILE} ${CMAKE_SOURCE_DIR}/${HLSL_FILE}
            DEPENDS ${CMAKE_SOURCE_DIR}/${HLSL_FILE}
            COMMENT "Compiling HLSL shader: ${HLSL_FILE}"
        )
        
        # Convert bytecode to C++ byte array
        set(HEADER_FILE "${CMAKE_BINARY_DIR}/generated/${HLSL_NAME}_bytecode.h")
        add_custom_command(
            OUTPUT ${HEADER_FILE}
            COMMAND ${CMAKE_COMMAND} 
                -DINPUT_FILE=${CSO_FILE}
                -DOUTPUT_FILE=${HEADER_FILE}
                -DARRAY_NAME=${HLSL_NAME}_bytecode
                -P ${CMAKE_SOURCE_DIR}/cmake/bin2header.cmake
            DEPENDS ${CSO_FILE}
            COMMENT "Generating bytecode header: ${HLSL_NAME}_bytecode.h"
        )
        
        list(APPEND COMPILED_SHADERS ${CSO_FILE})
        list(APPEND SHADER_HEADERS ${HEADER_FILE})
    endforeach()
    
    add_custom_target(CompileShaders ALL DEPENDS ${COMPILED_SHADERS} ${SHADER_HEADERS})
    add_dependencies(${PROJECT_NAME} CompileShaders)
    
    # Add generated directory to include path
    target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_BINARY_DIR}/generated)
    
else()
    message(FATAL_ERROR "FXC not found - cannot compile HLSL shaders. Please install Windows SDK.")
endif()

# --- Include Directories ---
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${LIBOBS_INCLUDE_DIR}
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${DIRECTML_INCLUDE_DIR}
)

if(USE_CUDA)
    target_include_directories(${PROJECT_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()

# --- Link Libraries ---
target_link_libraries(${PROJECT_NAME} PRIVATE
    libobs
    onnxruntime
    d3d11
    d3d12
    dxgi
    ${DIRECTML_LIB_DIR}/DirectML.lib
)

# Delay-load onnxruntime.dll to control DLL search path and avoid System32 DirectML.dll shadowing
if(MSVC)
    target_link_options(${PROJECT_NAME} PRIVATE "/DELAYLOAD:onnxruntime.dll")
    # Requires delayimp.lib to handle delay loading
    target_link_libraries(${PROJECT_NAME} PRIVATE delayimp)
endif()

# --- Compiler Definitions ---
target_compile_definitions(${PROJECT_NAME} PRIVATE
    UNICODE
    _UNICODE
    WIN32_LEAN_AND_MEAN
    NOMINMAX
)

# --- CUDA Compilation Options (only if CUDA is enabled) ---
if(USE_CUDA)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

# --- Output Name ---
set_target_properties(${PROJECT_NAME} PROPERTIES
    PREFIX ""
    OUTPUT_NAME "obs-onnx-plugin"
)

# --- Installation Rules ---
install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION "obs-plugins/64bit"
    RUNTIME DESTINATION "obs-plugins/64bit"
)

install(DIRECTORY data/
    DESTINATION "data/obs-plugins/${PROJECT_NAME}"
)

# --- Summary ---
message(STATUS "=== OBS ONNX Plugin Configuration ===")
message(STATUS "Project: ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "OBS Source: ${OBS_SOURCE_DIR}")
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_DIR}")
if(USE_CUDA)
    message(STATUS "CUDA: ${CUDAToolkit_VERSION} (ENABLED)")
else()
    message(STATUS "CUDA: Disabled (DirectML-only build)")
endif()
message(STATUS "Plugin Output: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
message(STATUS "=====================================")
