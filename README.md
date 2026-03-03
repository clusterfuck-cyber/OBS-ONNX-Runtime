# OBS ONNX Plugin

This directory contains the runtime binaries required to deploy the **OBS ONNX Plugin** (DirectML build).

## What Does This Plugin Do?

In plain terms: it lets OBS apply **AI-powered visual effects to your video in real time**, entirely on your GPU — no cloud, no external service.

You add it as a **filter** on any video source in OBS (webcam, game capture, etc.). It then runs an AI model (provided as an `.onnx` file) on every frame of that video as it comes in. Depending on the model you load, the plugin can do things like:

- **Style transfer** – make your footage look like a painting or artwork
- **Background removal / segmentation** – cut out a person without a green screen
- **Image enhancement** – upscale, denoise, or sharpen video on the fly
- **Custom effects** – anything expressible as an ONNX model

Under the hood, each video frame is resized and converted into the format the AI model expects, the model runs on the GPU (via Microsoft's DirectML layer, which works on NVIDIA, AMD, and Intel cards), and the result is converted back into a video frame that OBS displays or records.

Because all processing happens on the GPU using DirectML, it works on any modern Windows PC without needing CUDA or a specific brand of graphics card.

## Contents

| File | Version | Description |
|------|---------|-------------|
| `obs-onnx-plugin.dll` | 0.1.0 (RelWithDebInfo) | OBS plugin – AI inference filter |
| `onnxruntime.dll` | 1.19.0 | ONNX Runtime with DirectML execution provider |
| `DirectML.dll` | 1.15.1 | Microsoft DirectML – GPU-accelerated ML primitives |

## Installation

Copy all three DLLs to the OBS Studio plugin directory:

```
C:\Program Files\obs-studio\obs-plugins\64bit\
```

The easiest way is to run the provided installer script from the project root:

```powershell
.\install_latest_build.ps1
```

## Requirements

| Requirement | Minimum |
|-------------|---------|
| Windows | 10 version 1903 (build 18362) or later |
| OBS Studio | 30.0+ |
| GPU | DirectX 12 capable (NVIDIA, AMD, or Intel) |
| VRAM | 2 GB recommended |
| Driver | Latest GPU driver recommended |

## Build Information

- **Build date**: 2026-02-04
- **Build type**: RelWithDebInfo
- **ONNX Runtime**: `onnxruntime-win-x64-directml-1.19.0`
- **DirectML SDK**: `DirectML-1.15.1`
- **Source**: `E:\_DEV\OBSPlugins\obs-onnx-plugin`

## Dependency Chain

```
obs-onnx-plugin.dll
  └── onnxruntime.dll        (delay-loaded)
        └── DirectML.dll     (DirectML execution provider)
              └── d3d12.dll  (system – Windows inbox)
```

> **Note**: `onnxruntime.dll` is delay-loaded by the plugin to ensure `DirectML.dll`
> is resolved from the same directory rather than `System32`, preventing version mismatches.

