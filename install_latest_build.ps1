param(
    [switch]$NoWait
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "OBS ONNX Plugin Installer (DirectML)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if OBS is running
Write-Host "Checking if OBS is running..."
$obsProcess = Get-Process -Name "obs64" -ErrorAction SilentlyContinue
if ($obsProcess) {
    Write-Host "WARNING: OBS is currently running!" -ForegroundColor Yellow
    Write-Host "Please close OBS before installing the plugin." -ForegroundColor Yellow
    Write-Host ""
    if (-not $NoWait) { pause }
    exit 1
}
Write-Host "OBS is not running. Proceeding with installation..." -ForegroundColor Green
Write-Host ""

# Define paths
$ReleaseDll = "E:\_DEV\OBSPlugins\build\lib\Release\obs-onnx-plugin.dll"
$DebugDll = "E:\_DEV\OBSPlugins\build\lib\Debug\obs-onnx-plugin.dll"
$RelWithDebInfoDll = "E:\_DEV\OBSPlugins\build\lib\RelWithDebInfo\obs-onnx-plugin.dll"
$OnnxSourceDir = "E:\_DEV\OBSPlugins\onnxruntime-win-x64-directml-1.19.0\lib"
$DirectMLSourceDir = "E:\_DEV\OBSPlugins\DirectML-1.15.1\bin\x64-win"
$TargetDir = "C:\Program Files\obs-studio\obs-plugins\64bit"

Write-Host "Determining which build is newest..."

# Find the newest build
$builds = @()
if (Test-Path $ReleaseDll) {
    $builds += @{
        Type = "Release"
        Path = $ReleaseDll
        Time = (Get-Item $ReleaseDll).LastWriteTime
    }
}
if (Test-Path $DebugDll) {
    $builds += @{
        Type = "Debug"
        Path = $DebugDll
        Time = (Get-Item $DebugDll).LastWriteTime
    }
}
if (Test-Path $RelWithDebInfoDll) {
    $builds += @{
        Type = "RelWithDebInfo"
        Path = $RelWithDebInfoDll
        Time = (Get-Item $RelWithDebInfoDll).LastWriteTime
    }
}

if ($builds.Count -eq 0) {
    Write-Host "ERROR: No build found! Please build the plugin first." -ForegroundColor Red
    if (-not $NoWait) { pause }
    exit 1
}

# Select newest build
$newestBuild = $builds | Sort-Object -Property Time -Descending | Select-Object -First 1

Write-Host "Found builds:" -ForegroundColor Cyan
foreach ($build in $builds | Sort-Object -Property Time -Descending) {
    $marker = if ($build.Path -eq $newestBuild.Path) { ">>> " } else { "    " }
    Write-Host ("{0}{1}: {2}" -f $marker, $build.Type, $build.Time) -ForegroundColor $(if ($build.Path -eq $newestBuild.Path) { "Green" } else { "Gray" })
}
Write-Host ""

$BuildType = $newestBuild.Type
$PluginSourceDll = $newestBuild.Path

Write-Host "Using $BuildType build" -ForegroundColor Green
Write-Host "Copying plugin DLL..."
Copy-Item -Path $PluginSourceDll -Destination $TargetDir -Force
if (-not $?) {
    Write-Host "ERROR: Failed to copy obs-onnx-plugin.dll" -ForegroundColor Red
    if (-not $NoWait) { pause }
    exit 1
}
Write-Host "[OK] obs-onnx-plugin.dll ($BuildType)" -ForegroundColor Green

Write-Host ""
Write-Host "Copying ONNX Runtime DirectML dependencies..."
Copy-Item -Path "$OnnxSourceDir\onnxruntime.dll" -Destination $TargetDir -Force
if (-not $?) {
    Write-Host "ERROR: Failed to copy onnxruntime.dll" -ForegroundColor Red
    if (-not $NoWait) { pause }
    exit 1
}
Write-Host "[OK] onnxruntime.dll" -ForegroundColor Green

Copy-Item -Path "$DirectMLSourceDir\DirectML.dll" -Destination $TargetDir -Force
if (-not $?) {
    Write-Host "ERROR: Failed to copy DirectML.dll" -ForegroundColor Red
    if (-not $NoWait) { pause }
    exit 1
}
Write-Host "[OK] DirectML.dll" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Installation completed successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Plugin installed to: $TargetDir"
Write-Host "Using DirectML 1.15.5 GPU acceleration (System32 - matched to Windows version)"
Write-Host "DirectML works on NVIDIA, AMD, and Intel GPUs"
Write-Host ""
Write-Host "You can now launch OBS Studio."
Write-Host ""

if (-not $NoWait) { pause }
