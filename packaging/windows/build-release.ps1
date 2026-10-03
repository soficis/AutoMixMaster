# Builds the Windows release ZIP of AutoMixMaster.
#
# The package contains the app, its assets and the ONNX Runtime it links
# against. Point -OnnxRuntimeDir at the CUDA build
# (onnxruntime-win-x64-gpu_cuda13-<ver>) so GPU acceleration is available;
# users download NVIDIA's CUDA libraries on demand from inside the app, so they
# are never part of the package (the install step fails if they are).
#
#   powershell -File packaging\windows\build-release.ps1 `
#     -OnnxRuntimeDir C:\lib\onnxruntime-win-x64-gpu_cuda13-1.30.0
param(
  [string]$OnnxRuntimeDir = "",
  [string]$WebGpuPluginDir = "",
  [string]$BuildDir = "build-release",
  [string]$Generator = "Visual Studio 18 2026",
  [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $repo

if (-not $WebGpuPluginDir) {
  $defaultWebGpu = "C:\lib\webgpu-win-x64\runtimes\win-x64\native"
  if (Test-Path $defaultWebGpu) {
    $WebGpuPluginDir = $defaultWebGpu
  } elseif ($OnnxRuntimeDir) {
    Write-Warning "No WebGPU plugin: the package will use CUDA or CPU only."
  }
}

# A fresh directory: a reused cache could carry AUTOMIX_CUDA_RUNTIME_DIR or a
# different ONNX Runtime from a developer build.
if (Test-Path (Join-Path $BuildDir "CMakeCache.txt")) {
  throw "$BuildDir already holds a CMake cache; pass a new -BuildDir for a clean release build."
}

$cmakeExtraArgs = @()
if ($OnnxRuntimeDir) {
  $include = Join-Path $OnnxRuntimeDir "include"
  $library = Join-Path $OnnxRuntimeDir "lib\onnxruntime.lib"
  if (-not (Test-Path $library)) { throw "ONNX Runtime library not found: $library" }
  if (-not (Test-Path (Join-Path $OnnxRuntimeDir "lib\onnxruntime_providers_cuda.dll"))) {
    Write-Warning "This ONNX Runtime has no CUDA provider; the package will run on CPU only."
  }
  $cmakeExtraArgs += "-DONNXRUNTIME_INCLUDE_DIR=$include"
  $cmakeExtraArgs += "-DONNXRUNTIME_LIBRARY=$library"
  if ($WebGpuPluginDir) {
    $cmakeExtraArgs += "-DAUTOMIX_WEBGPU_PLUGIN_DIR=$WebGpuPluginDir"
  }
} else {
  $cmakeExtraArgs += "-DAUTOMIX_FETCH_ORT=ON"
  $cmakeExtraArgs += "-DAUTOMIX_ORT_FLAVOR=cuda"
  if ($WebGpuPluginDir) {
    $cmakeExtraArgs += "-DAUTOMIX_WEBGPU_PLUGIN_DIR=$WebGpuPluginDir"
  }
}

cmake -S . -B $BuildDir -G $Generator -A x64 `
  -DENABLE_ONNX=ON `
  @cmakeExtraArgs
if ($LASTEXITCODE -ne 0) { throw "configure failed" }


cmake --build $BuildDir --config Release --parallel
if ($LASTEXITCODE -ne 0) { throw "build failed" }

if (-not $SkipTests) {
  ctest --test-dir $BuildDir -C Release --output-on-failure
  if ($LASTEXITCODE -ne 0) { throw "tests failed" }
}

Push-Location $BuildDir
try {
  cpack -C Release
  if ($LASTEXITCODE -ne 0) { throw "packaging failed" }
  Get-ChildItem -Filter "AutoMixMaster-*.zip" | ForEach-Object { "Package: $($_.FullName) ($([math]::Round($_.Length / 1MB)) MB)" }
} finally {
  Pop-Location
}
