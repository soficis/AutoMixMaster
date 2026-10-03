# cmake/FetchOnnxRuntime.cmake
# Fetches pinned ONNX Runtime release archives and the WebGPU EP plugin.
#
# Pinned version: 1.30.0 (exact pin convention)
# Supported architectures:
#   Windows x64 (cpu or cuda), Windows arm64 (cpu)
#   Linux x64 (cpu or cuda), Linux aarch64 (cpu)
#   macOS arm64 (CoreML and WebGPU in-tree)
#   macOS x86_64 is explicitly unsupported for AI inference (FATAL_ERROR).

include(FetchContent)

set(AUTOMIX_ORT_VERSION "1.30.0" CACHE STRING "ONNX Runtime version to fetch")
set(AUTOMIX_ORT_FLAVOR "cpu" CACHE STRING "ONNX Runtime flavor (cpu or cuda)")

# Check U1: Intel Macs do not support ONNX Runtime
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
  message(FATAL_ERROR "ONNX Runtime is not supported on Intel macOS. Please configure with -DENABLE_ONNX=OFF.")
endif()

# Determine OS and Architecture keys
if(WIN32)
  set(_os "win")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64")
    set(_arch "arm64")
  else()
    set(_arch "x64")
  endif()
elseif(APPLE)
  set(_os "osx")
  set(_arch "arm64")
elseif(UNIX)
  set(_os "linux")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64")
    set(_arch "aarch64")
  else()
    set(_arch "x64")
  endif()
else()
  message(FATAL_ERROR "Unsupported platform for AUTOMIX_FETCH_ORT: ${CMAKE_SYSTEM_NAME}")
endif()

# Resolve archive URL and SHA-256 hash
set(_ort_url "")
set(_ort_hash "")

if(_os STREQUAL "win")
  if(_arch STREQUAL "x64")
    if(AUTOMIX_ORT_FLAVOR STREQUAL "cuda")
      set(_ort_url "https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/onnxruntime-win-x64-gpu_cuda13-1.30.0.zip")
      set(_ort_hash "SHA256=8fa4b08359af682cd605892cb59077049700b640128bb93fd2c7776cf9f55bdc")
    else()
      set(_ort_url "https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/onnxruntime-win-x64-1.30.0.zip")
      set(_ort_hash "SHA256=c6ba983baf5681af108599675d2a89c2d145512d02de28aed0bff177cd0ba949")
    endif()
  elseif(_arch STREQUAL "arm64")
    set(_ort_url "https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/onnxruntime-win-arm64-1.30.0.zip")
    set(_ort_hash "SHA256=e53db8a50b23ae35be901cc93428baf997dc8d420333b097b2eae53d3ea9f2d3")
  endif()
elseif(_os STREQUAL "linux")
  if(_arch STREQUAL "x64")
    if(AUTOMIX_ORT_FLAVOR STREQUAL "cuda")
      set(_ort_url "https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/onnxruntime-linux-x64-gpu_cuda13-1.30.0.tgz")
      set(_ort_hash "SHA256=382d79133112388cf94ce5855789b7c9bef12bef76a08b6b277e5a317213adcd")
    else()
      set(_ort_url "https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/onnxruntime-linux-x64-1.30.0.tgz")
      set(_ort_hash "SHA256=a5ed5a3cac51fbb2e90da632ae43d19212faaa20e76484e62bcb7c23ddb3b3fd")
    endif()
  elseif(_arch STREQUAL "aarch64")
    set(_ort_url "https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/onnxruntime-linux-aarch64-1.30.0.tgz")
    set(_ort_hash "SHA256=e16a27a8ed330bbc698df7330b0cf56e722f354e3bcc92118682c74ef3c3e3da")
  endif()
elseif(_os STREQUAL "osx")
  set(_ort_url "https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/onnxruntime-osx-arm64-1.30.0.tgz")
  set(_ort_hash "SHA256=6ebb5062a934537c352937821f9fe9718e7de1a2db1122a93dd363ffd53a7012")
endif()

if(NOT _ort_url)
  message(FATAL_ERROR "No ONNX Runtime 1.30.0 archive configured for ${_os}-${_arch} (flavor: ${AUTOMIX_ORT_FLAVOR})")
endif()

message(STATUS "Fetching ONNX Runtime ${AUTOMIX_ORT_VERSION} (${AUTOMIX_ORT_FLAVOR}) for ${_os}-${_arch}...")

FetchContent_Declare(
  onnxruntime_fetched
  URL "${_ort_url}"
  URL_HASH "${_ort_hash}"
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(onnxruntime_fetched)

set(_ort_root "${onnxruntime_fetched_SOURCE_DIR}")

# Fetch WebGPU EP plugin on Windows and Linux
if(WIN32 OR (UNIX AND NOT APPLE))
  message(STATUS "Fetching ONNX Runtime WebGPU EP plugin (NuGet 0.4.0)...")
  FetchContent_Declare(
    onnxruntime_webgpu_nupkg
    URL "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.EP.WebGpu/0.4.0"
    URL_HASH "SHA256=cc2e093a2eaa760a630d6c3299f015f1dd86885b9eb5877257135308f7c5e923"
    DOWNLOAD_NAME "Microsoft.ML.OnnxRuntime.EP.WebGpu.0.4.0.zip"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(onnxruntime_webgpu_nupkg)

  if(WIN32)
    if(_arch STREQUAL "arm64")
      set(AUTOMIX_WEBGPU_PLUGIN_DIR "${onnxruntime_webgpu_nupkg_SOURCE_DIR}/runtimes/win-arm64/native" CACHE PATH "Directory holding WebGPU plugin" FORCE)
    else()
      set(AUTOMIX_WEBGPU_PLUGIN_DIR "${onnxruntime_webgpu_nupkg_SOURCE_DIR}/runtimes/win-x64/native" CACHE PATH "Directory holding WebGPU plugin" FORCE)
    endif()
  elseif(UNIX AND NOT APPLE)
    if(_arch STREQUAL "aarch64")
      set(AUTOMIX_WEBGPU_PLUGIN_DIR "${onnxruntime_webgpu_nupkg_SOURCE_DIR}/runtimes/linux-arm64/native" CACHE PATH "Directory holding WebGPU plugin" FORCE)
    else()
      set(AUTOMIX_WEBGPU_PLUGIN_DIR "${onnxruntime_webgpu_nupkg_SOURCE_DIR}/runtimes/linux-x64/native" CACHE PATH "Directory holding WebGPU plugin" FORCE)
    endif()
  endif()
endif()

# Configure ONNX Runtime CMake target and variables
if(EXISTS "${_ort_root}/lib/cmake/onnxruntime/onnxruntimeConfig.cmake")
  set(onnxruntime_DIR "${_ort_root}/lib/cmake/onnxruntime")
  if(EXISTS "${_ort_root}/lib" AND NOT EXISTS "${_ort_root}/lib64")
    file(CREATE_LINK "${_ort_root}/lib" "${_ort_root}/lib64" SYMBOLIC)
  endif()
  if(NOT EXISTS "${_ort_root}/include/onnxruntime")
    file(MAKE_DIRECTORY "${_ort_root}/include/onnxruntime")
  endif()
  find_package(onnxruntime 1.30.0 EXACT CONFIG REQUIRED NO_DEFAULT_PATH)
  target_include_directories(onnxruntime::onnxruntime INTERFACE "${_ort_root}/include")
  set(ONNXRUNTIME_INCLUDE_DIR "${_ort_root}/include")
  if(APPLE)
    set(ONNXRUNTIME_LIBRARY "${_ort_root}/lib/libonnxruntime.dylib")
  else()
    set(ONNXRUNTIME_LIBRARY "${_ort_root}/lib/libonnxruntime.so")
  endif()
else()
  # Windows zip archives do not bundle CMake config files
  set(ONNXRUNTIME_INCLUDE_DIR "${_ort_root}/include")
  set(ONNXRUNTIME_LIBRARY "${_ort_root}/lib/onnxruntime.lib")
  if(NOT TARGET onnxruntime::onnxruntime)
    add_library(onnxruntime::onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
      IMPORTED_IMPLIB "${ONNXRUNTIME_LIBRARY}"
      IMPORTED_LOCATION "${_ort_root}/lib/onnxruntime.dll"
    )
  endif()
endif()
