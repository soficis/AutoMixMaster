cmake_minimum_required(VERSION 3.24)
project(AutoMixMaster VERSION 0.4.2 LANGUAGES C CXX)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 4.0)
  set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
endif()

if(WIN32 AND CMAKE_GENERATOR MATCHES "Visual Studio")
  if(NOT CMAKE_GENERATOR MATCHES "Visual Studio 1[789]")
    message(FATAL_ERROR "Windows Visual Studio builds require Visual Studio 2022 (17) or later.")
  endif()
endif()

option(BUILD_SHARED_LIBS "Build libraries as shared" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(DISTRIBUTION_MODE "OSS" CACHE STRING "Distribution mode: OSS or PROPRIETARY")
set_property(CACHE DISTRIBUTION_MODE PROPERTY STRINGS OSS PROPRIETARY)
string(TOUPPER "${DISTRIBUTION_MODE}" DISTRIBUTION_MODE)
if(NOT DISTRIBUTION_MODE STREQUAL "OSS" AND NOT DISTRIBUTION_MODE STREQUAL "PROPRIETARY")
  message(FATAL_ERROR "DISTRIBUTION_MODE must be OSS or PROPRIETARY.")
endif()

option(ENABLE_PHASELIMITER "Enable optional PhaseLimiter renderer" OFF)
option(ENABLE_ONNX "Enable optional ONNX model inference adapter" OFF)
option(ENABLE_RTNEURAL "Enable optional RTNeural model inference adapter" OFF)
option(RTNEURAL_XSIMD "Enable RTNeural XSIMD backend (if available)" ON)
option(RTNEURAL_USE_AVX "Enable RTNeural AVX backend on x86_64 (if available)" ON)
option(ENABLE_LIBEBUR128 "Enable libebur128 standards loudness metering" ON)
option(ENABLE_GPL_BUNDLED_LIMITERS "Enable bundled GPL limiters (OSS mode only)" OFF)
option(ENABLE_BUNDLED_LAME "Enable bundled LAME fallback lookup for MP3 export" ON)
option(ENABLE_EXTERNAL_TOOL_SUPPORT "Enable user-supplied external renderer integrations" ON)
option(ENABLE_SANITIZERS "Enable sanitizer instrumentation for debug builds" OFF)
option(ENABLE_CLANG_TIDY "Enable clang-tidy analysis while compiling" OFF)
option(BUILD_TESTING "Build tests" ON)
option(BUILD_TOOLS "Build developer tools" ON)

# Linux has no native HTTPS client in JUCE: without libcurl, JUCE falls back to a raw-socket client that cannot
# speak TLS, so model/tool downloads fail. Windows (WinINet) and macOS (NSURLSession) must keep JUCE_USE_CURL=0.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(AUTOMIX_JUCE_USE_CURL 1)
else()
  set(AUTOMIX_JUCE_USE_CURL 0)
endif()

if(DISTRIBUTION_MODE STREQUAL "PROPRIETARY" AND ENABLE_GPL_BUNDLED_LIMITERS)
  message(WARNING "ENABLE_GPL_BUNDLED_LIMITERS is disabled in PROPRIETARY mode.")
  set(ENABLE_GPL_BUNDLED_LIMITERS OFF CACHE BOOL "Enable bundled GPL limiters (OSS mode only)" FORCE)
endif()

include(FetchContent)

FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

FetchContent_Declare(
  juce
  GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
  GIT_TAG 8.0.8
)
FetchContent_MakeAvailable(juce)

if(ENABLE_LIBEBUR128)
  FetchContent_Declare(
    libebur128
    GIT_REPOSITORY https://github.com/jiixyj/libebur128.git
    GIT_TAG v1.2.6
  )
  FetchContent_MakeAvailable(libebur128)
  if(MSVC AND TARGET ebur128)
    # ebur128.c emits #warning in its non-SSE2 path (hit on ARM64, where _M_X64
    # is undefined). MSVC only accepts #warning in C mode with /std:clatest and
    # rejects it with /std:c17 or lower (error C1188). Do NOT "fix" this back
    # to /std:c17 - it breaks Windows ARM64 builds.
    target_compile_options(ebur128 PRIVATE /std:clatest)
  endif()
endif()

add_library(automix_core
  src/domain/Bus.cpp
  src/domain/BatchTypes.cpp
  src/domain/JsonSerialization.cpp
  src/domain/MasterPlan.cpp
  src/domain/MixPlan.cpp
  src/domain/ProjectProfile.cpp
  src/domain/RenderSettings.cpp
  src/domain/Session.cpp
  src/domain/Stem.cpp
  src/domain/StemOrigin.cpp
  src/engine/AudioBuffer.cpp
  src/engine/AudioFileIO.cpp
  src/engine/AudioPreviewEngine.cpp
  src/engine/LoudnessMeter.cpp
  src/engine/AudioResampler.cpp
  src/engine/TransportController.cpp
  src/engine/BatchQueueRunner.cpp
  src/engine/OfflineRenderPipeline.cpp
  src/engine/ResidualBlendProcessor.cpp
  src/engine/SessionRepository.cpp
  src/dsp/DeEsser.cpp
  src/dsp/DynamicDeHarshEq.cpp
  src/dsp/ItoMasterFxChain.cpp
  src/dsp/LookaheadLimiter.cpp
  src/dsp/MidSideProcessor.cpp
  src/dsp/MultibandProcessor.cpp
  src/dsp/SignalMath.cpp
  src/dsp/SoftClipper.cpp
  src/dsp/TruePeakDetector.cpp
  src/analysis/ArtifactRiskEstimator.cpp
  src/analysis/StemHealthAssistant.cpp
  src/analysis/StemAnalyzer.cpp
  src/analysis/SpectrogramFrontEnd.cpp
  src/automix/HeuristicAutoMixStrategy.cpp
  src/automaster/HeuristicAutoMasterStrategy.cpp
  src/automaster/ItoMasterStrategy.cpp
  src/automaster/OriginalMixReference.cpp
  src/automaster/ReferenceMatchStrategy.cpp
  src/renderers/BuiltInRenderer.cpp
  src/renderers/BundledToolDiscovery.cpp
  src/renderers/ExternalLimiterRenderer.cpp
  src/renderers/FfmpegDiscovery.cpp
  src/renderers/FfmpegRenderer.cpp
  src/renderers/RendererRegistry.cpp
  src/renderers/PhaseLimiterDiscovery.cpp
  src/renderers/PhaseLimiterRenderer.cpp
  src/renderers/PostRendererFactory.cpp
  src/renderers/RendererFactory.cpp
  src/renderers/RendererPipeline.cpp
  src/renderers/RsgainDiscovery.cpp
  src/renderers/RsgainRenderer.cpp
  src/renderers/SoxDiscovery.cpp
  src/renderers/SoxRenderer.cpp
  src/platform/ThirdPartyComponentRegistry.cpp
  src/ai/HuggingFaceModelHub.cpp
  src/ai/GitHubReleaseModelHub.cpp
  src/ai/ModelCatalogValidator.cpp
  src/ai/ModelPackLoader.cpp
  src/ai/ModelManager.cpp
  src/ai/ModelStrategy.cpp
  src/ai/AutoMixStrategyAI.cpp
  src/ai/AutoMasterStrategyAI.cpp
  src/ai/ItoMasterAdapter.cpp
  src/ai/FeatureSchema.cpp
  src/ai/MasteringCompliance.cpp
  src/ai/OnnxModelInference.cpp
  src/ai/GpuCapabilityDetector.cpp
  src/ai/StemSeparator.cpp
  src/ai/StemRoleClassifierAI.cpp
  src/ai/TensorTypes.cpp
  src/ai/SeparationRunner.cpp
  src/ai/OnnxTensorInference.cpp
  src/ai/OrtRuntime.cpp
  src/ai/OnnxExternalData.cpp
  src/ai/GpuMemory.cpp
  src/ai/GpuRuntimePack.cpp
  src/ai/ModelStorage.cpp
  src/util/StringUtils.cpp
  src/util/LameDownloader.cpp
  src/util/MetadataPolicy.cpp
  src/util/WavWriter.cpp
)

target_include_directories(automix_core PUBLIC src "${CMAKE_BINARY_DIR}/generated")

include(cmake/PhaseLimiterPins.cmake)
automix_generate_phaselimiter_pins_header()

if(WIN32)
  # GpuRuntimePack reads adapter memory through DXGI to decide whether to offer CUDA.
  target_link_libraries(automix_core PRIVATE dxgi)
endif()

set(AUTOMIX_HAS_NATIVE_ORT OFF)
set(AUTOMIX_HAS_EP_PLUGIN OFF)
set(AUTOMIX_HAS_EP_CONTEXT OFF)
option(AUTOMIX_FETCH_ORT "Automatically fetch pinned ONNX Runtime release archive" OFF)
set(AUTOMIX_ORT_VERSION "1.30.0" CACHE STRING "ONNX Runtime version to fetch")
set(AUTOMIX_ORT_FLAVOR "cpu" CACHE STRING "ONNX Runtime flavor (cpu or cuda)")

if(ENABLE_ONNX)
  if(AUTOMIX_FETCH_ORT)
    include(cmake/FetchOnnxRuntime.cmake)
    target_include_directories(automix_core PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})
    if(TARGET onnxruntime::onnxruntime)
      target_link_libraries(automix_core PUBLIC onnxruntime::onnxruntime)
    else()
      target_link_libraries(automix_core PUBLIC ${ONNXRUNTIME_LIBRARY})
    endif()
    set(AUTOMIX_HAS_NATIVE_ORT ON)
  else()
    find_path(ONNXRUNTIME_INCLUDE_DIR
      NAMES onnxruntime_cxx_api.h
      PATH_SUFFIXES onnxruntime/core/session
    )
    find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime)
    if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
      target_include_directories(automix_core PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})
      target_link_libraries(automix_core PUBLIC ${ONNXRUNTIME_LIBRARY})
      set(AUTOMIX_HAS_NATIVE_ORT ON)
      message(WARNING "Using unpinned external ONNX Runtime. Recommended: configure with -DAUTOMIX_FETCH_ORT=ON")
    endif()
  endif()

  if(AUTOMIX_HAS_NATIVE_ORT)
    target_compile_definitions(automix_core PUBLIC AUTOMIX_ORT_EXPECTED_VERSION="${AUTOMIX_ORT_VERSION}")

    # Feature-detect the optional APIs rather than parsing a version string out of
    # the headers. ONNX Runtime does not expose a reliable single version macro, and
    # a header either declares an entry point or it does not, which is exact where a
    # parsed version is only a guess.
    # The official release archives ship a flat include/ directory; source-tree
    # installs nest the same headers under onnxruntime/core/session/.
    file(GLOB _ort_api_headers
      "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_c*_api.h"
      "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime/core/session/onnxruntime_c*_api.h")
    foreach(_ort_header IN LISTS _ort_api_headers)
      file(STRINGS "${_ort_header}" _ort_header_text)
      if(_ort_header_text MATCHES "RegisterExecutionProviderLibrary")
        set(AUTOMIX_HAS_EP_PLUGIN ON)
      endif()
      if(_ort_header_text MATCHES "OrtCompileApi")
        set(AUTOMIX_HAS_EP_CONTEXT ON)
      endif()
    endforeach()

    if(AUTOMIX_HAS_EP_PLUGIN)
      target_compile_definitions(automix_core PUBLIC AUTOMIX_HAS_EP_PLUGIN=1)
    else()
      message(WARNING "ONNX Runtime headers do not declare the plugin execution-provider API (1.23+). GPU plugins such as CUDA cannot be loaded at runtime; built-in providers and CPU are unaffected.")
    endif()

    if(AUTOMIX_HAS_EP_CONTEXT)
      target_compile_definitions(automix_core PUBLIC AUTOMIX_HAS_EP_CONTEXT=1)
    else()
      message(WARNING "ONNX Runtime headers do not declare OrtCompileApi (1.22+). Pre-compiled EPContext model caches are unavailable.")
    endif()
  else()
    message(WARNING "ENABLE_ONNX is ON but native ONNX Runtime headers/library were not found. Falling back to deterministic ONNX adapter.")
  endif()
endif()

set(AUTOMIX_WEBGPU_PLUGIN_DIR "" CACHE PATH "Directory holding the ONNX Runtime WebGPU plugin EP (onnxruntime_providers_webgpu + dxcompiler/dxil on Windows)")

set(AUTOMIX_ORT_RUNTIME_FILES "")
if(AUTOMIX_HAS_NATIVE_ORT)
  get_filename_component(_ort_lib_dir "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
  if(WIN32)
    foreach(_core_dll onnxruntime.dll onnxruntime_providers_shared.dll onnxruntime_providers_cuda.dll)
      if(EXISTS "${_ort_lib_dir}/${_core_dll}")
        list(APPEND AUTOMIX_ORT_RUNTIME_FILES "${_ort_lib_dir}/${_core_dll}")
      endif()
    endforeach()
    if(AUTOMIX_WEBGPU_PLUGIN_DIR)
      foreach(_webgpu_dll onnxruntime_providers_webgpu.dll dxcompiler.dll dxil.dll)
        set(_webgpu_file "${AUTOMIX_WEBGPU_PLUGIN_DIR}/${_webgpu_dll}")
        if(EXISTS "${_webgpu_file}")
          list(APPEND AUTOMIX_ORT_RUNTIME_FILES "${_webgpu_file}")
        else()
          message(WARNING "WebGPU plugin file missing: ${_webgpu_file}")
        endif()
      endforeach()
    endif()
  elseif(APPLE)
    file(GLOB _ort_dylibs LIST_DIRECTORIES false "${_ort_lib_dir}/libonnxruntime*.dylib")
    foreach(_dylib IN LISTS _ort_dylibs)
      if(NOT _dylib MATCHES "providers_tensorrt")
        list(APPEND AUTOMIX_ORT_RUNTIME_FILES "${_dylib}")
      endif()
    endforeach()
    if(AUTOMIX_WEBGPU_PLUGIN_DIR)
      set(_webgpu_dylib "${AUTOMIX_WEBGPU_PLUGIN_DIR}/libonnxruntime_providers_webgpu.dylib")
      if(EXISTS "${_webgpu_dylib}")
        list(APPEND AUTOMIX_ORT_RUNTIME_FILES "${_webgpu_dylib}")
      endif()
    endif()
  elseif(UNIX)
    file(GLOB _ort_sos LIST_DIRECTORIES false "${_ort_lib_dir}/libonnxruntime*.so*")
    foreach(_so IN LISTS _ort_sos)
      if(NOT _so MATCHES "providers_tensorrt")
        list(APPEND AUTOMIX_ORT_RUNTIME_FILES "${_so}")
      endif()
    endforeach()
    if(AUTOMIX_WEBGPU_PLUGIN_DIR)
      set(_webgpu_so "${AUTOMIX_WEBGPU_PLUGIN_DIR}/libonnxruntime_providers_webgpu.so")
      if(EXISTS "${_webgpu_so}")
        list(APPEND AUTOMIX_ORT_RUNTIME_FILES "${_webgpu_so}")
      else()
        message(WARNING "WebGPU plugin file missing: ${_webgpu_so}")
      endif()
    endif()
  endif()
endif()

function(automix_stage_ort TARGET_NAME)
  if(NOT AUTOMIX_HAS_NATIVE_ORT OR NOT AUTOMIX_ORT_RUNTIME_FILES)
    return()
  endif()
  if(WIN32)
    set(_staging_files ${AUTOMIX_ORT_RUNTIME_FILES})
    if(AUTOMIX_CUDA_RUNTIME_DIR)
      file(GLOB _automix_cuda_dlls "${AUTOMIX_CUDA_RUNTIME_DIR}/*.dll")
      list(APPEND _staging_files ${_automix_cuda_dlls})
    endif()
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_staging_files} "$<TARGET_FILE_DIR:${TARGET_NAME}>"
      VERBATIM)
  elseif(APPLE)
    get_target_property(_is_bundle ${TARGET_NAME} MACOSX_BUNDLE)
    if(_is_bundle)
      add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${TARGET_NAME}>/../Frameworks"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${AUTOMIX_ORT_RUNTIME_FILES} "$<TARGET_FILE_DIR:${TARGET_NAME}>/../Frameworks"
        VERBATIM)
    else()
      add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${TARGET_NAME}>/lib"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${AUTOMIX_ORT_RUNTIME_FILES} "$<TARGET_FILE_DIR:${TARGET_NAME}>/lib"
        VERBATIM)
    endif()
  elseif(UNIX)
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${TARGET_NAME}>/lib"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${AUTOMIX_ORT_RUNTIME_FILES} "$<TARGET_FILE_DIR:${TARGET_NAME}>/lib"
      VERBATIM)
  endif()
endfunction()

function(automix_set_rpath TARGET_NAME)
  if(APPLE)
    get_target_property(_is_bundle ${TARGET_NAME} MACOSX_BUNDLE)
    if(_is_bundle)
      set_target_properties(${TARGET_NAME} PROPERTIES
        BUILD_RPATH "@executable_path/../Frameworks"
        INSTALL_RPATH "@executable_path/../Frameworks"
      )
    else()
      set_target_properties(${TARGET_NAME} PROPERTIES
        BUILD_RPATH "@loader_path/lib"
        INSTALL_RPATH "@loader_path/lib"
      )
    endif()
  elseif(UNIX AND NOT APPLE)
    set_target_properties(${TARGET_NAME} PROPERTIES
      BUILD_RPATH "$ORIGIN/lib"
      INSTALL_RPATH "$ORIGIN/lib"
    )
  endif()
endfunction()

target_link_libraries(automix_core
  PUBLIC
    nlohmann_json::nlohmann_json
    juce::juce_events
    juce::juce_audio_formats
    juce::juce_dsp
)

if(AUTOMIX_JUCE_USE_CURL)
  find_package(CURL REQUIRED)
  target_link_libraries(automix_core PUBLIC CURL::libcurl)
endif()

set(LIBEBUR128_BACKEND_AVAILABLE OFF)
if(ENABLE_LIBEBUR128)
  if(DEFINED libebur128_SOURCE_DIR)
    target_include_directories(automix_core PUBLIC "${libebur128_SOURCE_DIR}/ebur128")
  endif()
  if(TARGET ebur128)
    target_link_libraries(automix_core PUBLIC ebur128)
    set(LIBEBUR128_BACKEND_AVAILABLE ON)
  elseif(TARGET libebur128)
    target_link_libraries(automix_core PUBLIC libebur128)
    set(LIBEBUR128_BACKEND_AVAILABLE ON)
  elseif(TARGET libebur128::libebur128)
    target_link_libraries(automix_core PUBLIC libebur128::libebur128)
    set(LIBEBUR128_BACKEND_AVAILABLE ON)
  else()
    message(WARNING "libebur128 target not found; falling back to built-in loudness approximation.")
  endif()
endif()

target_compile_definitions(automix_core
  PUBLIC
    APP_VERSION="${PROJECT_VERSION}"
    SESSION_SCHEMA_VERSION=2
    DISTRIBUTION_MODE_${DISTRIBUTION_MODE}=1
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=${AUTOMIX_JUCE_USE_CURL}
    $<$<BOOL:${ENABLE_PHASELIMITER}>:ENABLE_PHASELIMITER=1>
    $<$<BOOL:${ENABLE_ONNX}>:ENABLE_ONNX=1>
    $<$<BOOL:${ENABLE_RTNEURAL}>:ENABLE_RTNEURAL=1>
    $<$<BOOL:${LIBEBUR128_BACKEND_AVAILABLE}>:ENABLE_LIBEBUR128=1>
    $<$<BOOL:${ENABLE_GPL_BUNDLED_LIMITERS}>:ENABLE_GPL_BUNDLED_LIMITERS=1>
    $<$<BOOL:${ENABLE_BUNDLED_LAME}>:ENABLE_BUNDLED_LAME=1>
    $<$<BOOL:${ENABLE_EXTERNAL_TOOL_SUPPORT}>:ENABLE_EXTERNAL_TOOL_SUPPORT=1>
    $<$<BOOL:${RTNEURAL_XSIMD}>:RTNEURAL_XSIMD=1>
    $<$<BOOL:${RTNEURAL_USE_AVX}>:RTNEURAL_USE_AVX=1>
    $<$<BOOL:${AUTOMIX_HAS_NATIVE_ORT}>:AUTOMIX_HAS_NATIVE_ORT=1>
)

if(ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY_EXE NAMES clang-tidy)
  if(CLANG_TIDY_EXE)
    set_target_properties(automix_core PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
  else()
    message(WARNING "ENABLE_CLANG_TIDY is ON but clang-tidy was not found.")
  endif()
endif()

if(MSVC)
  target_compile_options(automix_core PRIVATE /W4 /permissive-)
else()
  target_compile_options(automix_core PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
endif()

if(ENABLE_SANITIZERS)
  if(MSVC)
    target_compile_options(automix_core PRIVATE /fsanitize=address /Zi)
    target_link_options(automix_core PRIVATE /fsanitize=address)
  else()
    target_compile_options(automix_core PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(automix_core PRIVATE -fsanitize=address,undefined)
  endif()
endif()

set(_automix_needs_curl FALSE)
if(AUTOMIX_JUCE_USE_CURL)
  set(_automix_needs_curl TRUE)
endif()

juce_add_gui_app(AutoMixMasterApp
  NEEDS_CURL ${_automix_needs_curl}
  PRODUCT_NAME "AutoMixMaster"
  COMPANY_NAME "AutoMixMaster"
  ICON_BIG "assets/icon.png"
)

target_sources(AutoMixMasterApp PRIVATE
  src/app/Main.cpp
  src/app/style/AutoMixLookAndFeel.cpp
  src/app/controllers/ModelController.cpp
  src/app/controllers/ImportController.cpp
  src/app/controllers/ExportController.cpp
  src/app/controllers/ProcessingController.cpp
  src/app/controllers/SessionController.cpp
  src/app/controllers/PreviewController.cpp
  src/app/controllers/ProfileController.cpp
  src/app/ui/MainLayout.cpp
  src/app/ui/VerificationEngine.cpp
  src/app/ui/HeaderBar.cpp
  src/app/ui/HeroWaveform.cpp
  src/app/ui/TransportBar.cpp
  src/app/ui/GlowMeters.cpp
  src/app/ui/TaskCenterPanel.cpp
  src/app/ui/StemPanel.cpp
  src/app/ui/ControlDeck.cpp
  src/app/ui/SelectionState.cpp
  src/app/ui/TaskLifecycle.cpp
  src/app/ui/TaskOrchestrator.cpp
  src/app/ui/AudioPreviewManager.cpp
  src/app/ui/SessionManager.cpp
  src/app/ui/ModelBrowserPanel.cpp
)

target_include_directories(AutoMixMasterApp PRIVATE src)

target_link_libraries(AutoMixMasterApp
  PRIVATE
    automix_core
    juce::juce_gui_extra
    juce::juce_audio_devices
    juce::juce_audio_formats
    juce::juce_audio_utils
    juce::juce_dsp
    juce::juce_opengl
)

target_compile_definitions(AutoMixMasterApp PRIVATE
  JUCE_WEB_BROWSER=0
  JUCE_USE_CURL=${AUTOMIX_JUCE_USE_CURL}
)

# Pinned PhaseLimiter fetching and staging
# Defaults ON whenever assets/phaselimiter/bin has no native binary for the platform key.
include(cmake/FetchPhaseLimiter.cmake)
automix_phaselimiter_platform_key(_automix_host_pl_key)
if(_automix_host_pl_key STREQUAL "windows-x64")
  set(_automix_native_bin_name "phase_limiter.exe")
else()
  set(_automix_native_bin_name "phase_limiter")
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/assets/phaselimiter/bin/${_automix_native_bin_name}")
  set(AUTOMIX_FETCH_PHASELIMITER_DEFAULT OFF)
else()
  set(AUTOMIX_FETCH_PHASELIMITER_DEFAULT ON)
endif()
option(AUTOMIX_FETCH_PHASELIMITER "Fetch and stage pinned PhaseLimiter release archive" ${AUTOMIX_FETCH_PHASELIMITER_DEFAULT})

if(AUTOMIX_FETCH_PHASELIMITER)
  include(cmake/FetchPhaseLimiter.cmake)
endif()


if(EXISTS "${CMAKE_SOURCE_DIR}/assets/limiters")
  add_custom_command(TARGET AutoMixMasterApp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:AutoMixMasterApp>/assets"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/assets/limiters"
            "$<TARGET_FILE_DIR:AutoMixMasterApp>/assets/limiters"
    COMMENT "Copying bundled limiter descriptors next to AutoMixMaster executable"
    VERBATIM
  )
endif()

if(BUILD_TOOLS)
  add_executable(automix_dev_tools
    tools/dev_tools.cpp
    tools/commands/DevToolsUtils.cpp
    tools/commands/ModelCommands.cpp
    tools/commands/SessionCommands.cpp
    tools/commands/RenderCommands.cpp
    tools/commands/EvalCommands.cpp
    tools/commands/BatchCommands.cpp
    tests/regression/RegressionHarness.cpp
  )
  target_include_directories(automix_dev_tools PRIVATE src tools tests/regression)
  target_link_libraries(automix_dev_tools PRIVATE automix_core)
  target_compile_definitions(automix_dev_tools PRIVATE AUTOMIX_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
endif()

if(BUILD_TESTING)
  enable_testing()

  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
  )
  FetchContent_MakeAvailable(Catch2)

  add_executable(automix_tests
    tests/unit/TestMain.cpp
    tests/unit/ControllerTests.cpp
    tests/unit/TaskCenterPanelTests.cpp
    tests/unit/SessionSerializationTests.cpp
    tests/unit/AudioIoTests.cpp
    tests/unit/AnalysisTests.cpp
    tests/unit/StemOriginSafetyTests.cpp
    tests/unit/ResidualBlendTests.cpp
    tests/unit/PhaseLimiterDiscoveryTests.cpp
    tests/unit/BundledToolDiscoveryTests.cpp
    tests/unit/PhaseLimiterRendererTests.cpp
    tests/unit/ExternalLimiterRendererTests.cpp
    tests/unit/RendererFactoryTests.cpp
    tests/unit/PostRendererFactoryTests.cpp
    tests/unit/RendererPipelineTests.cpp
    tests/unit/RendererRegistryTests.cpp
    tests/unit/RendererRegistryTrustPolicyTests.cpp
    tests/unit/ProjectProfileTests.cpp
    tests/unit/MetadataPolicyTests.cpp
    tests/unit/AutoMasterTests.cpp
    tests/unit/AiExtensionTests.cpp
    tests/unit/ItoMasterFxChainTests.cpp
    tests/unit/ItoMasterAdapterTests.cpp
    tests/unit/ItoMasterStrategyTests.cpp
    tests/unit/LoudnessMeterTests.cpp
    tests/unit/LookaheadLimiterTests.cpp
    tests/unit/MasteringModulesTests.cpp
    tests/unit/AudioPreviewEngineTests.cpp
    tests/unit/TransportControllerTests.cpp
    tests/unit/BatchProcessingTests.cpp
    tests/unit/OfflineRenderPipelineTests.cpp
    tests/unit/OnnxModelInferenceTests.cpp
    tests/unit/StemSeparatorTests.cpp
    tests/unit/TensorInferenceTests.cpp
    tests/unit/ModelBenchmarkTests.cpp
    tests/unit/GpuBenchmarkTests.cpp
    tests/regression/RegressionHarness.cpp
    tests/regression/RegressionHarnessTests.cpp
    tests/integration/GoldenFileTest.cpp
    tests/integration/GoldenFileTests.cpp
    tests/integration/PipelineIntegrationTests.cpp
    src/app/ui/TaskCenterPanel.cpp
    src/app/style/AutoMixLookAndFeel.cpp
    src/app/controllers/ModelController.cpp
    src/app/controllers/ImportController.cpp
    src/app/controllers/ExportController.cpp
    src/app/controllers/ProcessingController.cpp
    src/app/controllers/SessionController.cpp
    src/app/controllers/PreviewController.cpp
    src/app/controllers/ProfileController.cpp
    src/app/controllers/OriginalMixController.cpp
  )

  target_include_directories(automix_tests PRIVATE src tests/regression tests/integration tools)
  target_link_libraries(automix_tests PRIVATE automix_core Catch2::Catch2WithMain juce::juce_gui_extra)
  target_compile_definitions(automix_tests PRIVATE
    AUTOMIX_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    JUCE_MODAL_LOOPS_PERMITTED=1
  )

  if(ENABLE_CLANG_TIDY AND CLANG_TIDY_EXE)
    set_target_properties(automix_tests PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
  endif()

  if(ENABLE_SANITIZERS)
    if(MSVC)
      target_compile_options(automix_tests PRIVATE /fsanitize=address /Zi)
      target_link_options(automix_tests PRIVATE /fsanitize=address)
    else()
      target_compile_options(automix_tests PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
      target_link_options(automix_tests PRIVATE -fsanitize=address,undefined)
    endif()
  endif()

  add_executable(automix_regression_cli
    tools/regression_runner.cpp
    tests/regression/RegressionHarness.cpp
  )
  target_include_directories(automix_regression_cli PRIVATE src tests/regression)
  target_link_libraries(automix_regression_cli PRIVATE automix_core)
  target_compile_definitions(automix_regression_cli PRIVATE AUTOMIX_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

  include(CTest)
  include(${Catch2_SOURCE_DIR}/extras/Catch.cmake)
  catch_discover_tests(automix_tests)
  add_test(NAME cmake_phaselimiter_arch COMMAND ${CMAKE_COMMAND} -P ${CMAKE_SOURCE_DIR}/tests/cmake/phaselimiter_arch_test.cmake)
  add_test(NAME cmake_phaselimiter_install_guard COMMAND ${CMAKE_COMMAND} -P ${CMAKE_SOURCE_DIR}/tests/cmake/phaselimiter_install_guard_test.cmake)
  set_tests_properties(cmake_phaselimiter_install_guard PROPERTIES LABELS "slow")
endif()

# Stage ONNX Runtime (and, optionally, CUDA/cuDNN) DLLs beside each executable,
# and configure rpath for Linux ($ORIGIN/lib) and macOS (@executable_path/../Frameworks, @loader_path/lib).
set(AUTOMIX_CUDA_RUNTIME_DIR "" CACHE PATH "Folder of CUDA runtime + cuDNN DLLs staged next to executables (optional)")

automix_stage_ort(AutoMixMasterApp)
if(COMMAND automix_stage_phaselimiter)
  automix_stage_phaselimiter(AutoMixMasterApp)
endif()
automix_set_rpath(AutoMixMasterApp)

if(TARGET automix_dev_tools)
  automix_stage_ort(automix_dev_tools)
  if(COMMAND automix_stage_phaselimiter)
    automix_stage_phaselimiter(automix_dev_tools)
  endif()
  automix_set_rpath(automix_dev_tools)
endif()

if(TARGET automix_tests)
  automix_stage_ort(automix_tests)
  if(COMMAND automix_stage_phaselimiter)
    automix_stage_phaselimiter(automix_tests)
  endif()
  automix_set_rpath(automix_tests)
endif()

if(TARGET automix_regression_cli)
  automix_stage_ort(automix_regression_cli)
  if(COMMAND automix_stage_phaselimiter)
    automix_stage_phaselimiter(automix_regression_cli)
  endif()
  automix_set_rpath(automix_regression_cli)
endif()

# ── Release packaging (Cross-platform portable package via CPack) ─────────────
# cmake --install <build> --config Release --component application --prefix <dir>
# or: cpack -C Release. Only the "application" component is packaged.
install(TARGETS AutoMixMasterApp
  RUNTIME DESTINATION . COMPONENT application
  BUNDLE DESTINATION . COMPONENT application
  RESOURCE DESTINATION . COMPONENT application
)

if(AUTOMIX_HAS_NATIVE_ORT AND AUTOMIX_ORT_RUNTIME_FILES)
  if(WIN32)
    install(FILES ${AUTOMIX_ORT_RUNTIME_FILES} DESTINATION . COMPONENT application)
  elseif(APPLE)
    install(FILES ${AUTOMIX_ORT_RUNTIME_FILES} DESTINATION "AutoMixMaster.app/Contents/Frameworks" COMPONENT application)
  elseif(UNIX)
    install(FILES ${AUTOMIX_ORT_RUNTIME_FILES} DESTINATION lib COMPONENT application)
  endif()
endif()

if(APPLE)
  set(_automix_pl_dest "AutoMixMaster.app/Contents/MacOS/assets/phaselimiter")
else()
  set(_automix_pl_dest "assets/phaselimiter")
endif()

set(_automix_pl_install_root "")
if(AUTOMIX_PHASELIMITER_ROOT)
  set(_automix_pl_install_root "${AUTOMIX_PHASELIMITER_ROOT}")
elseif(EXISTS "${CMAKE_SOURCE_DIR}/assets/phaselimiter")
  set(_automix_pl_install_root "${CMAKE_SOURCE_DIR}/assets/phaselimiter")
endif()

if(_automix_pl_install_root)
  foreach(_dir bin resource licenses)
    if(EXISTS "${_automix_pl_install_root}/${_dir}")
      install(DIRECTORY "${_automix_pl_install_root}/${_dir}" DESTINATION "${_automix_pl_dest}" COMPONENT application)
    endif()
  endforeach()
  foreach(_file LICENSE README.md)
    if(EXISTS "${_automix_pl_install_root}/${_file}")
      install(FILES "${_automix_pl_install_root}/${_file}" DESTINATION "${_automix_pl_dest}" COMPONENT application)
    endif()
  endforeach()
endif()

# Hard guard for PhaseLimiter: fail the install if no native PhaseLimiter binary or reference file is packaged
if(ENABLE_PHASELIMITER)
  install(CODE "
    set(_pl_bin_check \"\${CMAKE_INSTALL_PREFIX}/${_automix_pl_dest}/bin/${_automix_native_bin_name}\")
    set(_pl_res_check \"\${CMAKE_INSTALL_PREFIX}/${_automix_pl_dest}/resource/mastering_reference.json\")
    if(NOT EXISTS \"\${_pl_bin_check}\" OR NOT EXISTS \"\${_pl_res_check}\")
      message(FATAL_ERROR \"Package has no native PhaseLimiter for ${_automix_host_pl_key}\")
    endif()
  " COMPONENT application)
endif()
if(EXISTS "${CMAKE_SOURCE_DIR}/assets/limiters")
  install(DIRECTORY "${CMAKE_SOURCE_DIR}/assets/limiters" DESTINATION assets COMPONENT application)
endif()
foreach(_doc NOTICE LICENSE README.md)
  if(EXISTS "${CMAKE_SOURCE_DIR}/${_doc}")
    install(FILES "${CMAKE_SOURCE_DIR}/${_doc}" DESTINATION . COMPONENT application)
  endif()
endforeach()

# macOS arm64: re-sign every bundled dylib with runtime hardened options before final packaging
if(APPLE)
  install(CODE [[
    file(GLOB _framework_dylibs "${CMAKE_INSTALL_PREFIX}/AutoMixMaster.app/Contents/Frameworks/*.dylib")
    foreach(_dylib IN LISTS _framework_dylibs)
      execute_process(COMMAND codesign --force --options runtime --sign - "${_dylib}")
    endforeach()
  ]] COMPONENT application)
endif()

# Hard guard for NOTICE's promises: fail the install if a model file or an
# NVIDIA CUDA library got into the package.
install(CODE [[
  file(GLOB_RECURSE _automix_forbidden
    "${CMAKE_INSTALL_PREFIX}/*.onnx" "${CMAKE_INSTALL_PREFIX}/*.onnx.data"
    "${CMAKE_INSTALL_PREFIX}/*cudart*.dll" "${CMAKE_INSTALL_PREFIX}/*cublas*.dll"
    "${CMAKE_INSTALL_PREFIX}/*cudnn*.dll" "${CMAKE_INSTALL_PREFIX}/*cufft*.dll"
    "${CMAKE_INSTALL_PREFIX}/*libcudart*.so*" "${CMAKE_INSTALL_PREFIX}/*libcublas*.so*"
    "${CMAKE_INSTALL_PREFIX}/*libcudnn*.so*" "${CMAKE_INSTALL_PREFIX}/*libcufft*.so*"
    "${CMAKE_INSTALL_PREFIX}/*libcudart*.dylib*" "${CMAKE_INSTALL_PREFIX}/*libcublas*.dylib*"
    "${CMAKE_INSTALL_PREFIX}/*libcudnn*.dylib*" "${CMAKE_INSTALL_PREFIX}/*libcufft*.dylib*")
  if(_automix_forbidden)
    message(FATAL_ERROR "Package must not contain model weights or NVIDIA CUDA libraries: ${_automix_forbidden}")
  endif()
]] COMPONENT application)

set(CPACK_PACKAGE_NAME "AutoMixMaster")
set(CPACK_PACKAGE_VENDOR "AutoMixMaster")

if(WIN32)
  set(CPACK_GENERATOR ZIP)
  if(AUTOMIX_HAS_NATIVE_ORT)
    get_filename_component(_ort_pkg_dir "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
    if(EXISTS "${_ort_pkg_dir}/onnxruntime_providers_cuda.dll" OR AUTOMIX_ORT_FLAVOR STREQUAL "cuda")
      set(CPACK_PACKAGE_FILE_NAME "AutoMixMaster-${PROJECT_VERSION}-win64-cuda")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64")
      set(CPACK_PACKAGE_FILE_NAME "AutoMixMaster-${PROJECT_VERSION}-winarm64")
    else()
      set(CPACK_PACKAGE_FILE_NAME "AutoMixMaster-${PROJECT_VERSION}-win64")
    endif()
  else()
    set(CPACK_PACKAGE_FILE_NAME "AutoMixMaster-${PROJECT_VERSION}-win64")
  endif()
elseif(APPLE)
  set(CPACK_GENERATOR TGZ)
  set(CPACK_PACKAGE_FILE_NAME "AutoMixMaster-${PROJECT_VERSION}-macos-arm64")
elseif(UNIX)
  set(CPACK_GENERATOR TGZ)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64")
    set(CPACK_PACKAGE_FILE_NAME "AutoMixMaster-${PROJECT_VERSION}-linux-arm64")
  else()
    set(CPACK_PACKAGE_FILE_NAME "AutoMixMaster-${PROJECT_VERSION}-linux-x64")
  endif()
endif()

set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY ON)
set(CPACK_COMPONENTS_ALL application)
set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
set(CPACK_COMPONENTS_GROUPING ALL_COMPONENTS_IN_ONE)
include(CPack)

