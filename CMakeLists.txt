cmake_minimum_required(VERSION 3.24)
project(AutoMixMaster VERSION 0.1.0 LANGUAGES C CXX)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 4.0)
  set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
endif()

if(WIN32 AND CMAKE_GENERATOR MATCHES "Visual Studio")
  if(NOT CMAKE_GENERATOR MATCHES "Visual Studio 18 2026")
    message(FATAL_ERROR "Windows Visual Studio builds require Visual Studio 2026 tools (generator: \"Visual Studio 18 2026\").")
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
  src/dsp/LookaheadLimiter.cpp
  src/dsp/MidSideProcessor.cpp
  src/dsp/MultibandProcessor.cpp
  src/dsp/SignalMath.cpp
  src/dsp/SoftClipper.cpp
  src/dsp/TruePeakDetector.cpp
  src/analysis/ArtifactRiskEstimator.cpp
  src/analysis/StemHealthAssistant.cpp
  src/analysis/StemAnalyzer.cpp
  src/automix/HeuristicAutoMixStrategy.cpp
  src/automaster/HeuristicAutoMasterStrategy.cpp
  src/automaster/OriginalMixReference.cpp
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
  src/ai/FeatureSchema.cpp
  src/ai/MasteringCompliance.cpp
  src/ai/OnnxModelInference.cpp
  src/ai/RtNeuralInference.cpp
  src/ai/StemSeparator.cpp
  src/ai/StemRoleClassifierAI.cpp
  src/util/StringUtils.cpp
  src/util/LameDownloader.cpp
  src/util/MetadataPolicy.cpp
  src/util/WavWriter.cpp
)

target_include_directories(automix_core PUBLIC src)

set(AUTOMIX_HAS_NATIVE_ORT OFF)
if(ENABLE_ONNX)
  find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATH_SUFFIXES onnxruntime/core/session
  )
  find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime)
  if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
    target_include_directories(automix_core PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(automix_core PUBLIC ${ONNXRUNTIME_LIBRARY})
    set(AUTOMIX_HAS_NATIVE_ORT ON)
  else()
    message(WARNING "ENABLE_ONNX is ON but native ONNX Runtime headers/library were not found. Falling back to deterministic ONNX adapter.")
  endif()
endif()

target_link_libraries(automix_core
  PUBLIC
    nlohmann_json::nlohmann_json
    juce::juce_events
    juce::juce_audio_formats
    juce::juce_dsp
)

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
    JUCE_USE_CURL=0
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

juce_add_gui_app(AutoMixMasterApp
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
)

target_compile_definitions(AutoMixMasterApp PRIVATE
  JUCE_WEB_BROWSER=0
  JUCE_USE_CURL=0
)

if(EXISTS "${CMAKE_SOURCE_DIR}/assets/phaselimiter")
  add_custom_command(TARGET AutoMixMasterApp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:AutoMixMasterApp>/assets"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/assets/phaselimiter"
            "$<TARGET_FILE_DIR:AutoMixMasterApp>/assets/phaselimiter"
    COMMENT "Copying bundled PhaseLimiter assets next to AutoMixMaster executable"
    VERBATIM
  )
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
    tests/unit/LoudnessMeterTests.cpp
    tests/unit/LookaheadLimiterTests.cpp
    tests/unit/MasteringModulesTests.cpp
    tests/unit/AudioPreviewEngineTests.cpp
    tests/unit/TransportControllerTests.cpp
    tests/unit/BatchProcessingTests.cpp
    tests/unit/OfflineRenderPipelineTests.cpp
    tests/unit/OnnxModelInferenceTests.cpp
    tests/unit/StemSeparatorTests.cpp
    tests/regression/RegressionHarness.cpp
    tests/regression/RegressionHarnessTests.cpp
    src/app/controllers/ModelController.cpp
    src/app/controllers/ImportController.cpp
    src/app/controllers/ExportController.cpp
    src/app/controllers/ProcessingController.cpp
    src/app/controllers/SessionController.cpp
    src/app/controllers/PreviewController.cpp
    src/app/controllers/ProfileController.cpp
    src/app/controllers/OriginalMixController.cpp
  )

  target_include_directories(automix_tests PRIVATE src tests/regression)
  target_link_libraries(automix_tests PRIVATE automix_core Catch2::Catch2WithMain)
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
endif()
