cmake_minimum_required(VERSION 3.24)
project(AutoMixMaster VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_PHASELIMITER "Enable optional PhaseLimiter renderer" OFF)
option(ENABLE_ONNX "Enable optional ONNX model inference adapter" OFF)
option(ENABLE_RTNEURAL "Enable optional RTNeural model inference adapter" OFF)
option(BUILD_TESTING "Build tests" ON)
option(BUILD_TOOLS "Build developer tools" ON)

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

add_library(automix_core
  src/domain/Bus.cpp
  src/domain/JsonSerialization.cpp
  src/domain/MasterPlan.cpp
  src/domain/MixPlan.cpp
  src/domain/RenderSettings.cpp
  src/domain/Session.cpp
  src/domain/Stem.cpp
  src/domain/StemOrigin.cpp
  src/engine/AudioBuffer.cpp
  src/engine/AudioFileIO.cpp
  src/engine/AudioResampler.cpp
  src/engine/OfflineRenderPipeline.cpp
  src/engine/ResidualBlendProcessor.cpp
  src/engine/SessionRepository.cpp
  src/dsp/SignalMath.cpp
  src/analysis/ArtifactRiskEstimator.cpp
  src/analysis/StemAnalyzer.cpp
  src/automix/HeuristicAutoMixStrategy.cpp
  src/automaster/HeuristicAutoMasterStrategy.cpp
  src/automaster/OriginalMixReference.cpp
  src/renderers/BuiltInRenderer.cpp
  src/renderers/PhaseLimiterDiscovery.cpp
  src/renderers/PhaseLimiterRenderer.cpp
  src/renderers/RendererFactory.cpp
  src/ai/ModelPackLoader.cpp
  src/ai/ModelManager.cpp
  src/ai/ModelStrategy.cpp
  src/ai/AutoMixStrategyAI.cpp
  src/ai/AutoMasterStrategyAI.cpp
  src/ai/MasteringCompliance.cpp
  src/ai/RtNeuralInference.cpp
  src/ai/StemRoleClassifierAI.cpp
  src/util/WavWriter.cpp
)

if(ENABLE_ONNX)
  target_sources(automix_core PRIVATE src/ai/OnnxModelInference.cpp)
endif()

target_include_directories(automix_core PUBLIC src)

target_link_libraries(automix_core
  PUBLIC
    nlohmann_json::nlohmann_json
    juce::juce_audio_formats
    juce::juce_dsp
)

target_compile_definitions(automix_core
  PUBLIC
    APP_VERSION="${PROJECT_VERSION}"
    SESSION_SCHEMA_VERSION=2
    $<$<BOOL:${ENABLE_PHASELIMITER}>:ENABLE_PHASELIMITER=1>
    $<$<BOOL:${ENABLE_ONNX}>:ENABLE_ONNX=1>
    $<$<BOOL:${ENABLE_RTNEURAL}>:ENABLE_RTNEURAL=1>
)

if(MSVC)
  target_compile_options(automix_core PRIVATE /W4 /permissive-)
else()
  target_compile_options(automix_core PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
endif()

juce_add_gui_app(AutoMixMasterApp
  PRODUCT_NAME "AutoMixMaster"
  COMPANY_NAME "AutoMixMaster"
)

target_sources(AutoMixMasterApp PRIVATE
  src/app/Main.cpp
  src/app/MainComponent.cpp
)

target_include_directories(AutoMixMasterApp PRIVATE src)

target_link_libraries(AutoMixMasterApp
  PRIVATE
    automix_core
    juce::juce_gui_extra
    juce::juce_audio_formats
    juce::juce_dsp
)

target_compile_definitions(AutoMixMasterApp PRIVATE
  JUCE_WEB_BROWSER=0
  JUCE_USE_CURL=0
)

if(BUILD_TOOLS)
  add_executable(automix_dev_tools tools/dev_tools.cpp)
  target_include_directories(automix_dev_tools PRIVATE src)
  target_link_libraries(automix_dev_tools PRIVATE automix_core)
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
    tests/unit/SessionSerializationTests.cpp
    tests/unit/AudioIoTests.cpp
    tests/unit/AnalysisTests.cpp
    tests/unit/StemOriginSafetyTests.cpp
    tests/unit/ResidualBlendTests.cpp
    tests/unit/PhaseLimiterDiscoveryTests.cpp
    tests/unit/PhaseLimiterRendererTests.cpp
    tests/unit/AutoMasterTests.cpp
    tests/unit/AiExtensionTests.cpp
    tests/regression/RegressionHarness.cpp
    tests/regression/RegressionHarnessTests.cpp
  )

  target_include_directories(automix_tests PRIVATE src tests/regression)
  target_link_libraries(automix_tests PRIVATE automix_core Catch2::Catch2WithMain)
  target_compile_definitions(automix_tests PRIVATE AUTOMIX_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

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
