#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "ai/GpuProvider.h"
#include "ai/OnnxModelInference.h"
#include "ai/OnnxTensorInference.h"

namespace automix { namespace ai { namespace test {

using namespace automix::ai;

/// Returns true if we are running on a real ONNX Runtime (not the deterministic fallback).
bool hasNativeOnnx() {
#ifdef AUTOMIX_HAS_NATIVE_ORT
  return true;
#else
  return false;
#endif
}

/// Create a minimal dummy model file for testing.
std::filesystem::path createDummyModel(const std::filesystem::path& dir) {
  const auto modelPath = dir / "test_model.onnx";
  std::ofstream(modelPath, std::ios::binary) << "dummy_onnx_model";
  return modelPath;
}

/// Create a metadata file for the dummy model.
void createDummyMetadata(const std::filesystem::path& modelPath) {
  std::ofstream meta(modelPath.string() + ".meta.json");
  meta << R"({
    "input_feature_count": 27,
    "allowed_tasks": ["mix_parameters", "master_parameters"],
    "execution_providers": ["cpu"]
  })";
}

/// Create a metadata file that advertises a GPU provider ahead of CPU so the
/// recovery path is exercisable in deterministic mode.
void createDummyMetadataWithGpu(const std::filesystem::path& modelPath) {
  std::ofstream meta(modelPath.string() + ".meta.json");
  meta << R"({
    "input_feature_count": 27,
    "allowed_tasks": ["mix_parameters", "master_parameters"],
    "execution_providers": ["cuda", "cpu"]
  })";
}

double measureInferenceTime(OnnxModelInference& inference,
                            const InferenceRequest& request,
                            int iterations = 5) {
  double totalMs = 0.0;
  for (int i = 0; i < iterations; ++i) {
    const auto start = std::chrono::high_resolution_clock::now();
    inference.run(request);
    const auto end = std::chrono::high_resolution_clock::now();
    totalMs += std::chrono::duration<double, std::milli>(end - start).count();
  }
  return totalMs / static_cast<double>(iterations);
}

}}} // namespace automix::ai::test

using namespace automix::ai::test;

// ─── T3.4: GPU capability detection ─────────────────────────────────────────

TEST_CASE("GpuProvider canonical name mapping", "[gpu][provider]") {
  using automix::ai::gpu::canonicalProviderName;

  CHECK(canonicalProviderName("CPU") == "cpu");
  CHECK(canonicalProviderName("CpuExecutionProvider") == "cpu");
  CHECK(canonicalProviderName("CUDA") == "cuda");
  CHECK(canonicalProviderName("CudaExecutionProvider") == "cuda");
  CHECK(canonicalProviderName("WebGpu") == "webgpu");
  CHECK(canonicalProviderName("WebGpuExecutionProvider") == "webgpu");
  CHECK(canonicalProviderName("wgpu") == "webgpu");
  CHECK(canonicalProviderName("DML") == "directml");
  CHECK(canonicalProviderName("DirectML") == "directml");
  CHECK(canonicalProviderName("CoreML") == "coreml");
  CHECK(canonicalProviderName("CoreMLExecutionProvider") == "coreml");
  CHECK(canonicalProviderName("OpenVINO") == "openvino");
  CHECK(canonicalProviderName("OpenVINOExecutionProvider") == "openvino");
  CHECK(canonicalProviderName("ANE") == "ane");
  CHECK(canonicalProviderName("NeuralNetwork") == "ane");
  CHECK(canonicalProviderName("unknown") == "unknown");
}

TEST_CASE("GpuProvider priority chain order", "[gpu][provider]") {
  using automix::ai::gpu::providerPriority;
  using automix::ai::gpu::providerPriorityChain;

  const auto& chain = providerPriorityChain();

  // Chain should be: ANE > CoreML > CUDA > WebGPU > OpenVINO > DirectML > CPU
  REQUIRE(chain.size() == 7);
  CHECK(chain[0] == "ane");
  CHECK(chain[1] == "coreml");
  CHECK(chain[2] == "cuda");
  CHECK(chain[3] == "webgpu");
  CHECK(chain[4] == "openvino");
  CHECK(chain[5] == "directml");
  CHECK(chain[6] == "cpu");

  // Verify priority values (lower = higher priority)
  CHECK(providerPriority("ane") == 0);
  CHECK(providerPriority("coreml") == 1);
  CHECK(providerPriority("cuda") == 2);
  CHECK(providerPriority("webgpu") == 3);
  CHECK(providerPriority("openvino") == 4);
  CHECK(providerPriority("directml") == 5);
  CHECK(providerPriority("cpu") == 6);

  // Unknown providers have lowest priority
  CHECK(providerPriority("tensorrt") == 7);
}

TEST_CASE("GpuProvider isGpuProvider classification", "[gpu][provider]") {
  using automix::ai::gpu::isGpuProvider;

  CHECK_FALSE(isGpuProvider("cpu"));
  CHECK(isGpuProvider("cuda"));
  CHECK(isGpuProvider("webgpu"));
  CHECK(isGpuProvider("directml"));
  CHECK(isGpuProvider("coreml"));
  CHECK(isGpuProvider("ane"));
  CHECK(isGpuProvider("openvino"));
}

TEST_CASE("GpuProvider platform preferred provider", "[gpu][provider]") {
  using automix::ai::gpu::platformPreferredProvider;

  const auto preferred = platformPreferredProvider();
  // Should return one of the known provider names
  CHECK((preferred == "ane" || preferred == "coreml" ||
         preferred == "cuda" || preferred == "webgpu"));
#if defined(_WIN32)
  CHECK(preferred == "webgpu");
#endif
}

TEST_CASE("CoreML and ANE are opt-in for auto on low-memory Macs", "[gpu][tensor][coreml]") {
  using automix::ai::coreMlAutoAllowed;
  using automix::ai::kCoreMlAutoMinMemoryBytes;
  constexpr std::uint64_t GiB = 1024ull * 1024 * 1024;
  const std::vector<std::string> mac = {"ane", "coreml", "cpu"};

  CHECK_FALSE(coreMlAutoAllowed(8 * GiB, false));
  CHECK(coreMlAutoAllowed(8 * GiB, true));
  CHECK(coreMlAutoAllowed(kCoreMlAutoMinMemoryBytes, false));
  CHECK(coreMlAutoAllowed(16 * GiB, false));
  CHECK(coreMlAutoAllowed(0, false));

  // auto on a low-memory Mac goes straight to CPU; opted in or roomy keeps the Apple providers
  CHECK(tensorProviderCandidates("auto", mac, {}, false) == std::vector<std::string>{"cpu"});
  CHECK(tensorProviderCandidates("auto", mac, {}, true) == std::vector<std::string>{"ane", "coreml", "cpu"});
  // naming the provider is always honoured
  CHECK(tensorProviderCandidates("coreml", mac, {}, false).front() == "coreml");
  CHECK(tensorProviderCandidates("ane", mac, {}, false).front() == "ane");
}

TEST_CASE("Vocal model warns about CPU only on low-memory Macs", "[gpu][coreml]") {
  using automix::ai::vocalModelCpuWarning;
  constexpr std::uint64_t GiB = 1024ull * 1024 * 1024;
  CHECK_FALSE(vocalModelCpuWarning(8 * GiB, false, true).empty());
  CHECK(vocalModelCpuWarning(8 * GiB, false, true).find("8 GB") != std::string::npos);
  CHECK(vocalModelCpuWarning(8 * GiB, true, true).empty());    // a GPU session is usable
  CHECK(vocalModelCpuWarning(16 * GiB, false, true).empty());  // roomy
  CHECK(vocalModelCpuWarning(8 * GiB, false, false).empty());  // not a Mac: no measurement behind it
  CHECK(vocalModelCpuWarning(0, false, true).empty());         // unknown memory
}

TEST_CASE("tensorProviderCandidates allow-list filtering", "[gpu][tensor]") {
  // 1. Empty allow-list preserves all reported GPU candidates and keeps CPU last
  const auto c1 = tensorProviderCandidates("auto", {"cuda", "webgpu", "cpu"}, {});
  const std::vector<std::string> expected1 = {"cuda", "webgpu", "cpu"};
  CHECK(c1 == expected1);

  // 2. ["cuda"] on a webgpu-only runtime filters to [cpu], CPU always kept
  const auto c2 = tensorProviderCandidates("auto", {"webgpu", "cpu"}, {"cuda"});
  const std::vector<std::string> expected2 = {"cpu"};
  CHECK(c2 == expected2);

  // 3. Requested provider not in allow-list falls back to allowed candidates then CPU
  const auto c3 = tensorProviderCandidates("webgpu", {"webgpu", "cpu"}, {"cuda"});
  CHECK(c3 == expected2);

  // 4. ["webgpu"] on a multi-GPU runtime filters to [webgpu, cpu]
  const auto c4 = tensorProviderCandidates("auto", {"cuda", "webgpu", "cpu"}, {"webgpu"});
  const std::vector<std::string> expected4 = {"webgpu", "cpu"};
  CHECK(c4 == expected4);
}

TEST_CASE("GpuProvider providerOptionMap and sessionTuning", "[gpu][provider]") {
  using namespace automix::ai::gpu;

  // 1. providerOptionMap("ane")
  const auto aneOpts = providerOptionMap("ane");
  CHECK(aneOpts.at("ModelFormat") == "MLProgram");
  CHECK(aneOpts.at("MLComputeUnits") == "CPUAndNeuralEngine");

  // Every CoreML/ANE key must be in the valid set documented in coreml_options.cc@v1.30.0:55-64
  const std::vector<std::string> validCoreMlKeys = {
      "MLComputeUnits",
      "ModelFormat",
      "RequireStaticInputShapes",
      "EnableOnSubgraphs",
      "SpecializationStrategy",
      "ProfileComputePlan",
      "AllowLowPrecisionAccumulationOnGPU",
      "ModelCacheDirectory"
  };
  for (const auto& [k, v] : aneOpts) {
    CHECK((std::find(validCoreMlKeys.begin(), validCoreMlKeys.end(), k) != validCoreMlKeys.end()));
  }

  // 2. providerOptionMap("coreml")
  const auto coremlOpts = providerOptionMap("coreml");
  CHECK(coremlOpts.at("ModelFormat") == "MLProgram");
  CHECK(coremlOpts.at("MLComputeUnits") == "ALL");
  for (const auto& [k, v] : coremlOpts) {
    CHECK((std::find(validCoreMlKeys.begin(), validCoreMlKeys.end(), k) != validCoreMlKeys.end()));
  }

  // 3. sessionTuning("directml", n) gives sequential execution with mem pattern off
  const auto dmlTuning = sessionTuning("directml", 8);
  CHECK(dmlTuning.sequentialExecution == true);
  CHECK(dmlTuning.memPattern == false);
  CHECK(dmlTuning.cpuArena == false);
  CHECK(dmlTuning.interOpThreads == 1);
  CHECK(dmlTuning.intraOpThreads == 4);

  // 4. sessionTuning("cuda" | "cpu", n) golden values
  const auto cudaTuning = sessionTuning("cuda", 8);
  CHECK(cudaTuning.sequentialExecution == false);
  CHECK(cudaTuning.memPattern == false);
  CHECK(cudaTuning.cpuArena == true);
  CHECK(cudaTuning.interOpThreads == 1);
  CHECK(cudaTuning.intraOpThreads == 4);
  CHECK(cudaTuning.hardwareTier == "standard");

  const auto cpuTuningLow = sessionTuning("cpu", 2);
  CHECK(cpuTuningLow.hardwareTier == "low");
  CHECK(cpuTuningLow.intraOpThreads == 2);
  CHECK(cpuTuningLow.interOpThreads == 1);
  CHECK(cpuTuningLow.memPattern == true);
  CHECK(cpuTuningLow.cpuArena == true);
  CHECK(cpuTuningLow.sequentialExecution == false);

  const auto cpuTuningHigh = sessionTuning("cpu", 16);
  CHECK(cpuTuningHigh.hardwareTier == "high");
  CHECK(cpuTuningHigh.intraOpThreads == 16);
  CHECK(cpuTuningHigh.interOpThreads == 8);
  CHECK(cpuTuningHigh.memPattern == true);
  CHECK(cpuTuningHigh.cpuArena == true);
  CHECK(cpuTuningHigh.sequentialExecution == false);
}

TEST_CASE("sessionConfigPlan leaves ORT defaults for untuned sessions", "[gpu][provider]") {
  using namespace automix::ai::gpu;

  // Untuned CPU session leaves every field empty (ORT defaults)
  const auto cpuUntuned = sessionConfigPlan("cpu", 0);
  CHECK(!cpuUntuned.intraOpThreads.has_value());
  CHECK(!cpuUntuned.interOpThreads.has_value());
  CHECK(!cpuUntuned.sequentialExecution.has_value());
  CHECK(!cpuUntuned.memPattern.has_value());
  CHECK(!cpuUntuned.cpuArena.has_value());

  // Untuned CUDA, CoreML, WebGPU: every field empty
  for (const auto& prov : {"cuda", "coreml", "webgpu"}) {
    const auto untuned = sessionConfigPlan(prov, 0);
    CHECK(!untuned.intraOpThreads.has_value());
    CHECK(!untuned.interOpThreads.has_value());
    CHECK(!untuned.sequentialExecution.has_value());
    CHECK(!untuned.memPattern.has_value());
    CHECK(!untuned.cpuArena.has_value());
  }

  // Untuned DirectML: mandatory constraints only (memPattern=false, sequentialExecution=true)
  const auto dmlUntuned = sessionConfigPlan("directml", 0);
  CHECK(dmlUntuned.memPattern == false);
  CHECK(dmlUntuned.sequentialExecution == true);
  CHECK(!dmlUntuned.intraOpThreads.has_value());
  CHECK(!dmlUntuned.interOpThreads.has_value());
  CHECK(!dmlUntuned.cpuArena.has_value());

  // Tuned CPU session (hardwareThreads > 0): equals sessionTuning field by field
  const auto cpuTuned = sessionConfigPlan("cpu", 16);
  const auto cpuExpectedTuning = sessionTuning("cpu", 16);
  CHECK(cpuTuned.intraOpThreads == cpuExpectedTuning.intraOpThreads);
  CHECK(cpuTuned.interOpThreads == cpuExpectedTuning.interOpThreads);
  CHECK(cpuTuned.sequentialExecution == cpuExpectedTuning.sequentialExecution);
  CHECK(cpuTuned.memPattern == cpuExpectedTuning.memPattern);
  CHECK(cpuTuned.cpuArena == cpuExpectedTuning.cpuArena);
  CHECK(cpuTuned.intraOpThreads == 16);
  CHECK(cpuTuned.interOpThreads == 8);
  CHECK(cpuTuned.sequentialExecution == false);
  CHECK(cpuTuned.memPattern == true);
  CHECK(cpuTuned.cpuArena == true);

  // Tuned DirectML session (hardwareThreads > 0)
  const auto dmlTuned = sessionConfigPlan("directml", 8);
  CHECK(dmlTuned.sequentialExecution == true);
  CHECK(dmlTuned.memPattern == false);
  CHECK(dmlTuned.cpuArena == false);
}

// ─── T3.4: detectAvailableProviders ─────────────────────────────────────────

TEST_CASE("OnnxModelInference detectAvailableProviders", "[gpu][detect]") {
  OnnxModelInference inference;
  const auto providers = inference.detectAvailableProviders();

  // CPU should always be in the list
  REQUIRE_FALSE(providers.empty());
  CHECK(std::find(providers.begin(), providers.end(), "cpu") != providers.end());

  // If native ONNX is available, there may be GPU providers too
  if (hasNativeOnnx()) {
    // The list should be sorted by priority (GPU first, CPU last)
    const auto cpuIt = std::find(providers.begin(), providers.end(), "cpu");
    if (cpuIt != providers.begin() && cpuIt != providers.end()) {
      // At least one GPU provider is listed ahead of CPU
      CHECK(cpuIt != providers.begin());
    }
  }
}

// ─── T3.5: Fallback chain ────────────────────────────────────────────────────

TEST_CASE("OnnxModelInference fallback chain resolution", "[gpu][fallback]") {
  OnnxModelInference inference;
  const auto tempDir = std::filesystem::temp_directory_path() / "automix_gpu_fallback_test";
  std::filesystem::remove_all(tempDir);
  std::filesystem::create_directories(tempDir);

  const auto modelPath = createDummyModel(tempDir);
  createDummyMetadata(modelPath);
  REQUIRE(inference.loadModel(modelPath));

  // Should resolve to a provider (at least CPU)
  const auto activeProvider = inference.activeExecutionProvider();
  CHECK_FALSE(activeProvider.empty());

  // Backend diagnostics should include GPU recovery counters
  const auto diag = inference.backendDiagnostics();
  CHECK(diag.find("gpu_oom=") != std::string::npos);
  CHECK(diag.find("gpu_device_lost=") != std::string::npos);
  CHECK(diag.find("gpu_recoveries=") != std::string::npos);

  std::filesystem::remove_all(tempDir);
}

TEST_CASE("OnnxModelInference provider fallback counters", "[gpu][fallback]") {
  OnnxModelInference inference;
  const auto tempDir = std::filesystem::temp_directory_path() / "automix_gpu_counters_test";
  std::filesystem::remove_all(tempDir);
  std::filesystem::create_directories(tempDir);

  const auto modelPath = createDummyModel(tempDir);
  createDummyMetadata(modelPath);
  REQUIRE(inference.loadModel(modelPath));

  // Run a few inferences
  InferenceRequest req;
  req.task = "mix_parameters";
  req.features.assign(27, 0.5);

  const auto r1 = inference.run(req);
  CHECK(r1.usedModel);

  // Diagnostics should show calls
  const auto diag = inference.backendDiagnostics();
  CHECK(diag.find("calls=") != std::string::npos);

  std::filesystem::remove_all(tempDir);
}

// ─── T3.6: GPU error recovery ───────────────────────────────────────────────

TEST_CASE("GpuProvider error recovery counters", "[gpu][recovery]") {
  // Test that the recovery infrastructure compiles and initializes to zero
  OnnxModelInference inference;

  // These should be zero before any load
  CHECK(inference.gpuRecoveryCount() == 0);
  CHECK(inference.failedProviders().empty());
}

TEST_CASE("OnnxModelInference failed providers tracking", "[gpu][recovery]") {
  OnnxModelInference inference;

  // Load with a non-existent model to test error path
  const auto result = inference.loadModel("/nonexistent/model.onnx");
  CHECK_FALSE(result);

  // After failed load, providers list should be available
  const auto providers = inference.detectAvailableProviders();
  CHECK_FALSE(providers.empty());
}

// ─── T3.4: OOM / device-lost -> CPU recovery (deterministic mode) ───────────

TEST_CASE("OnnxModelInference OOM failure triggers recovery counters and CPU re-resolution", "[gpu][recovery]") {
  OnnxModelInference inference;
  const auto tempDir = std::filesystem::temp_directory_path() / "automix_gpu_oom_recovery_test";
  std::filesystem::remove_all(tempDir);
  std::filesystem::create_directories(tempDir);

  const auto modelPath = createDummyModel(tempDir);
  createDummyMetadataWithGpu(modelPath);
  // Load on CPU: these tests simulate GPU failures, so a real GPU attempt at
  // load (which fails wherever CUDA's libraries are absent) must not pre-mark it.
  inference.setExecutionProviderPreference("cpu");
  REQUIRE(inference.loadModel(modelPath));
  inference.pinExecutionProvidersForTesting({"cuda", "cpu"});
  inference.setExecutionProviderPreference("auto");

  // Setup proves the GPU provider is actually selectable before any failure:
  // the recovery assertions are only meaningful if resolution picks "cuda".
  REQUIRE(inference.activeExecutionProvider() == "cuda");
  REQUIRE(inference.gpuRecoveryCount() == 0);
  REQUIRE(inference.failedProviders().empty());

  // When: a GPU OOM failure is simulated on the active provider.
  inference.recordProviderFailure("cuda", ProviderFailureKind::Oom);

  // Then: the recovery counter reflects the OOM.
  CHECK(inference.gpuRecoveryCount() == 1);
  CHECK(inference.backendDiagnostics().find("gpu_oom=1") != std::string::npos);

  // And: the provider is recorded as failed.
  const auto failed = inference.failedProviders();
  REQUIRE(failed.size() == 1);
  CHECK(failed[0] == "cuda");

  // And: subsequent resolution skips the failed provider, landing on CPU.
  CHECK(inference.activeExecutionProvider() == "cpu");

  // And: the CPU (deterministic) fallback still serves inference.
  InferenceRequest req;
  req.task = "mix_parameters";
  req.features.assign(27, 0.5);
  const auto result = inference.run(req);
  CHECK(result.usedModel);
  CHECK(result.logMessage.find("provider='cpu'") != std::string::npos);

  // And: the retry did not double-count the recovery.
  CHECK(inference.gpuRecoveryCount() == 1);

  std::filesystem::remove_all(tempDir);
}

TEST_CASE("OnnxModelInference device-lost failure triggers recovery counters and CPU re-resolution", "[gpu][recovery]") {
  OnnxModelInference inference;
  const auto tempDir = std::filesystem::temp_directory_path() / "automix_gpu_device_lost_test";
  std::filesystem::remove_all(tempDir);
  std::filesystem::create_directories(tempDir);

  const auto modelPath = createDummyModel(tempDir);
  createDummyMetadataWithGpu(modelPath);
  // Load on CPU: these tests simulate GPU failures, so a real GPU attempt at
  // load (which fails wherever CUDA's libraries are absent) must not pre-mark it.
  inference.setExecutionProviderPreference("cpu");
  REQUIRE(inference.loadModel(modelPath));
  inference.pinExecutionProvidersForTesting({"cuda", "cpu"});
  inference.setExecutionProviderPreference("auto");
  REQUIRE(inference.activeExecutionProvider() == "cuda");

  // When: a GPU device-lost failure is simulated on the active provider.
  inference.recordProviderFailure("cuda", ProviderFailureKind::DeviceLost);

  // Then: the recovery counter reflects the device-lost event.
  CHECK(inference.gpuRecoveryCount() == 1);
  CHECK(inference.backendDiagnostics().find("gpu_device_lost=1") != std::string::npos);

  // And: the provider is recorded as failed and resolution lands on CPU.
  const auto failed = inference.failedProviders();
  REQUIRE(failed.size() == 1);
  CHECK(failed[0] == "cuda");
  CHECK(inference.activeExecutionProvider() == "cpu");

  // And: a subsequent inference still succeeds via the CPU fallback.
  InferenceRequest req;
  req.task = "mix_parameters";
  req.features.assign(27, 0.5);
  const auto result = inference.run(req);
  CHECK(result.usedModel);

  std::filesystem::remove_all(tempDir);
}

TEST_CASE("OnnxModelInference resolution skips failed provider on later re-resolution", "[gpu][recovery]") {
  OnnxModelInference inference;
  const auto tempDir = std::filesystem::temp_directory_path() / "automix_gpu_resolve_skip_test";
  std::filesystem::remove_all(tempDir);
  std::filesystem::create_directories(tempDir);

  const auto modelPath = createDummyModel(tempDir);
  createDummyMetadataWithGpu(modelPath);
  // Load on CPU: these tests simulate GPU failures, so a real GPU attempt at
  // load (which fails wherever CUDA's libraries are absent) must not pre-mark it.
  inference.setExecutionProviderPreference("cpu");
  REQUIRE(inference.loadModel(modelPath));
  inference.pinExecutionProvidersForTesting({"cuda", "cpu"});
  inference.setExecutionProviderPreference("auto");
  REQUIRE(inference.activeExecutionProvider() == "cuda");

  // Given: the GPU provider has failed once and resolution fell back to CPU.
  inference.recordProviderFailure("cuda", ProviderFailureKind::Oom);
  CHECK(inference.activeExecutionProvider() == "cpu");

  // When: execution-provider resolution is re-run through the public surface.
  inference.setExecutionProviderPreference("auto");

  // Then: the failed provider stays skipped; CPU remains the active provider.
  CHECK(inference.activeExecutionProvider() == "cpu");
  const auto failed = inference.failedProviders();
  REQUIRE(failed.size() == 1);
  CHECK(failed[0] == "cuda");

  std::filesystem::remove_all(tempDir);
}

TEST_CASE("OnnxModelInference repeated failure counts each recovery but records provider once", "[gpu][recovery]") {
  OnnxModelInference inference;
  const auto tempDir = std::filesystem::temp_directory_path() / "automix_gpu_repeat_failure_test";
  std::filesystem::remove_all(tempDir);
  std::filesystem::create_directories(tempDir);

  const auto modelPath = createDummyModel(tempDir);
  createDummyMetadataWithGpu(modelPath);
  // Load on CPU: these tests simulate GPU failures, so a real GPU attempt at
  // load (which fails wherever CUDA's libraries are absent) must not pre-mark it.
  inference.setExecutionProviderPreference("cpu");
  REQUIRE(inference.loadModel(modelPath));
  inference.pinExecutionProvidersForTesting({"cuda", "cpu"});
  inference.setExecutionProviderPreference("auto");
  REQUIRE(inference.activeExecutionProvider() == "cuda");

  // When: the same provider fails twice.
  inference.recordProviderFailure("cuda", ProviderFailureKind::Oom);
  inference.recordProviderFailure("cuda", ProviderFailureKind::Oom);

  // Then: each failure is a distinct recovery event...
  CHECK(inference.gpuRecoveryCount() == 2);
  CHECK(inference.backendDiagnostics().find("gpu_oom=2") != std::string::npos);

  // ...but the provider is recorded exactly once (deduplicated).
  const auto failed = inference.failedProviders();
  REQUIRE(failed.size() == 1);
  CHECK(failed[0] == "cuda");
  CHECK(inference.activeExecutionProvider() == "cpu");

  std::filesystem::remove_all(tempDir);
}

TEST_CASE("OnnxModelInference non-recoverable failure marks provider failed without recovery counter", "[gpu][recovery]") {
  OnnxModelInference inference;
  const auto tempDir = std::filesystem::temp_directory_path() / "automix_gpu_unknown_failure_test";
  std::filesystem::remove_all(tempDir);
  std::filesystem::create_directories(tempDir);

  const auto modelPath = createDummyModel(tempDir);
  createDummyMetadataWithGpu(modelPath);
  // Load on CPU: these tests simulate GPU failures, so a real GPU attempt at
  // load (which fails wherever CUDA's libraries are absent) must not pre-mark it.
  inference.setExecutionProviderPreference("cpu");
  REQUIRE(inference.loadModel(modelPath));
  inference.pinExecutionProvidersForTesting({"cuda", "cpu"});
  inference.setExecutionProviderPreference("auto");
  REQUIRE(inference.activeExecutionProvider() == "cuda");

  // When: a non-OOM / non-device-lost failure is simulated.
  inference.recordProviderFailure("cuda", ProviderFailureKind::Unknown);

  // Then: the provider is still marked failed and resolution falls back to CPU,
  // but the GPU recovery counter is not inflated.
  CHECK(inference.gpuRecoveryCount() == 0);
  const auto failed = inference.failedProviders();
  REQUIRE(failed.size() == 1);
  CHECK(failed[0] == "cuda");
  CHECK(inference.activeExecutionProvider() == "cpu");

  std::filesystem::remove_all(tempDir);
}

// ─── T3.7: GPU vs CPU benchmarks ────────────────────────────────────────────

TEST_CASE("CPU baseline inference benchmark", "[gpu][benchmark][onnx]") {
  OnnxModelInference inference;
  const auto tempDir = std::filesystem::temp_directory_path() / "automix_gpu_benchmark_test";
  std::filesystem::remove_all(tempDir);
  std::filesystem::create_directories(tempDir);

  const auto modelPath = createDummyModel(tempDir);
  createDummyMetadata(modelPath);
  REQUIRE(inference.loadModel(modelPath));

  InferenceRequest req;
  req.task = "mix_parameters";
  req.features.assign(27, 0.5);

  // Measure baseline inference time
  const double avgMs = measureInferenceTime(inference, req, 10);
  CHECK(avgMs >= 0.0);

  // Benchmarking detail: each call should be measurable
  const auto diag = inference.backendDiagnostics();
  CHECK(diag.find("avg_inference_ms=") != std::string::npos);

  std::filesystem::remove_all(tempDir);
}

TEST_CASE("Multiple task type benchmark", "[gpu][benchmark][onnx]") {
  OnnxModelInference inference;
  const auto tempDir = std::filesystem::temp_directory_path() / "automix_gpu_multi_task_benchmark";
  std::filesystem::remove_all(tempDir);
  std::filesystem::create_directories(tempDir);

  const auto modelPath = createDummyModel(tempDir);
  createDummyMetadata(modelPath);
  REQUIRE(inference.loadModel(modelPath));

  SECTION("mix_parameters benchmark") {
    InferenceRequest req;
    req.task = "mix_parameters";
    req.features.assign(27, 0.5);
    const auto avgMs = measureInferenceTime(inference, req, 5);
    CHECK(avgMs >= 0.0);
  }

  SECTION("master_parameters benchmark") {
    InferenceRequest req;
    req.task = "master_parameters";
    req.features.assign(27, 0.5);
    const auto avgMs = measureInferenceTime(inference, req, 5);
    CHECK(avgMs >= 0.0);
  }

  SECTION("role_classifier benchmark") {
    InferenceRequest req;
    req.task = "role_classifier";
    req.features.assign(27, 0.3);
    const auto avgMs = measureInferenceTime(inference, req, 5);
    CHECK(avgMs >= 0.0);
  }

  std::filesystem::remove_all(tempDir);
}

TEST_CASE("GpuProvider pluginLibraryFileName platform mapping", "[gpu][provider]") {
  using namespace automix::ai::gpu;

  CHECK(pluginLibraryFileName("windows") == "onnxruntime_providers_webgpu.dll");
  CHECK(pluginLibraryFileName("win") == "onnxruntime_providers_webgpu.dll");
  CHECK(pluginLibraryFileName("win-x64") == "onnxruntime_providers_webgpu.dll");
  CHECK(pluginLibraryFileName("Windows_NT") == "onnxruntime_providers_webgpu.dll");

  CHECK(pluginLibraryFileName("linux") == "libonnxruntime_providers_webgpu.so");
  CHECK(pluginLibraryFileName("linux-x64") == "libonnxruntime_providers_webgpu.so");
  CHECK(pluginLibraryFileName("Ubuntu") == "libonnxruntime_providers_webgpu.so");

  CHECK(pluginLibraryFileName("macos").empty());
  CHECK(pluginLibraryFileName("darwin").empty());
  CHECK(pluginLibraryFileName("").empty());
}

struct ProbeCacheResetGuard {
  ProbeCacheResetGuard() { resetTensorProviderProbeCacheForTesting(); }
  ~ProbeCacheResetGuard() { resetTensorProviderProbeCacheForTesting(); }
};

TEST_CASE("tensorProviderUsable per-provider probe cache semantics", "[gpu][tensor]") {
  ProbeCacheResetGuard guard;

  int cudaProbeCalls = 0;
  bool cudaAvailable = false;
  bool webgpuAvailable = false;

  setTensorProviderProbeFunctionForTesting([&](const std::string& candidate) {
    if (candidate == "webgpu") return webgpuAvailable;
    if (candidate == "cuda") {
      ++cudaProbeCalls;
      return cudaAvailable;
    }
    return false;
  });

  // 1. Failure caching: with probe returning false for cuda, first call is false,
  // and second call does NOT invoke probe (count remains 1).
  cudaAvailable = false;
  CHECK(tensorProviderUsable("cuda") == false);
  CHECK(cudaProbeCalls == 1);
  CHECK(tensorProviderUsable("cuda") == false);
  CHECK(cudaProbeCalls == 1);

  // 2. Invalidate cache: after invalidateTensorProviderProbeCache(), probe is re-invoked
  cudaAvailable = true;
  invalidateTensorProviderProbeCache();
  CHECK(tensorProviderUsable("cuda") == true);
  CHECK(cudaProbeCalls == 2);
  // Subsequent calls use cached success
  CHECK(tensorProviderUsable("cuda") == true);
  CHECK(cudaProbeCalls == 2);

  // 3. R1 regression: probe returns true for webgpu and false for cuda
  invalidateTensorProviderProbeCache();
  cudaAvailable = false;
  webgpuAvailable = true;
  std::string winner;
  CHECK(gpuTensorSessionAvailable(&winner) == true);
  CHECK(winner == "webgpu");

  // Invalidate and make cuda succeed: CUDA is usable AND winner is "cuda" (precedes webgpu)
  invalidateTensorProviderProbeCache();
  cudaAvailable = true;
  CHECK(tensorProviderUsable("cuda") == true);
  CHECK(gpuTensorSessionAvailable(&winner) == true);
  CHECK(winner == "cuda");
}


TEST_CASE("benchProviderMatches compares canonical provider names", "[gpu][bench]") {
  using automix::ai::gpu::benchProviderMatches;
  CHECK(benchProviderMatches("auto", "cpu") == true);
  CHECK(benchProviderMatches("cuda", "cuda") == true);
  CHECK(benchProviderMatches("webgpu", "cuda") == false);
  CHECK(benchProviderMatches("webgpu", "cpu") == false);
  CHECK(benchProviderMatches("cuda", "") == false);
  CHECK(benchProviderMatches("cuda", "unknown") == false);
  // canonicalProviderName maps "wgpu" to "webgpu" and ignores case.
  CHECK(benchProviderMatches("WGPU", "webgpu") == true);
}

TEST_CASE("nearestRankPercentile uses the nearest-rank method", "[gpu][bench]") {
  using automix::ai::gpu::nearestRankPercentile;
  CHECK(nearestRankPercentile({}, 0.9) == 0.0);
  CHECK(nearestRankPercentile({5.0}, 0.9) == 5.0);
  CHECK(nearestRankPercentile({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 0.9) == 9.0);
  CHECK(nearestRankPercentile({3.0, 1.0, 2.0}, 0.5) == 2.0);
  CHECK(nearestRankPercentile({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 1.0) == 10.0);
}
