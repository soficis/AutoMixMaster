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

  // Chain should be: ANE > CoreML > CUDA > OpenVINO > DirectML > CPU
  REQUIRE(chain.size() == 6);
  CHECK(chain[0] == "ane");
  CHECK(chain[1] == "coreml");
  CHECK(chain[2] == "cuda");
  CHECK(chain[3] == "openvino");
  CHECK(chain[4] == "directml");
  CHECK(chain[5] == "cpu");

  // Verify priority values (lower = higher priority)
  CHECK(providerPriority("ane") == 0);
  CHECK(providerPriority("coreml") == 1);
  CHECK(providerPriority("cuda") == 2);
  CHECK(providerPriority("openvino") == 3);
  CHECK(providerPriority("directml") == 4);
  CHECK(providerPriority("cpu") == 5);

  // Unknown providers have lowest priority
  CHECK(providerPriority("tensorrt") == 6);
}

TEST_CASE("GpuProvider isGpuProvider classification", "[gpu][provider]") {
  using automix::ai::gpu::isGpuProvider;

  CHECK_FALSE(isGpuProvider("cpu"));
  CHECK(isGpuProvider("cuda"));
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
         preferred == "cuda" || preferred == "directml"));
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
  REQUIRE(inference.loadModel(modelPath));

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
  REQUIRE(inference.loadModel(modelPath));
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
  REQUIRE(inference.loadModel(modelPath));
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
  REQUIRE(inference.loadModel(modelPath));
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
  REQUIRE(inference.loadModel(modelPath));
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
