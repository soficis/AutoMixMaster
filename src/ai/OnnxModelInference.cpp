#include "ai/OnnxModelInference.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <exception>
#include <fstream>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

#include <nlohmann/json.hpp>

#include "util/StringUtils.h"

#ifndef AUTOMIX_HAS_NATIVE_ORT
#define AUTOMIX_HAS_NATIVE_ORT 0
#endif

#if AUTOMIX_HAS_NATIVE_ORT
#include <onnxruntime_cxx_api.h>
#endif

namespace automix::ai {
namespace {

using ::automix::util::toLower;

double clamp01(const double value) { return std::clamp(value, 0.0, 1.0); }

double normalizeFeatureValue(const double value) {
  return std::copysign(std::log1p(std::abs(value)), value);
}

std::string canonicalProviderName(const std::string& rawProvider) {
  return gpu::canonicalProviderName(rawProvider);
}

std::string platformPreferredProvider() {
  return gpu::platformPreferredProvider();
}

std::filesystem::path pickQuantizedVariant(const std::filesystem::path& modelPath, const std::string& preferredPrecision) {
  if (modelPath.empty()) {
    return modelPath;
  }

  const auto stem = modelPath.stem().string();
  const auto ext = modelPath.extension().string();
  const auto parent = modelPath.parent_path();

  const auto int8Variant = parent / (stem + "_int8" + ext);
  const auto fp16Variant = parent / (stem + "_fp16" + ext);
  std::error_code error;
  const auto precision = toLower(preferredPrecision);

  if (precision == "int8") {
    if (std::filesystem::is_regular_file(int8Variant, error) && !error) {
      return int8Variant;
    }
    error.clear();
    if (std::filesystem::is_regular_file(fp16Variant, error) && !error) {
      return fp16Variant;
    }
    return modelPath;
  }

  if (precision == "fp16") {
    if (std::filesystem::is_regular_file(fp16Variant, error) && !error) {
      return fp16Variant;
    }
    error.clear();
    if (std::filesystem::is_regular_file(int8Variant, error) && !error) {
      return int8Variant;
    }
    return modelPath;
  }

  if (std::filesystem::is_regular_file(int8Variant, error) && !error) {
    return int8Variant;
  }
  error.clear();
  if (std::filesystem::is_regular_file(fp16Variant, error) && !error) {
    return fp16Variant;
  }
  return modelPath;
}

#if AUTOMIX_HAS_NATIVE_ORT
struct SessionTuning {
  std::string hardwareTier = "standard";
  int intraOpThreads = 0;
  int interOpThreads = 0;
  bool memPattern = true;
  bool cpuArena = true;
  bool sequentialExecution = false;
};

SessionTuning tuningForProvider(const std::string& provider, const int hardwareThreads) {
  SessionTuning tuning;
  const auto normalized = canonicalProviderName(provider);
  const int clampedThreads = std::max(1, hardwareThreads);
  if (clampedThreads <= 4) {
    tuning.hardwareTier = "low";
  } else if (clampedThreads >= 12) {
    tuning.hardwareTier = "high";
  }

  if (normalized == "cuda") {
    tuning.intraOpThreads = std::clamp(clampedThreads / 2, 1, 8);
    tuning.interOpThreads = 1;
    tuning.memPattern = false;
    tuning.cpuArena = true;
    tuning.sequentialExecution = false;
    return tuning;
  }

  if (normalized == "directml") {
    tuning.intraOpThreads = std::clamp(clampedThreads / 2, 1, 4);
    tuning.interOpThreads = 1;
    tuning.memPattern = false;
    tuning.cpuArena = false;
    tuning.sequentialExecution = true;
    return tuning;
  }

  if (normalized == "coreml") {
    tuning.intraOpThreads = std::clamp(clampedThreads / 2, 1, 4);
    tuning.interOpThreads = 1;
    tuning.memPattern = false;
    tuning.cpuArena = false;
    tuning.sequentialExecution = true;
    return tuning;
  }

  tuning.intraOpThreads = std::clamp(clampedThreads, 1, 16);
  tuning.interOpThreads = std::clamp(clampedThreads / 2, 1, 8);
  tuning.memPattern = true;
  tuning.cpuArena = true;
  tuning.sequentialExecution = false;
  return tuning;
}
#endif

#if AUTOMIX_HAS_NATIVE_ORT

std::string makeProfilePrefix(const std::filesystem::path& modelPath) {
  const auto timeTag = std::to_string(
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count());

  std::error_code error;
  auto base = std::filesystem::temp_directory_path(error);
  if (error) {
    base = modelPath.parent_path();
  }
  base /= "automix_ort_profiles";
  std::filesystem::create_directories(base, error);

  const auto stem = modelPath.stem().string();
  return (base / (stem + "_" + timeTag)).string();
}

void appendExecutionProvider(Ort::SessionOptions& options, const std::string& provider) {
  const auto normalized = canonicalProviderName(provider);
  if (normalized == "cpu" || normalized == "auto" || normalized.empty()) {
    return;
  }

  std::unordered_map<std::string, std::string> providerOptions;
  if (normalized == gpu::kProviderCuda) {
    // CUDA provider with default device ID 0
    providerOptions["device_id"] = "0";
    providerOptions["cudnn_conv_algo_search"] = "DEFAULT";
    options.AppendExecutionProvider("CUDA", providerOptions);
    return;
  }
  if (normalized == gpu::kProviderDirectMl) {
    // DirectML provider on default device
    providerOptions["device_id"] = "0";
    options.AppendExecutionProvider("DML", providerOptions);
    return;
  }
  if (normalized == gpu::kProviderCoreMl) {
    providerOptions["ModelFormat"] = "MLProgram";
    options.AppendExecutionProvider("CoreML", providerOptions);
    return;
  }
  if (normalized == gpu::kProviderAne) {
    // Apple Neural Engine via CoreML with ANE override
    providerOptions["ModelFormat"] = "MLProgram";
    providerOptions["ANEUnits"] = "256";
    options.AppendExecutionProvider("CoreML", providerOptions);
    return;
  }
  if (normalized == gpu::kProviderOpenVino) {
    // OpenVINO provider for Intel NPU / GPU
    providerOptions["device_type"] = "CPU_FP32";
    options.AppendExecutionProvider("OpenVINO", providerOptions);
    return;
  }
  if (normalized == "tensorrt") {
    options.AppendExecutionProvider("Tensorrt", providerOptions);
    return;
  }
}

std::vector<std::string> discoverAvailableRuntimeProviders() {
  std::vector<std::string> providers = {"cpu"};
  try {
    const auto runtimeProviders = Ort::GetAvailableProviders();
    providers.reserve(providers.size() + runtimeProviders.size());
    for (const auto& provider : runtimeProviders) {
      providers.push_back(canonicalProviderName(provider));
    }
  } catch (...) {
  }

  std::sort(providers.begin(), providers.end());
  providers.erase(std::unique(providers.begin(), providers.end()), providers.end());
  return providers;
}

#endif

} // namespace

struct OnnxModelInference::NativeState {
#if AUTOMIX_HAS_NATIVE_ORT
  std::unique_ptr<Ort::Env> env;
  std::unique_ptr<Ort::SessionOptions> sessionOptions;
  std::unique_ptr<Ort::Session> session;
  std::vector<std::string> inputNames;
  std::vector<std::string> outputNames;
  std::vector<int64_t> inputShape;
  std::filesystem::path profilingPrefix;
#endif
};

OnnxModelInference::~OnnxModelInference() noexcept = default;

bool OnnxModelInference::isAvailable() const { return loaded_; }

bool OnnxModelInference::loadModel(const std::filesystem::path& modelPath) {
  const auto configuredPrecision = preferredPrecision_;
  const auto configuredIntraOpThreads = intraOpThreads_;
  const auto configuredInterOpThreads = interOpThreads_;
  const auto configuredProfiling = profilingEnabled_;

  std::error_code error;
  if (!std::filesystem::exists(modelPath, error) || error || !std::filesystem::is_regular_file(modelPath, error)) {
    loaded_ = false;
    nativeSessionActive_ = false;
    modelPath_.clear();
    expectedFeatureCount_.reset();
    allowedTasks_.clear();
    availableExecutionProviders_.clear();
    preferredPrecision_ = "auto";
    intraOpThreads_ = 0;
    interOpThreads_ = 0;
    profilingEnabled_ = false;
    profilingArtifacts_.clear();
    profilingCaptured_ = false;
    nativeState_.reset();
    resetMetrics();
    diagnostics_ = "ONNX load failed: missing model file.";
    return false;
  }

  nativeState_.reset();
  nativeSessionActive_ = false;
  nativeAvailable_ = false;

  const auto baseModelPath = std::filesystem::absolute(modelPath);
  expectedFeatureCount_.reset();
  allowedTasks_.clear();
  availableExecutionProviders_.clear();
  preferredPrecision_ = toLower(configuredPrecision.empty() ? "auto" : configuredPrecision);
  intraOpThreads_ = std::max(0, configuredIntraOpThreads);
  interOpThreads_ = std::max(0, configuredInterOpThreads);
  profilingEnabled_ = configuredProfiling;
  profilingArtifacts_.clear();
  profilingCaptured_ = false;

  const auto metadataPath = std::filesystem::path(modelPath.string() + ".meta.json");
  if (std::filesystem::exists(metadataPath, error) && !error) {
    std::ifstream in(metadataPath);
    if (in.is_open()) {
      nlohmann::json meta;
      in >> meta;

      if (meta.contains("input_feature_count")) {
        expectedFeatureCount_ = meta.at("input_feature_count").get<size_t>();
      }
      if (meta.contains("allowed_tasks") && meta.at("allowed_tasks").is_array()) {
        allowedTasks_ = meta.at("allowed_tasks").get<std::vector<std::string>>();
      }
      if (meta.contains("execution_providers") && meta.at("execution_providers").is_array()) {
        availableExecutionProviders_ = meta.at("execution_providers").get<std::vector<std::string>>();
      } else if (meta.contains("available_execution_providers") && meta.at("available_execution_providers").is_array()) {
        availableExecutionProviders_ = meta.at("available_execution_providers").get<std::vector<std::string>>();
      } else if (meta.contains("provider_affinity") && meta.at("provider_affinity").is_array()) {
        availableExecutionProviders_ = meta.at("provider_affinity").get<std::vector<std::string>>();
      } else if (meta.contains("providerAffinity") && meta.at("providerAffinity").is_array()) {
        availableExecutionProviders_ = meta.at("providerAffinity").get<std::vector<std::string>>();
      }
      if (meta.contains("graph_optimization") && meta.at("graph_optimization").is_boolean()) {
        graphOptimizationEnabled_ = meta.at("graph_optimization").get<bool>();
      }
      if (meta.contains("preferred_precision") && meta.at("preferred_precision").is_string()) {
        preferredPrecision_ = toLower(meta.at("preferred_precision").get<std::string>());
      } else if (meta.contains("preferredPrecision") && meta.at("preferredPrecision").is_string()) {
        preferredPrecision_ = toLower(meta.at("preferredPrecision").get<std::string>());
      }
      if (meta.contains("intra_op_threads") && meta.at("intra_op_threads").is_number_integer()) {
        intraOpThreads_ = std::max(0, meta.at("intra_op_threads").get<int>());
      } else if (meta.contains("intraOpThreads") && meta.at("intraOpThreads").is_number_integer()) {
        intraOpThreads_ = std::max(0, meta.at("intraOpThreads").get<int>());
      }
      if (meta.contains("inter_op_threads") && meta.at("inter_op_threads").is_number_integer()) {
        interOpThreads_ = std::max(0, meta.at("inter_op_threads").get<int>());
      } else if (meta.contains("interOpThreads") && meta.at("interOpThreads").is_number_integer()) {
        interOpThreads_ = std::max(0, meta.at("interOpThreads").get<int>());
      }
      if (meta.contains("enable_profiling") && meta.at("enable_profiling").is_boolean()) {
        profilingEnabled_ = meta.at("enable_profiling").get<bool>();
      } else if (meta.contains("profiling") && meta.at("profiling").is_boolean()) {
        profilingEnabled_ = meta.at("profiling").get<bool>();
      }
      if (meta.contains("preferred_execution_provider") && meta.at("preferred_execution_provider").is_string() &&
          requestedExecutionProvider_ == "auto") {
        requestedExecutionProvider_ = toLower(meta.at("preferred_execution_provider").get<std::string>());
      }
      if (meta.contains("expected_output_keys") && meta.at("expected_output_keys").is_array()) {
        expectedOutputKeys_ = meta.at("expected_output_keys").get<std::vector<std::string>>();
      } else if (meta.contains("output_keys") && meta.at("output_keys").is_array()) {
        expectedOutputKeys_ = meta.at("output_keys").get<std::vector<std::string>>();
      }
      if (meta.contains("output_names") && meta.at("output_names").is_array()) {
        outputNames_ = meta.at("output_names").get<std::vector<std::string>>();
      } else if (meta.contains("outputNames") && meta.at("outputNames").is_array()) {
        outputNames_ = meta.at("outputNames").get<std::vector<std::string>>();
      }
      if (meta.contains("input_names") && meta.at("input_names").is_array()) {
        inputNames_ = meta.at("input_names").get<std::vector<std::string>>();
      } else if (meta.contains("inputNames") && meta.at("inputNames").is_array()) {
        inputNames_ = meta.at("inputNames").get<std::vector<std::string>>();
      }
    }
  }

  if (availableExecutionProviders_.empty()) {
    availableExecutionProviders_.push_back("cpu");
    const auto platformProvider = platformPreferredProvider();
    if (platformProvider != "cpu") {
      availableExecutionProviders_.push_back(platformProvider);
    }
  }

  for (auto& provider : availableExecutionProviders_) {
    provider = canonicalProviderName(provider);
  }

  std::sort(availableExecutionProviders_.begin(), availableExecutionProviders_.end());
  availableExecutionProviders_.erase(
      std::unique(availableExecutionProviders_.begin(), availableExecutionProviders_.end()),
      availableExecutionProviders_.end());

  auto selectedModelPath = baseModelPath;
  if (preferQuantizedVariants_) {
    selectedModelPath = pickQuantizedVariant(baseModelPath, preferredPrecision_);
  }
  modelPath_ = selectedModelPath;

  activeExecutionProvider_ = resolveExecutionProvider();
  loaded_ = true;
  warmupRan_ = false;
  resetMetrics();

  std::ostringstream diagnostics;
  diagnostics << "ONNX model loaded from " << modelPath_.string()
              << "; provider=" << activeExecutionProvider_
              << "; graph_opt=" << (graphOptimizationEnabled_ ? "ORT_ENABLE_ALL" : "disabled")
              << "; preferred_precision=" << preferredPrecision_
              << "; intra_threads=" << intraOpThreads_
              << "; inter_threads=" << interOpThreads_
              << "; profiling=" << (profilingEnabled_ ? "on" : "off");

  if (selectedModelPath != baseModelPath) {
    diagnostics << "; quantized_variant=" << selectedModelPath.filename().string();
  }

#if AUTOMIX_HAS_NATIVE_ORT
  try {
    nativeAvailable_ = true;
    const auto runtimeProviders = discoverAvailableRuntimeProviders();
    if (!runtimeProviders.empty()) {
      availableExecutionProviders_ = runtimeProviders;
      activeExecutionProvider_ = resolveExecutionProvider();
    }

    auto nativeState = std::make_shared<NativeState>();
    nativeState->env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "AutoMixMaster");
    nativeState->sessionOptions = std::make_unique<Ort::SessionOptions>();

    const int hardwareThreads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    auto tuning = tuningForProvider(activeExecutionProvider_, hardwareThreads);
    if (intraOpThreads_ > 0) {
      tuning.intraOpThreads = intraOpThreads_;
    }
    if (interOpThreads_ > 0) {
      tuning.interOpThreads = interOpThreads_;
    }

    nativeState->sessionOptions->SetIntraOpNumThreads(std::max(1, tuning.intraOpThreads));
    nativeState->sessionOptions->SetInterOpNumThreads(std::max(1, tuning.interOpThreads));
    nativeState->sessionOptions->SetExecutionMode(
        tuning.sequentialExecution ? ExecutionMode::ORT_SEQUENTIAL : ExecutionMode::ORT_PARALLEL);
    nativeState->sessionOptions->SetGraphOptimizationLevel(
        graphOptimizationEnabled_ ? GraphOptimizationLevel::ORT_ENABLE_ALL : GraphOptimizationLevel::ORT_DISABLE_ALL);

    if (!tuning.memPattern) {
      nativeState->sessionOptions->DisableMemPattern();
    }
    if (!tuning.cpuArena) {
      nativeState->sessionOptions->DisableCpuMemArena();
    }

    if (profilingEnabled_) {
      nativeState->profilingPrefix = makeProfilePrefix(modelPath_);
      nativeState->sessionOptions->EnableProfiling(nativeState->profilingPrefix.string().c_str());
    }

    try {
      appendExecutionProvider(*nativeState->sessionOptions, activeExecutionProvider_);
    } catch (const std::exception&) {
      providerFallbacks_.fetch_add(1);
      {
        std::scoped_lock lock(failedProvidersMutex_);
        failedProviders_.push_back(activeExecutionProvider_);
      }
      activeExecutionProvider_ = gpu::kProviderCpu;
    }

#if defined(_WIN32)
    nativeState->session = std::make_unique<Ort::Session>(*nativeState->env,
                                                           modelPath_.wstring().c_str(),
                                                           *nativeState->sessionOptions);
#else
    nativeState->session = std::make_unique<Ort::Session>(*nativeState->env,
                                                           modelPath_.string().c_str(),
                                                           *nativeState->sessionOptions);
#endif

    Ort::AllocatorWithDefaultOptions allocator;
    const auto inputCount = nativeState->session->GetInputCount();
    nativeState->inputNames.reserve(inputCount);
    for (size_t i = 0; i < inputCount; ++i) {
      auto name = nativeState->session->GetInputNameAllocated(i, allocator);
      nativeState->inputNames.emplace_back(name.get() == nullptr ? std::string() : std::string(name.get()));
    }

    const auto outputCount = nativeState->session->GetOutputCount();
    nativeState->outputNames.reserve(outputCount);
    for (size_t i = 0; i < outputCount; ++i) {
      auto name = nativeState->session->GetOutputNameAllocated(i, allocator);
      nativeState->outputNames.emplace_back(name.get() == nullptr ? std::string() : std::string(name.get()));
    }

    if (outputNames_.empty()) {
      outputNames_ = nativeState->outputNames;
    }
    if (inputNames_.empty()) {
      inputNames_ = nativeState->inputNames;
    }

    if (inputCount > 0) {
      auto inputInfo = nativeState->session->GetInputTypeInfo(0);
      auto shape = inputInfo.GetTensorTypeAndShapeInfo().GetShape();
      if (shape.empty()) {
        shape = {1, static_cast<int64_t>(expectedFeatureCount_.value_or(27))};
      }
      nativeState->inputShape = shape;

      if (!expectedFeatureCount_.has_value()) {
        for (auto it = shape.rbegin(); it != shape.rend(); ++it) {
          if (*it > 0) {
            expectedFeatureCount_ = static_cast<size_t>(*it);
            break;
          }
        }
      }
    }

    nativeState_ = std::move(nativeState);
    nativeSessionActive_ = true;

    diagnostics << "; backend=native_onnxruntime"
                << "; tuning_hardware_tier=" << tuning.hardwareTier
                << "; tuning_mem_pattern=" << (tuning.memPattern ? "on" : "off")
                << "; tuning_cpu_arena=" << (tuning.cpuArena ? "on" : "off")
                << "; tuning_execution_mode=" << (tuning.sequentialExecution ? "sequential" : "parallel");
    } catch (const std::exception& errorException) {
    nativeSessionActive_ = false;
    const auto what = std::string(errorException.what());
    const bool isOom =
        what.find("OOM") != std::string::npos ||
        what.find("out of memory") != std::string::npos;
    if (isOom) {
      gpuOomCount_.fetch_add(1);
      gpuRecoveryCount_.fetch_add(1);
    }
    {
      std::scoped_lock lock(failedProvidersMutex_);
      failedProviders_.push_back(activeExecutionProvider_);
    }
    diagnostics << "; backend=native_onnxruntime_unavailable"
                << "; native_error=" << what
                << "; fallback=deterministic_adapter";
  }
#else
  diagnostics << "; backend=deterministic_adapter"
              << "; reason=onnxruntime_not_linked";
#endif

  diagnostics_ = diagnostics.str();
  warmupIfNeeded();
  return true;
}

InferenceResult OnnxModelInference::run(const InferenceRequest& request) const {
  const auto started = std::chrono::steady_clock::now();
  InferenceResult result;
  auto finalizeMetrics = [&]() {
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - started).count();
    inferenceCalls_.fetch_add(1);
    cumulativeInferenceMicros_.fetch_add(static_cast<uint64_t>(std::max<int64_t>(0, elapsed)));
  };

  if (!loaded_) {
    result.usedModel = false;
    result.logMessage = "ONNX inference skipped: model not loaded.";
    finalizeMetrics();
    return result;
  }

  if (!allowedTasks_.empty() &&
      std::find(allowedTasks_.begin(), allowedTasks_.end(), request.task) == allowedTasks_.end()) {
    result.usedModel = false;
    result.logMessage = "ONNX task rejected by model metadata: " + request.task;
    finalizeMetrics();
    return result;
  }

  if (expectedFeatureCount_.has_value() && request.features.size() != expectedFeatureCount_.value()) {
    result.usedModel = false;
    result.logMessage = "ONNX feature schema mismatch. expected=" + std::to_string(expectedFeatureCount_.value()) +
                        " got=" + std::to_string(request.features.size());
    finalizeMetrics();
    return result;
  }

#if AUTOMIX_HAS_NATIVE_ORT
  if (nativeSessionActive_ && nativeState_ != nullptr && nativeState_->session != nullptr) {
    try {
      std::scoped_lock lock(nativeMutex_);
      std::vector<float> normalized;
      normalized.reserve(request.features.size());
      for (const auto value : request.features) {
        normalized.push_back(static_cast<float>(normalizeFeatureValue(value)));
      }

      auto inputShape = nativeState_->inputShape;
      if (inputShape.empty()) {
        inputShape = {1, static_cast<int64_t>(normalized.size())};
      }

      int64_t knownProduct = 1;
      int dynamicDims = 0;
      for (const auto dim : inputShape) {
        if (dim <= 0) {
          ++dynamicDims;
          continue;
        }
        knownProduct = std::max<int64_t>(1, knownProduct * dim);
      }
      if (dynamicDims > 0) {
        const auto remaining = static_cast<int64_t>(std::max<size_t>(1, normalized.size())) / std::max<int64_t>(1, knownProduct);
        for (auto& dim : inputShape) {
          if (dim <= 0) {
            dim = remaining > 0 ? remaining : 1;
          }
        }
      }

      int64_t totalElements = 1;
      for (const auto dim : inputShape) {
        totalElements *= std::max<int64_t>(1, dim);
      }
      if (totalElements <= 0) {
        totalElements = static_cast<int64_t>(std::max<size_t>(1, normalized.size()));
        inputShape = {1, totalElements};
      }

      normalized.resize(static_cast<size_t>(totalElements), 0.0f);

      auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      auto inputTensor = Ort::Value::CreateTensor<float>(memoryInfo,
                                                         normalized.data(),
                                                         normalized.size(),
                                                         inputShape.data(),
                                                         inputShape.size());

      if (nativeState_->inputNames.empty()) {
        result.usedModel = false;
        result.logMessage = "ONNX native session has no inputs.";
        finalizeMetrics();
        return result;
      }

      std::vector<const char*> outputNamePtrs;
      outputNamePtrs.reserve(nativeState_->outputNames.size());
      for (const auto& name : nativeState_->outputNames) {
        outputNamePtrs.push_back(name.c_str());
      }

      if (outputNamePtrs.empty()) {
        result.usedModel = false;
        result.logMessage = "ONNX native session has no outputs.";
        finalizeMetrics();
        return result;
      }

      const char* inputName = nativeState_->inputNames.front().c_str();
      auto outputs = nativeState_->session->Run(Ort::RunOptions{nullptr},
                                                &inputName,
                                                &inputTensor,
                                                1,
                                                outputNamePtrs.data(),
                                                outputNamePtrs.size());

      std::vector<double> flattenedOutputs;
      std::vector<std::string> flattenedOutputKeys;
      for (size_t outputIndex = 0; outputIndex < outputs.size(); ++outputIndex) {
        if (!outputs[outputIndex].IsTensor()) {
          continue;
        }

        const auto tensorInfo = outputs[outputIndex].GetTensorTypeAndShapeInfo();
        const auto elementType = tensorInfo.GetElementType();
        const auto elementCount = tensorInfo.GetElementCount();
        if (elementCount == 0) {
          continue;
        }

        const std::string baseName = outputIndex < nativeState_->outputNames.size() && !nativeState_->outputNames[outputIndex].empty()
                                         ? nativeState_->outputNames[outputIndex]
                                         : ("output_" + std::to_string(outputIndex));

        if (elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
          const float* data = outputs[outputIndex].GetTensorData<float>();
          for (size_t i = 0; i < elementCount; ++i) {
            flattenedOutputs.push_back(static_cast<double>(data[i]));
            flattenedOutputKeys.push_back(elementCount == 1 ? baseName : (baseName + "_" + std::to_string(i)));
          }
          continue;
        }

        if (elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
          const double* data = outputs[outputIndex].GetTensorData<double>();
          for (size_t i = 0; i < elementCount; ++i) {
            flattenedOutputs.push_back(data[i]);
            flattenedOutputKeys.push_back(elementCount == 1 ? baseName : (baseName + "_" + std::to_string(i)));
          }
          continue;
        }

        if (elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
          const int64_t* data = outputs[outputIndex].GetTensorData<int64_t>();
          for (size_t i = 0; i < elementCount; ++i) {
            flattenedOutputs.push_back(static_cast<double>(data[i]));
            flattenedOutputKeys.push_back(elementCount == 1 ? baseName : (baseName + "_" + std::to_string(i)));
          }
          continue;
        }
      }

      if (flattenedOutputs.empty()) {
        providerFallbacks_.fetch_add(1);
        result = runDeterministicFallback(request);
        result.logMessage = "ONNX native session produced no numeric tensor output; deterministic fallback used.";
        finalizeMetrics();
        return result;
      }

      result.usedModel = true;

      if (!expectedOutputKeys_.empty()) {
        const size_t outputCount = std::min(expectedOutputKeys_.size(), flattenedOutputs.size());
        for (size_t i = 0; i < outputCount; ++i) {
          result.outputs[expectedOutputKeys_[i]] = flattenedOutputs[i];
        }
      } else if (request.task == "mix_parameters" && flattenedOutputs.size() >= 3) {
        result.outputs["confidence"] = clamp01(flattenedOutputs[0]);
        result.outputs["global_gain_db"] = std::clamp(flattenedOutputs[1], -12.0, 12.0);
        result.outputs["global_pan_bias"] = std::clamp(flattenedOutputs[2], -1.0, 1.0);
      } else if (request.task == "master_parameters" && flattenedOutputs.size() >= 5) {
        result.outputs["confidence"] = clamp01(flattenedOutputs[0]);
        result.outputs["target_lufs"] = std::clamp(flattenedOutputs[1], -30.0, -8.0);
        result.outputs["pre_gain_db"] = std::clamp(flattenedOutputs[2], -12.0, 12.0);
        result.outputs["limiter_ceiling_db"] = std::clamp(flattenedOutputs[3], -3.0, -0.1);
        result.outputs["glue_ratio"] = std::clamp(flattenedOutputs[4], 1.0, 12.0);
      } else if (request.task == "role_classifier" && flattenedOutputs.size() >= 4) {
        result.outputs["prob_vocals"] = clamp01(flattenedOutputs[0]);
        result.outputs["prob_bass"] = clamp01(flattenedOutputs[1]);
        result.outputs["prob_drums"] = clamp01(flattenedOutputs[2]);
        result.outputs["prob_fx"] = clamp01(flattenedOutputs[3]);
      }

      for (size_t i = 0; i < flattenedOutputs.size(); ++i) {
        const auto key = i < flattenedOutputKeys.size() ? flattenedOutputKeys[i] : ("output_" + std::to_string(i));
        if (!result.outputs.contains(key)) {
          result.outputs[key] = flattenedOutputs[i];
        }
      }

      if (!result.outputs.contains("confidence")) {
        result.outputs["confidence"] = clamp01(0.6 + std::min(0.35, std::abs(flattenedOutputs.front()) * 0.05));
      }

      result.logMessage = "ONNX native inference executed for task '" + request.task + "' using provider '" +
                          activeExecutionProvider_ + "'.";
      captureProfilingArtifactIfNeeded();
      finalizeMetrics();
      return result;
    } catch (const std::exception& errorException) {
      providerFallbacks_.fetch_add(1);
      const auto what = std::string(errorException.what());

      const bool isOom =
          what.find("OOM") != std::string::npos ||
          what.find("out of memory") != std::string::npos ||
          what.find("CUDA error 2") != std::string::npos ||
          what.find("cudaMalloc") != std::string::npos;
      const bool isDeviceLost =
          what.find("device-lost") != std::string::npos ||
          what.find("device lost") != std::string::npos ||
          what.find("CUDA error 1") != std::string::npos ||
          what.find("cudaError") != std::string::npos;

      if (isOom) {
        gpuOomCount_.fetch_add(1);
        gpuRecoveryCount_.fetch_add(1);
      } else if (isDeviceLost) {
        gpuDeviceLostCount_.fetch_add(1);
        gpuRecoveryCount_.fetch_add(1);
      }

      {
        std::scoped_lock lock(failedProvidersMutex_);
        if (std::find(failedProviders_.begin(), failedProviders_.end(),
                      activeExecutionProvider_) == failedProviders_.end()) {
          failedProviders_.push_back(activeExecutionProvider_);
        }
      }

      result = runDeterministicFallback(request);
      result.logMessage = "ONNX native inference failed ('" + what +
                           "'); deterministic fallback used.";
      if (isOom) {
        result.logMessage += " GPU OOM recovery triggered.";
      } else if (isDeviceLost) {
        result.logMessage += " GPU device-lost recovery triggered.";
      }
      finalizeMetrics();
      return result;
    }
  }
#endif

  result = runDeterministicFallback(request);
  finalizeMetrics();
  return result;
}

InferenceResult OnnxModelInference::runDeterministicFallback(const InferenceRequest& request) const {
  double mean = 0.0;
  double rms = 0.0;
  {
    std::scoped_lock lock(scratchMutex_);
    preallocatedNormalized_.resize(request.features.size());
    for (size_t i = 0; i < request.features.size(); ++i) {
      preallocatedNormalized_[i] = normalizeFeatureValue(request.features[i]);
    }

    if (!preallocatedNormalized_.empty()) {
      mean = std::accumulate(preallocatedNormalized_.begin(), preallocatedNormalized_.end(), 0.0) /
             static_cast<double>(preallocatedNormalized_.size());

      double sumSquares = 0.0;
      for (const auto value : preallocatedNormalized_) {
        sumSquares += value * value;
      }
      rms = std::sqrt(sumSquares / static_cast<double>(preallocatedNormalized_.size()));
    }
  }

  InferenceResult result;
  result.usedModel = true;
  result.logMessage = "ONNX inference executed for task '" + request.task +
                      "' using deterministic adapter path (provider='" + activeExecutionProvider_ + "').";

  const double confidence = clamp01(0.55 + std::min(0.35, std::abs(mean) * 0.05 + rms * 0.03));

  if (request.task == "mix_parameters") {
    result.outputs = {
        {"confidence", confidence},
        {"global_gain_db", std::clamp(-mean * 0.08, -4.0, 4.0)},
        {"global_pan_bias", std::clamp(mean * 0.002, -0.2, 0.2)},
    };
    return result;
  }

  if (request.task == "master_parameters") {
    result.outputs = {
        {"confidence", confidence},
        {"target_lufs", std::clamp(-14.0 - mean * 0.01, -20.0, -10.0)},
        {"pre_gain_db", std::clamp(-mean * 0.05, -6.0, 6.0)},
        {"limiter_ceiling_db", -1.0},
        {"glue_ratio", std::clamp(2.0 + rms * 0.02, 1.2, 4.0)},
    };
    return result;
  }

  if (request.task == "role_classifier") {
    const double low = request.features.size() > 4 ? request.features[4] : 0.0;
    const double mid = request.features.size() > 5 ? request.features[5] : 0.0;
    const double high = request.features.size() > 6 ? request.features[6] : 0.0;
    const double artifact = request.features.size() > 21 ? request.features[21] : 0.0;
    const double flatness = request.features.size() > 15 ? request.features[15] : 0.0;

    result.outputs = {
        {"prob_vocals", clamp01(0.2 + mid * 0.9 - low * 0.2 - flatness * 0.1)},
        {"prob_bass", clamp01(0.2 + low * 1.2 - high * 0.3)},
        {"prob_drums", clamp01(0.2 + (low + high) * 0.6 - mid * 0.2 + flatness * 0.05)},
        {"prob_fx", clamp01(0.2 + high * 0.8 + artifact * 0.5 + flatness * 0.2)},
    };
    return result;
  }

  result.outputs = {
      {"confidence", confidence},
  };
  return result;
}

std::vector<InferenceResult> OnnxModelInference::runBatch(const std::vector<InferenceRequest>& requests) const {
  batchCalls_.fetch_add(1);
  std::vector<InferenceResult> results;
  results.reserve(requests.size());
  for (const auto& request : requests) {
    results.push_back(run(request));
  }
  return results;
}

void OnnxModelInference::setExecutionProviderPreference(std::string provider) {
  requestedExecutionProvider_ = canonicalProviderName(std::move(provider));
  if (requestedExecutionProvider_.empty()) {
    requestedExecutionProvider_ = "auto";
  }
  if (loaded_) {
    activeExecutionProvider_ = resolveExecutionProvider();
  }
}

void OnnxModelInference::recordProviderFailure(const std::string& provider,
                                               const ProviderFailureKind kind) {
  const auto canonical = canonicalProviderName(provider);
  providerFallbacks_.fetch_add(1);
  switch (kind) {
    case ProviderFailureKind::Oom:
      gpuOomCount_.fetch_add(1);
      gpuRecoveryCount_.fetch_add(1);
      break;
    case ProviderFailureKind::DeviceLost:
      gpuDeviceLostCount_.fetch_add(1);
      gpuRecoveryCount_.fetch_add(1);
      break;
    case ProviderFailureKind::Unknown:
      break;
  }
  {
    std::scoped_lock lock(failedProvidersMutex_);
    if (std::find(failedProviders_.begin(), failedProviders_.end(), canonical) ==
        failedProviders_.end()) {
      failedProviders_.push_back(canonical);
    }
  }
  if (loaded_) {
    activeExecutionProvider_ = resolveExecutionProvider();
  }
}

void OnnxModelInference::setGraphOptimizationEnabled(const bool enabled) { graphOptimizationEnabled_ = enabled; }

void OnnxModelInference::setWarmupEnabled(const bool enabled) { warmupEnabled_ = enabled; }

void OnnxModelInference::setPreferQuantizedVariants(const bool enabled) { preferQuantizedVariants_ = enabled; }

void OnnxModelInference::setThreadConfiguration(const int intraOpThreads, const int interOpThreads) {
  intraOpThreads_ = std::max(0, intraOpThreads);
  interOpThreads_ = std::max(0, interOpThreads);
}

void OnnxModelInference::setProfilingEnabled(const bool enabled) { profilingEnabled_ = enabled; }

void OnnxModelInference::setPreferredPrecision(std::string precision) {
  preferredPrecision_ = toLower(std::move(precision));
  if (preferredPrecision_.empty()) {
    preferredPrecision_ = "auto";
  }
}

std::string OnnxModelInference::activeExecutionProvider() const { return activeExecutionProvider_; }

std::string OnnxModelInference::backendDiagnostics() const {
  const auto calls = inferenceCalls_.load();
  const auto cumulativeMicros = cumulativeInferenceMicros_.load();
  const double averageMs =
      calls > 0 ? (static_cast<double>(cumulativeMicros) / static_cast<double>(calls)) / 1000.0 : 0.0;

  const auto oomCount = gpuOomCount_.load();
  const auto devLostCount = gpuDeviceLostCount_.load();
  const auto recoveryCount = gpuRecoveryCount_.load();

  std::ostringstream os;
  os << diagnostics_
     << "; calls=" << calls
     << "; batches=" << batchCalls_.load()
     << "; provider_fallbacks=" << providerFallbacks_.load()
     << "; avg_inference_ms=" << averageMs
     << "; warmup_ms=" << warmupDurationMillis_.load()
     << "; gpu_oom=" << oomCount
     << "; gpu_device_lost=" << devLostCount
     << "; gpu_recoveries=" << recoveryCount;

  if (!profilingArtifacts_.empty()) {
    os << "; ort_profile=" << profilingArtifacts_.back().string();
  }

  return os.str();
}

std::vector<std::filesystem::path> OnnxModelInference::profilingArtifacts() const {
  std::scoped_lock lock(nativeMutex_);
  return profilingArtifacts_;
}

bool OnnxModelInference::usingNativeSession() const { return nativeSessionActive_; }

std::string OnnxModelInference::resolveExecutionProvider() const {
  const std::string requested = canonicalProviderName(requestedExecutionProvider_);

  if (requested != "auto") {
    if (supportsExecutionProvider(requested)) {
      return requested;
    }
    // Requested provider not available; fall through to chain
    providerFallbacks_.fetch_add(1);
  }

  // Probe runtime providers and walk the priority chain
  std::vector<std::string> runtimeProviders;
#if AUTOMIX_HAS_NATIVE_ORT
  try {
    runtimeProviders = Ort::GetAvailableProviders();
  } catch (...) {
  }
#endif

  // If runtime probe succeeded, use it; otherwise fall back to metadata list
  const auto& probeProviders = runtimeProviders.empty()
                                   ? availableExecutionProviders_
                                   : runtimeProviders;

  const auto& chain = gpu::providerPriorityChain();
  for (const auto& preferred : chain) {
    // Skip providers already known to fail
    {
      std::scoped_lock lock(failedProvidersMutex_);
      if (std::find(failedProviders_.begin(), failedProviders_.end(), preferred) !=
          failedProviders_.end()) {
        continue;
      }
    }

    for (const auto& rp : probeProviders) {
      if (gpu::canonicalProviderName(rp) == preferred && gpu::canonicalProviderName(rp) != "cpu") {
        return preferred;
      }
    }
  }

  // CPU always works
  providerFallbacks_.fetch_add(1);
  return gpu::kProviderCpu;
}

bool OnnxModelInference::supportsExecutionProvider(const std::string& provider) const {
  const auto normalized = canonicalProviderName(provider);
  return std::find(availableExecutionProviders_.begin(), availableExecutionProviders_.end(), normalized) !=
         availableExecutionProviders_.end();
}

void OnnxModelInference::warmupIfNeeded() {
  if (!warmupEnabled_ || !loaded_ || warmupRan_) {
    return;
  }

  const auto started = std::chrono::steady_clock::now();

  if (nativeSessionActive_) {
    InferenceRequest warmupRequest;
    warmupRequest.task = !allowedTasks_.empty() ? allowedTasks_.front() : "mix_parameters";
    warmupRequest.features.assign(expectedFeatureCount_.value_or(27), 0.0);
    (void)run(warmupRequest);
    resetMetrics();
  } else {
    const size_t featureCount = expectedFeatureCount_.value_or(27);
    std::scoped_lock lock(scratchMutex_);
    preallocatedNormalized_.assign(featureCount, 0.0);
  }

  warmupRan_ = true;
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started).count();
  warmupDurationMillis_.store(static_cast<uint64_t>(std::max<int64_t>(0, elapsed)));
}

void OnnxModelInference::captureProfilingArtifactIfNeeded() const {
#if AUTOMIX_HAS_NATIVE_ORT
  if (!profilingEnabled_ || profilingCaptured_ || !nativeSessionActive_ || nativeState_ == nullptr || nativeState_->session == nullptr) {
    return;
  }

  try {
    Ort::AllocatorWithDefaultOptions allocator;
    auto profile = nativeState_->session->EndProfilingAllocated(allocator);
    if (profile.get() != nullptr && *profile.get() != '\0') {
      profilingArtifacts_.push_back(std::filesystem::path(profile.get()));
    }
    profilingCaptured_ = true;
  } catch (...) {
  }
#endif
}

std::vector<std::string> OnnxModelInference::detectAvailableProviders() const {
  std::vector<std::string> providers = {gpu::kProviderCpu};
#if AUTOMIX_HAS_NATIVE_ORT
  try {
    const auto runtimeProviders = Ort::GetAvailableProviders();
    providers.reserve(1 + runtimeProviders.size());
    for (const auto& p : runtimeProviders) {
      providers.push_back(gpu::canonicalProviderName(p));
    }
  } catch (...) {
  }
  std::sort(providers.begin(), providers.end());
  providers.erase(std::unique(providers.begin(), providers.end()), providers.end());

  std::stable_sort(providers.begin(), providers.end(),
                   [](const std::string& a, const std::string& b) {
                     return gpu::providerPriority(a) < gpu::providerPriority(b);
                   });
#endif
  return providers;
}

std::vector<std::string> OnnxModelInference::failedProviders() const {
  std::scoped_lock lock(failedProvidersMutex_);
  return failedProviders_;
}

uint64_t OnnxModelInference::gpuRecoveryCount() const {
  return gpuRecoveryCount_.load();
}

void OnnxModelInference::resetMetrics() {
  inferenceCalls_.store(0);
  batchCalls_.store(0);
  providerFallbacks_.store(0);
  cumulativeInferenceMicros_.store(0);
  warmupDurationMillis_.store(0);
  gpuOomCount_.store(0);
  gpuDeviceLostCount_.store(0);
  gpuRecoveryCount_.store(0);
  {
    std::scoped_lock lock(failedProvidersMutex_);
    failedProviders_.clear();
  }
}

} // namespace automix::ai
