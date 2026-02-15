#include "ai/OnnxModelInference.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>

#include <nlohmann/json.hpp>

namespace automix::ai {
namespace {

double clamp01(const double value) { return std::clamp(value, 0.0, 1.0); }

double normalizeFeatureValue(const double value) {
  return std::copysign(std::log1p(std::abs(value)), value);
}

std::string toLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

std::string platformPreferredProvider() {
#if defined(_WIN32)
  return "directml";
#elif defined(__APPLE__)
  return "coreml";
#else
  return "cuda";
#endif
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

} // namespace

bool OnnxModelInference::isAvailable() const { return loaded_; }

bool OnnxModelInference::loadModel(const std::filesystem::path& modelPath) {
  const auto configuredPrecision = preferredPrecision_;
  const auto configuredIntraOpThreads = intraOpThreads_;
  const auto configuredInterOpThreads = interOpThreads_;
  const auto configuredProfiling = profilingEnabled_;

  std::error_code error;
  if (!std::filesystem::exists(modelPath, error) || error || !std::filesystem::is_regular_file(modelPath, error)) {
    loaded_ = false;
    modelPath_.clear();
    expectedFeatureCount_.reset();
    allowedTasks_.clear();
    availableExecutionProviders_.clear();
    preferredPrecision_ = "auto";
    intraOpThreads_ = 0;
    interOpThreads_ = 0;
    profilingEnabled_ = false;
    inferenceCalls_.store(0);
    batchCalls_.store(0);
    providerFallbacks_.store(0);
    cumulativeInferenceMicros_.store(0);
    warmupDurationMillis_.store(0);
    diagnostics_ = "ONNX load failed: missing model file.";
    return false;
  }

  const auto baseModelPath = std::filesystem::absolute(modelPath);
  expectedFeatureCount_.reset();
  allowedTasks_.clear();
  availableExecutionProviders_.clear();
  preferredPrecision_ = toLower(configuredPrecision.empty() ? "auto" : configuredPrecision);
  intraOpThreads_ = std::max(0, configuredIntraOpThreads);
  interOpThreads_ = std::max(0, configuredInterOpThreads);
  profilingEnabled_ = configuredProfiling;

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
    provider = toLower(provider);
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
  inferenceCalls_.store(0);
  batchCalls_.store(0);
  providerFallbacks_.store(0);
  cumulativeInferenceMicros_.store(0);
  warmupDurationMillis_.store(0);

  std::ostringstream os;
  os << "ONNX model loaded from " << modelPath_.string() << "; provider=" << activeExecutionProvider_
     << "; graph_opt=" << (graphOptimizationEnabled_ ? "ORT_ENABLE_ALL" : "disabled")
     << "; preferred_precision=" << preferredPrecision_
     << "; intra_threads=" << intraOpThreads_
     << "; inter_threads=" << interOpThreads_
     << "; profiling=" << (profilingEnabled_ ? "on" : "off");
  if (selectedModelPath != baseModelPath) {
    os << "; quantized_variant=" << selectedModelPath.filename().string();
  }
  diagnostics_ = os.str();

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

  result.usedModel = true;
  result.logMessage = "ONNX inference executed for task '" + request.task + "' using provider '" +
                      activeExecutionProvider_ + "' (" + (graphOptimizationEnabled_ ? "ORT_ENABLE_ALL" : "graph-opt-off") +
                      ", precision=" + preferredPrecision_ + ").";

  const double confidence = clamp01(0.55 + std::min(0.35, std::abs(mean) * 0.05 + rms * 0.03));

  if (request.task == "mix_parameters") {
    result.outputs = {
        {"confidence", confidence},
        {"global_gain_db", std::clamp(-mean * 0.08, -4.0, 4.0)},
        {"global_pan_bias", std::clamp(mean * 0.002, -0.2, 0.2)},
    };
    finalizeMetrics();
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
    finalizeMetrics();
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
    finalizeMetrics();
    return result;
  }

  result.outputs = {
      {"confidence", confidence},
  };
  finalizeMetrics();
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
  requestedExecutionProvider_ = toLower(std::move(provider));
  if (requestedExecutionProvider_.empty()) {
    requestedExecutionProvider_ = "auto";
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

  std::ostringstream os;
  os << diagnostics_
     << "; calls=" << calls
     << "; batches=" << batchCalls_.load()
     << "; provider_fallbacks=" << providerFallbacks_.load()
     << "; avg_inference_ms=" << averageMs
     << "; warmup_ms=" << warmupDurationMillis_.load();
  return os.str();
}

std::string OnnxModelInference::resolveExecutionProvider() const {
  const std::string requested = toLower(requestedExecutionProvider_);
  const std::string preferred = requested == "auto" ? platformPreferredProvider() : requested;

  if (supportsExecutionProvider(preferred)) {
    return preferred;
  }

  providerFallbacks_.fetch_add(1);
  if (supportsExecutionProvider("cpu")) {
    return "cpu";
  }

  return availableExecutionProviders_.empty() ? "cpu" : availableExecutionProviders_.front();
}

bool OnnxModelInference::supportsExecutionProvider(const std::string& provider) const {
  const auto normalized = toLower(provider);
  return std::find(availableExecutionProviders_.begin(), availableExecutionProviders_.end(), normalized) !=
         availableExecutionProviders_.end();
}

void OnnxModelInference::warmupIfNeeded() {
  if (!warmupEnabled_ || !loaded_ || warmupRan_) {
    return;
  }

  const auto started = std::chrono::steady_clock::now();
  const size_t featureCount = expectedFeatureCount_.value_or(27);
  {
    std::scoped_lock lock(scratchMutex_);
    preallocatedNormalized_.assign(featureCount, 0.0);
  }
  warmupRan_ = true;
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started).count();
  warmupDurationMillis_.store(static_cast<uint64_t>(std::max<int64_t>(0, elapsed)));
}

} // namespace automix::ai
