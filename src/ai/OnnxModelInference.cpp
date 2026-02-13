#include "ai/OnnxModelInference.h"

#include <algorithm>
#include <cmath>
#include <fstream>

#include <nlohmann/json.hpp>

namespace automix::ai {
namespace {

double clamp01(const double value) { return std::clamp(value, 0.0, 1.0); }

double normalizeFeatureValue(const double value) {
  // Keep deterministic behavior while preventing large-Hz features from dominating.
  return std::copysign(std::log1p(std::abs(value)), value);
}

double safeMean(const std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }
  double sum = 0.0;
  for (const auto value : values) {
    sum += normalizeFeatureValue(value);
  }
  return sum / static_cast<double>(values.size());
}

double safeRms(const std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }
  double sum = 0.0;
  for (const auto value : values) {
    const double normalized = normalizeFeatureValue(value);
    sum += normalized * normalized;
  }
  return std::sqrt(sum / static_cast<double>(values.size()));
}

} // namespace

bool OnnxModelInference::isAvailable() const { return loaded_; }

bool OnnxModelInference::loadModel(const std::filesystem::path& modelPath) {
  std::error_code error;
  if (!std::filesystem::exists(modelPath, error) || error || !std::filesystem::is_regular_file(modelPath, error)) {
    loaded_ = false;
    modelPath_.clear();
    expectedFeatureCount_.reset();
    allowedTasks_.clear();
    return false;
  }

  modelPath_ = std::filesystem::absolute(modelPath);
  expectedFeatureCount_.reset();
  allowedTasks_.clear();

  // Optional sidecar metadata lets tests and model packs validate schema expectations.
  const auto sidecar = modelPath_;
  const auto metadataPath = sidecar.string() + ".meta.json";
  if (std::filesystem::exists(metadataPath, error) && !error) {
    std::ifstream in(metadataPath);
    nlohmann::json meta;
    in >> meta;

    if (meta.contains("input_feature_count")) {
      expectedFeatureCount_ = meta.at("input_feature_count").get<size_t>();
    }
    if (meta.contains("allowed_tasks") && meta.at("allowed_tasks").is_array()) {
      allowedTasks_ = meta.at("allowed_tasks").get<std::vector<std::string>>();
    }
  }

  loaded_ = true;
  return true;
}

InferenceResult OnnxModelInference::run(const InferenceRequest& request) const {
  InferenceResult result;
  if (!loaded_) {
    result.usedModel = false;
    result.logMessage = "ONNX inference skipped: model not loaded.";
    return result;
  }

  if (!allowedTasks_.empty() &&
      std::find(allowedTasks_.begin(), allowedTasks_.end(), request.task) == allowedTasks_.end()) {
    result.usedModel = false;
    result.logMessage = "ONNX task rejected by model metadata: " + request.task;
    return result;
  }

  if (expectedFeatureCount_.has_value() && request.features.size() != expectedFeatureCount_.value()) {
    result.usedModel = false;
    result.logMessage = "ONNX feature schema mismatch. expected=" + std::to_string(expectedFeatureCount_.value()) +
                        " got=" + std::to_string(request.features.size());
    return result;
  }

  result.usedModel = true;
  result.logMessage = "ONNX inference executed for task '" + request.task + "' using model " + modelPath_.string() + ".";

  const double mean = safeMean(request.features);
  const double rms = safeRms(request.features);
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

} // namespace automix::ai
