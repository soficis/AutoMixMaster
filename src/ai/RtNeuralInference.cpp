#include "ai/RtNeuralInference.h"

#include <algorithm>
#include <cmath>
#include <filesystem>

namespace automix::ai {
namespace {

double sigmoid(const double value) { return 1.0 / (1.0 + std::exp(-value)); }

double safeMean(const std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }
  double sum = 0.0;
  for (const auto value : values) {
    sum += value;
  }
  return sum / static_cast<double>(values.size());
}

} // namespace

bool RtNeuralInference::isAvailable() const { return loaded_; }

bool RtNeuralInference::loadModel(const std::filesystem::path& modelPath) {
  std::error_code error;
  if (!std::filesystem::exists(modelPath, error) || error || !std::filesystem::is_regular_file(modelPath, error)) {
    loaded_ = false;
    loadedModelPath_.clear();
    return false;
  }

  loaded_ = true;
  loadedModelPath_ = std::filesystem::absolute(modelPath);
  return true;
}

InferenceResult RtNeuralInference::run(const InferenceRequest& request) const {
  InferenceResult result;
  if (!loaded_) {
    result.usedModel = false;
    result.logMessage = "RTNeural model not loaded.";
    return result;
  }

  result.usedModel = true;
  result.logMessage = "RTNeural inference stub executed for task '" + request.task + "'.";
  const double low = request.features.size() > 1 ? request.features[1] : 0.0;
  const double mid = request.features.size() > 2 ? request.features[2] : 0.0;
  const double high = request.features.size() > 3 ? request.features[3] : 0.0;
  const double artifact = request.features.size() > 4 ? request.features[4] : 0.0;
  const double mean = safeMean(request.features);
  const double confidence = std::clamp(0.5 + std::abs(mean) * 0.05, 0.0, 1.0);

  if (request.task == "mix_parameters") {
    result.outputs = {
        {"confidence", confidence},
        {"global_gain_db", std::clamp(-mean * 0.08, -6.0, 6.0)},
        {"global_pan_bias", std::clamp(mean * 0.003, -0.3, 0.3)},
    };
    return result;
  }

  if (request.task == "master_parameters") {
    result.outputs = {
        {"confidence", confidence},
        {"target_lufs", std::clamp(-14.0 - mean * 0.02, -20.0, -10.0)},
        {"pre_gain_db", std::clamp(-mean * 0.05, -6.0, 6.0)},
        {"limiter_ceiling_db", -1.0},
        {"glue_ratio", std::clamp(2.0 + std::abs(mean) * 0.05, 1.2, 4.0)},
    };
    return result;
  }

  const double vocals = sigmoid(mid * 3.2 - high * 1.5 + 0.2);
  const double bass = sigmoid(low * 3.4 - high * 1.0);
  const double drums = sigmoid((low + high) * 1.4 - mid * 0.6);
  const double fx = sigmoid(high * 3.6 + artifact * 1.5 - low * 1.4);

  result.outputs = {
      {"prob_vocals", vocals},
      {"prob_bass", bass},
      {"prob_drums", drums},
      {"prob_fx", fx},
  };
  return result;
}

} // namespace automix::ai
