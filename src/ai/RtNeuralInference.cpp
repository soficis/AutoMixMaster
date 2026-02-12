#include "ai/RtNeuralInference.h"

#include <cmath>

namespace automix::ai {
namespace {

double sigmoid(const double value) { return 1.0 / (1.0 + std::exp(-value)); }

} // namespace

bool RtNeuralInference::isAvailable() const {
#ifdef ENABLE_RTNEURAL
  return loaded_;
#else
  return false;
#endif
}

bool RtNeuralInference::loadModel(const std::filesystem::path& modelPath) {
#ifdef ENABLE_RTNEURAL
  loaded_ = true;
  loadedModelPath_ = modelPath;
  return true;
#else
  (void)modelPath;
  loaded_ = false;
  loadedModelPath_.clear();
  return false;
#endif
}

InferenceResult RtNeuralInference::run(const InferenceRequest& request) const {
  InferenceResult result;
#ifndef ENABLE_RTNEURAL
  (void)request;
  result.usedModel = false;
  result.logMessage = "RTNeural disabled at build time.";
  return result;
#else
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
#endif
}

} // namespace automix::ai
