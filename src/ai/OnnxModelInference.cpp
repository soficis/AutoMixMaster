#include "ai/OnnxModelInference.h"

#ifdef ENABLE_ONNX

namespace automix::ai {

bool OnnxModelInference::isAvailable() const { return loaded_; }

bool OnnxModelInference::loadModel(const std::filesystem::path&) {
  // ONNX runtime integration point. Stub returns true to validate plumbing.
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

  result.usedModel = true;
  result.logMessage = "ONNX inference stub executed for task '" + request.task + "'.";

  result.outputs = {
      {"dryWet", 0.92},
      {"targetLufs", -14.0},
      {"confidence", 0.75},
  };
  return result;
}

} // namespace automix::ai

#endif
