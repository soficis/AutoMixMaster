#pragma once

#ifdef ENABLE_ONNX

#include "ai/IModelInference.h"

namespace automix::ai {

class OnnxModelInference final : public IModelInference {
 public:
  bool isAvailable() const override;
  bool loadModel(const std::filesystem::path& modelPath) override;
  InferenceResult run(const InferenceRequest& request) const override;

 private:
  bool loaded_ = false;
};

} // namespace automix::ai

#endif
