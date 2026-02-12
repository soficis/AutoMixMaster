#pragma once

#include "ai/IModelInference.h"

namespace automix::ai {

class RtNeuralInference final : public IModelInference {
 public:
  bool isAvailable() const override;
  bool loadModel(const std::filesystem::path& modelPath) override;
  InferenceResult run(const InferenceRequest& request) const override;

 private:
  bool loaded_ = false;
  std::filesystem::path loadedModelPath_;
};

} // namespace automix::ai
