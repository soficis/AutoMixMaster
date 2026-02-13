#pragma once

#include <optional>
#include <vector>

#include "ai/IModelInference.h"

namespace automix::ai {

class OnnxModelInference final : public IModelInference {
 public:
  bool isAvailable() const override;
  bool loadModel(const std::filesystem::path& modelPath) override;
  InferenceResult run(const InferenceRequest& request) const override;

 private:
  bool loaded_ = false;
  std::filesystem::path modelPath_;
  std::optional<size_t> expectedFeatureCount_;
  std::vector<std::string> allowedTasks_;
};

} // namespace automix::ai
