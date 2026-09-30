#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "ai/TensorTypes.h"

namespace automix::ai {

// Sibling of IModelInference for graphs whose inputs and outputs are shaped
// float32 tensors rather than a feature vector and named scalars. The spec
// probes (inputSpecs/outputSpecs) are the load-bearing addition: tensor names
// are frequently unpublished, so binding by name requires reading them back
// from the loaded graph.
class ITensorInference {
 public:
  virtual ~ITensorInference() = default;

  virtual bool isAvailable() const = 0;
  virtual bool loadModel(const std::filesystem::path& modelPath) = 0;
  virtual std::vector<TensorSpec> inputSpecs() const = 0;
  virtual std::vector<TensorSpec> outputSpecs() const = 0;
  virtual TensorInferenceResult run(const std::vector<TensorBinding>& inputs) const = 0;
};

class NullTensorInference final : public ITensorInference {
 public:
  bool isAvailable() const override { return false; }
  bool loadModel(const std::filesystem::path&) override {
    lastLog_ = "NullTensorInference: no tensor backend is configured.";
    return false;
  }
  std::vector<TensorSpec> inputSpecs() const override { return {}; }
  std::vector<TensorSpec> outputSpecs() const override { return {}; }
  TensorInferenceResult run(const std::vector<TensorBinding>& inputs) const override {
    TensorInferenceResult result;
    result.usedModel = false;
    result.logMessage = "NullTensorInference: skipped run with " + std::to_string(inputs.size()) +
                        " input binding(s) (no model loaded).";
    return result;
  }

  const std::string& lastLog() const { return lastLog_; }

 private:
  mutable std::string lastLog_;
};

} // namespace automix::ai
