#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "ai/InferenceTypes.h"

namespace automix::ai {

class IModelInference {
 public:
  virtual ~IModelInference() = default;

  virtual bool isAvailable() const = 0;
  virtual bool loadModel(const std::filesystem::path& modelPath) = 0;
  virtual InferenceResult run(const InferenceRequest& request) const = 0;
};

class NullModelInference final : public IModelInference {
 public:
  bool isAvailable() const override { return false; }
  bool loadModel(const std::filesystem::path&) override {
    lastLog_ = "NullModelInference: no model backend is configured.";
    return false;
  }
  InferenceResult run(const InferenceRequest& request) const override {
    InferenceResult result;
    result.usedModel = false;
    result.logMessage = "NullModelInference: skipped task '" + request.task + "' (no model loaded).";
    return result;
  }

  const std::string& lastLog() const { return lastLog_; }

 private:
  mutable std::string lastLog_;
};

} // namespace automix::ai
