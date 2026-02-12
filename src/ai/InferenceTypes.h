#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace automix::ai {

struct InferenceRequest {
  std::string task;
  std::vector<double> features;
  std::unordered_map<std::string, double> scalars;
};

struct InferenceResult {
  bool usedModel = false;
  std::unordered_map<std::string, double> outputs;
  std::string logMessage;
};

} // namespace automix::ai
