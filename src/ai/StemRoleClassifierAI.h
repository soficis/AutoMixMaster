#pragma once

#include <string>

#include "ai/IModelInference.h"
#include "analysis/AnalysisResult.h"
#include "domain/StemRole.h"

namespace automix::ai {

struct RolePrediction {
  domain::StemRole role = domain::StemRole::Unknown;
  double confidence = 0.0;
};

class StemRoleClassifierAI {
 public:
  explicit StemRoleClassifierAI(IModelInference* inference);

  RolePrediction predict(const std::string& stemName, const analysis::AnalysisResult& metrics) const;

 private:
  IModelInference* inference_ = nullptr;
};

} // namespace automix::ai
