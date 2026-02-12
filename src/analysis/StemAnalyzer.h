#pragma once

#include <string>
#include <vector>

#include "analysis/AnalysisResult.h"
#include "domain/Session.h"
#include "engine/AudioBuffer.h"

namespace automix::analysis {

class StemAnalyzer {
 public:
  AnalysisResult analyzeBuffer(const engine::AudioBuffer& buffer) const;
  std::vector<StemAnalysisEntry> analyzeSession(const domain::Session& session) const;
  std::string toJsonReport(const std::vector<StemAnalysisEntry>& entries) const;
};

} // namespace automix::analysis
