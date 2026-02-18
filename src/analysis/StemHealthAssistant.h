#pragma once

#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "analysis/StemAnalyzer.h"
#include "domain/Session.h"

namespace automix::analysis {

enum class StemHealthSeverity {
  Info,
  Warning,
  Critical,
};

struct StemHealthIssue {
  std::string stemId;
  std::string stemName;
  std::string code;
  std::string message;
  StemHealthSeverity severity = StemHealthSeverity::Info;
  double score = 0.0;
};

struct StemHealthReport {
  std::vector<StemHealthIssue> issues;
  double overallRisk = 0.0;
  bool hasCriticalIssues = false;
};

class StemHealthAssistant {
 public:
  StemHealthReport analyze(const domain::Session& session,
                           const std::vector<StemAnalysisEntry>& analysisEntries) const;

  nlohmann::json toJson(const StemHealthReport& report) const;
  std::string toText(const StemHealthReport& report) const;
};

std::string toString(StemHealthSeverity severity);

} // namespace automix::analysis
