#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace automix::analysis {
struct AnalysisResult;
}

namespace automix::ai {

class FeatureSchemaV1 {
 public:
  static constexpr const char* kVersion = "1.0.0";

  static const std::vector<std::string>& names();
  static bool isCompatible(const std::string& version);
  static size_t featureCount();
  static std::vector<double> extract(const analysis::AnalysisResult& metrics);
};

} // namespace automix::ai
