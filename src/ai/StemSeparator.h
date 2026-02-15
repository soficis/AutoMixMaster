#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "domain/Stem.h"

namespace automix::ai {

class StemSeparator final {
 public:
  struct SeparationOptions {
    std::optional<int> targetStemCount;
    std::optional<size_t> gpuMemoryBudgetMb;
    std::optional<int> maxStreams;
  };

  struct SeparationQaMetrics {
    double energyLeakage = 0.0;
    double residualDistortion = 0.0;
    double transientRetention = 0.0;
  };

  struct SeparationResult {
    bool success = false;
    bool usedModel = false;
    int stemVariantCount = 0;
    std::vector<domain::Stem> stems;
    std::vector<std::filesystem::path> generatedFiles;
    std::filesystem::path qaReportPath;
    SeparationQaMetrics qaMetrics;
    std::string logMessage;
  };

  explicit StemSeparator(std::filesystem::path modelRoot = "assets/models/stem-separator");

  [[nodiscard]] bool isModelAvailable() const;
  SeparationResult separate(const std::filesystem::path& mixPath,
                            const std::filesystem::path& outputDir,
                            const SeparationOptions& options = {}) const;

 private:
  [[nodiscard]] std::filesystem::path resolveModelPath() const;
  std::filesystem::path modelRoot_;
};

} // namespace automix::ai
