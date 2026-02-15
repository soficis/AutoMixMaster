#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "domain/Stem.h"

namespace automix::ai {

class StemSeparator final {
 public:
  struct SeparationResult {
    bool success = false;
    bool usedModel = false;
    std::vector<domain::Stem> stems;
    std::vector<std::filesystem::path> generatedFiles;
    std::string logMessage;
  };

  explicit StemSeparator(std::filesystem::path modelRoot = "assets/models/stem-separator");

  [[nodiscard]] bool isModelAvailable() const;
  SeparationResult separate(const std::filesystem::path& mixPath, const std::filesystem::path& outputDir) const;

 private:
  [[nodiscard]] std::filesystem::path resolveModelPath() const;
  std::filesystem::path modelRoot_;
};

} // namespace automix::ai
