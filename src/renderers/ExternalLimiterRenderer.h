#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "renderers/IRenderer.h"

namespace automix::renderers {

class ExternalLimiterRenderer final : public IRenderer {
 public:
  struct ValidationResult {
    bool valid = false;
    std::string version;
    std::vector<std::string> supportedFeatures;
    std::string errorCode;
    std::string diagnostics;
  };

  static ValidationResult validateBinary(const std::filesystem::path& binaryPath, int timeoutMs = 5000);

  bool isAvailable() const override;

  RenderResult render(const domain::Session& session,
                      const domain::RenderSettings& settings,
                      const ProgressCallback& onProgress,
                      std::atomic_bool* cancelFlag) const override;
};

} // namespace automix::renderers
