#pragma once

#include <optional>
#include <string>
#include <vector>

#include "ai/HuggingFaceModelHub.h"

namespace automix::ai {

class GitHubReleaseModelHub {
 public:
  std::vector<HubModelInfo> discoverRecommended(const HubModelQueryOptions& options = {}) const;
  std::optional<HubModelInfo> modelInfo(const std::string& modelId) const;
  HubInstallResult installModel(const std::string& modelId, const HubInstallOptions& options = {}) const;

  std::string resolveToken() const;
};

} // namespace automix::ai
