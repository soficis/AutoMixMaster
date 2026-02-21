#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "ai/HuggingFaceModelHub.h"

namespace automix::ai {

struct ModelCompatibilityResult {
  bool compatible = false;
  std::string reason;
  std::string taskScope;
  std::string packType;
  std::string engine;
  std::vector<std::string> expectedOutputKeys;
};

std::string inferTaskScope(const HubModelInfo& model);
ModelCompatibilityResult validateCatalogModel(const HubModelInfo& model);
std::string normalizeModelIdForPack(const std::string& modelId);
bool writeTurnkeyModelPackManifest(const std::filesystem::path& installPath,
                                   const HubModelInfo& model,
                                   const HubInstallResult& installResult,
                                   const ModelCompatibilityResult& compatibility,
                                   std::string* errorOut);

} // namespace automix::ai
