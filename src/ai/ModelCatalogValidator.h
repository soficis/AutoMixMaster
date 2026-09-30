#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "ai/HuggingFaceModelHub.h"
#include "ai/ModelPackLoader.h"

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
// Installed form of a catalog tensor contract: loads the downloaded graph
// against the contract and writes the graph's real tensor names in. Without
// native ONNX Runtime nothing can be probed, so the names stay omitted and
// load-time checks match positionally. With it, a graph that does not satisfy
// the contract fails here (errorOut names the mismatch) rather than at import.
std::optional<TensorContract> resolveInstalledTensorContract(const TensorContract& catalogContract,
                                                             const std::filesystem::path& modelPath,
                                                             std::string& errorOut);
bool writeTurnkeyModelPackManifest(const std::filesystem::path& installPath,
                                   const HubModelInfo& model,
                                   const HubInstallResult& installResult,
                                   const ModelCompatibilityResult& compatibility,
                                   const TensorContract* tensorContract,
                                   std::string* errorOut);

} // namespace automix::ai
