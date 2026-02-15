#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace automix::renderers {

enum class RendererLinkMode {
  InProcess,
  External,
};

struct RendererInfo {
  std::string id;
  std::string name;
  std::string version;
  std::string licenseId;
  RendererLinkMode linkMode = RendererLinkMode::InProcess;
  bool bundledByDefault = true;
  bool available = false;
  std::string discovery;
  std::filesystem::path binaryPath;
  std::string trustPolicyStatus;
  std::vector<std::string> pinnedProfileIds;
  std::filesystem::path capabilitySnapshotPath;
};

struct ExternalRendererConfig {
  std::string id;
  std::string name;
  std::string version = "unknown";
  std::string licenseId = "unknown";
  std::filesystem::path binaryPath;
  bool bundledByDefault = false;
  std::string signerId;
  std::string signatureAlgorithm;
  std::string signatureValue;
  bool signatureValid = false;
  std::string trustPolicyStatus = "unsigned";
  std::vector<std::string> pinnedProfileIds;
  std::string capabilitySnapshot;
};

class RendererRegistry {
 public:
  std::vector<RendererInfo> list(const std::vector<ExternalRendererConfig>& externalConfigs = {}) const;
};

std::string toString(RendererLinkMode mode);

} // namespace automix::renderers
