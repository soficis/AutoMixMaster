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
};

struct ExternalRendererConfig {
  std::string id;
  std::string name;
  std::string version = "unknown";
  std::string licenseId = "unknown";
  std::filesystem::path binaryPath;
  bool bundledByDefault = false;
};

class RendererRegistry {
 public:
  std::vector<RendererInfo> list(const std::vector<ExternalRendererConfig>& externalConfigs = {}) const;
};

std::string toString(RendererLinkMode mode);

} // namespace automix::renderers
