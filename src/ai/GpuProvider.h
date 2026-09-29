#pragma once

#include <algorithm>
#include <string>
#include <string_view>
#include <vector>

namespace automix::ai::gpu {

inline constexpr const char* kProviderCpu = "cpu";
inline constexpr const char* kProviderAne = "ane";
inline constexpr const char* kProviderCoreMl = "coreml";
inline constexpr const char* kProviderCuda = "cuda";
inline constexpr const char* kProviderOpenVino = "openvino";
inline constexpr const char* kProviderDirectMl = "directml";

inline const std::vector<std::string>& providerPriorityChain() {
  static const std::vector<std::string> chain = {
      kProviderAne,
      kProviderCoreMl,
      kProviderCuda,
      kProviderOpenVino,
      kProviderDirectMl,
      kProviderCpu,
  };
  return chain;
}

inline std::string canonicalProviderName(const std::string& raw) {
  auto lower = raw;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

  if (lower.find("cpu") != std::string::npos) return kProviderCpu;
  if (lower.find("ane") != std::string::npos || lower.find("neural") != std::string::npos)
    return kProviderAne;
  if (lower.find("coreml") != std::string::npos) return kProviderCoreMl;
  if (lower.find("cuda") != std::string::npos) return kProviderCuda;
  if (lower.find("openvino") != std::string::npos || lower.find("vino") != std::string::npos)
    return kProviderOpenVino;
  if (lower.find("dml") != std::string::npos || lower.find("directml") != std::string::npos)
    return kProviderDirectMl;
  if (lower.find("tensorrt") != std::string::npos) return "tensorrt";
  if (lower.find("rocm") != std::string::npos) return "rocm";

  return lower;
}

inline std::string platformPreferredProvider() {
#if defined(__APPLE__) && defined(__arm64__)
  return kProviderAne;
#elif defined(__APPLE__)
  return kProviderCoreMl;
#elif defined(_WIN32)
  return kProviderDirectMl;
#else
  return kProviderCuda;
#endif
}

inline bool isGpuProvider(const std::string& provider) {
  const auto canon = canonicalProviderName(provider);
  return canon != kProviderCpu;
}

inline int providerPriority(const std::string& provider) {
  const auto canon = canonicalProviderName(provider);
  const auto& chain = providerPriorityChain();
  const auto it = std::find(chain.begin(), chain.end(), canon);
  if (it != chain.end()) {
    return static_cast<int>(std::distance(chain.begin(), it));
  }
  return static_cast<int>(chain.size());
}

// --- Optional ONNX Runtime capabilities -------------------------------------
// The two optional runtime paths -- a CUDA provider supplied by a plugin
// library, and a per-GPU compiled-model cache -- are guarded at compile time by
// AUTOMIX_HAS_EP_PLUGIN and AUTOMIX_HAS_EP_CONTEXT, both of which are off on a
// machine with no ONNX Runtime SDK. Their deciding logic lives here, as pure
// functions, so it is covered by the test suite on every build. Keep the ORT
// calls themselves thin: ask these, then make the call.

struct OrtVersion {
  int major = 0;
  int minor = 0;
  int patch = 0;
  bool known = false;
};

inline OrtVersion parseOrtVersion(const std::string& text) {
  OrtVersion version;
  int* fields[3] = {&version.major, &version.minor, &version.patch};
  std::size_t cursor = 0;
  for (int field = 0; field < 3; ++field) {
    if (field > 0) {
      if (cursor >= text.size() || text[cursor] != '.') break;
      ++cursor;
    }
    if (cursor >= text.size() || text[cursor] < '0' || text[cursor] > '9') break;
    int value = 0;
    while (cursor < text.size() && text[cursor] >= '0' && text[cursor] <= '9') {
      value = value * 10 + (text[cursor] - '0');
      ++cursor;
    }
    *fields[field] = value;
  }
  version.known = version.major > 0;
  return version;
}

inline std::string toString(const OrtVersion& version) {
  if (!version.known) return "unknown";
  return std::to_string(version.major) + "." + std::to_string(version.minor) + "." +
         std::to_string(version.patch);
}

inline bool supportsEpPlugin(const OrtVersion& version) {
  return version.known && (version.major > 1 || (version.major == 1 && version.minor >= 23));
}

inline bool supportsEpContext(const OrtVersion& version) {
  return version.known && (version.major > 1 || (version.major == 1 && version.minor >= 22));
}

struct PluginEpDecision {
  bool attempt = false;
  // Populated whenever attempt is false, so the reason can be reported instead
  // of silently degrading to the next provider in the chain.
  std::string reason;
};

inline PluginEpDecision decidePluginEpAttempt(bool compiledIn,
                                              const OrtVersion& runtime,
                                              const std::string& requestedProvider,
                                              const std::string& pluginLibraryPath) {
  PluginEpDecision decision;
  if (!compiledIn) {
    decision.reason =
        "plugin execution providers were not compiled in (needs ONNX Runtime 1.23 or newer)";
    return decision;
  }
  if (!isGpuProvider(requestedProvider)) {
    decision.reason = "requested provider '" + requestedProvider + "' does not need a plugin";
    return decision;
  }
  if (!supportsEpPlugin(runtime)) {
    decision.reason = "ONNX Runtime " + toString(runtime) +
                      " predates the plugin provider API; built-in providers are unaffected";
    return decision;
  }
  if (pluginLibraryPath.empty()) {
    decision.reason = "no provider plugin library was found on disk";
    return decision;
  }
  decision.attempt = true;
  return decision;
}

inline bool isSha256Hex64(const std::string& text) {
  if (text.size() != 64) return false;
  for (const char c : text) {
    const bool hex = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
    if (!hex) return false;
  }
  return true;
}

// Cache keys become filenames, so anything that is not portable on disk is
// folded to '-' rather than passed through.
inline std::string cacheKeyToken(const std::string& raw) {
  std::string token;
  token.reserve(raw.size());
  for (const char c : raw) {
    const bool safe = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                      c == '.' || c == '_' || c == '-';
    token.push_back(safe ? static_cast<char>(std::tolower(static_cast<unsigned char>(c))) : '-');
  }
  return token;
}

// The model digest is what keeps two different models from sharing one cache
// entry, so an unusable digest returns no key at all rather than a key that
// would collide across models.
inline std::string compiledModelCacheKey(const std::string& modelSha256,
                                         const std::string& provider,
                                         const std::string& gpuArchitecture,
                                         const std::string& driverVersion,
                                         const std::string& ortVersion) {
  if (!isSha256Hex64(modelSha256)) return {};
  return cacheKeyToken(modelSha256) + "-" + cacheKeyToken(canonicalProviderName(provider)) + "-" +
         cacheKeyToken(gpuArchitecture) + "-" + cacheKeyToken(driverVersion) + "-" +
         cacheKeyToken(ortVersion);
}

} // namespace automix::ai::gpu
