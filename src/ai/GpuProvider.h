#pragma once

#include <algorithm>
#include <string>
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

} // namespace automix::ai::gpu
