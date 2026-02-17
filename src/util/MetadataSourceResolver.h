#pragma once

#include <filesystem>
#include <optional>
#include <system_error>

#include "domain/Session.h"

namespace automix::util {

inline std::optional<std::filesystem::path> metadataSourcePath(const domain::Session& session) {
  if (session.originalMixPath.has_value()) {
    const std::filesystem::path originalPath(session.originalMixPath.value());
    std::error_code error;
    if (std::filesystem::is_regular_file(originalPath, error) && !error) {
      return originalPath;
    }
  }

  for (const auto& stem : session.stems) {
    if (!stem.enabled || stem.filePath.empty()) {
      continue;
    }
    const std::filesystem::path stemPath(stem.filePath);
    std::error_code error;
    if (std::filesystem::is_regular_file(stemPath, error) && !error) {
      return stemPath;
    }
  }

  for (const auto& stem : session.stems) {
    if (stem.filePath.empty()) {
      continue;
    }
    const std::filesystem::path stemPath(stem.filePath);
    std::error_code error;
    if (std::filesystem::is_regular_file(stemPath, error) && !error) {
      return stemPath;
    }
  }

  return std::nullopt;
}

} // namespace automix::util
