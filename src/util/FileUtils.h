#pragma once

#include <filesystem>
#include <system_error>

namespace automix::util {

inline bool isRegularFile(const std::filesystem::path& path) {
  std::error_code error;
  return std::filesystem::is_regular_file(path, error) && !error;
}

} // namespace automix::util
