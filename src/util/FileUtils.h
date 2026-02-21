#pragma once

#include <filesystem>
#include <string>
#include <system_error>

namespace automix::util {

inline bool isRegularFile(const std::filesystem::path& path) {
  std::error_code error;
  return std::filesystem::is_regular_file(path, error) && !error;
}

inline std::string pathToUtf8(const std::filesystem::path& path) {
#if defined(__cpp_lib_char8_t)
  const auto utf8 = path.u8string();
  return std::string(reinterpret_cast<const char*>(utf8.data()), utf8.size());
#else
  return path.u8string();
#endif
}

inline std::string pathToGenericUtf8(const std::filesystem::path& path) {
#if defined(__cpp_lib_char8_t)
  const auto utf8 = path.generic_u8string();
  return std::string(reinterpret_cast<const char*>(utf8.data()), utf8.size());
#else
  return path.generic_u8string();
#endif
}

inline std::filesystem::path pathFromUtf8(const std::string& utf8Path) {
#if defined(__cpp_lib_char8_t)
  return std::filesystem::path(
      std::u8string(reinterpret_cast<const char8_t*>(utf8Path.data()), utf8Path.size()));
#else
  return std::filesystem::u8path(utf8Path);
#endif
}

} // namespace automix::util
