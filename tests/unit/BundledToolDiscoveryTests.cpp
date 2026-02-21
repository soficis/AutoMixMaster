#include <cstdlib>
#include <filesystem>
#include <fstream>

#include <catch2/catch_test_macros.hpp>

#include "renderers/FfmpegDiscovery.h"
#include "renderers/RsgainDiscovery.h"
#include "renderers/SoxDiscovery.h"

namespace {

std::string executableName(const std::string& baseName) {
#if defined(_WIN32)
  return baseName + ".exe";
#else
  return baseName;
#endif
}

void setEnvValue(const char* key, const std::string& value) {
#if defined(_WIN32)
  _putenv_s(key, value.c_str());
#else
  if (value.empty()) {
    unsetenv(key);
  } else {
    setenv(key, value.c_str(), 1);
  }
#endif
}

std::string readEnvValue(const char* key) {
  if (const char* value = std::getenv(key); value != nullptr) {
    return value;
  }
  return "";
}

} // namespace

TEST_CASE("Bundled tool discovery resolves FFMPEG_BIN override", "[renderer][discovery]") {
  const auto root = std::filesystem::temp_directory_path() / "automix_ffmpeg_discovery_env";
  const auto binary = root / executableName("ffmpeg");

  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  std::ofstream(binary).put('\n');

  const auto previousValue = readEnvValue("FFMPEG_BIN");
  setEnvValue("FFMPEG_BIN", binary.string());

  const auto result = automix::renderers::FfmpegDiscovery().find();
  REQUIRE(result.has_value());
  REQUIRE(result->executablePath == std::filesystem::absolute(binary));

  setEnvValue("FFMPEG_BIN", previousValue);
  std::filesystem::remove_all(root);
}

TEST_CASE("Bundled tool discovery resolves SOX_BIN override", "[renderer][discovery]") {
  const auto root = std::filesystem::temp_directory_path() / "automix_sox_discovery_env";
  const auto binary = root / executableName("sox");

  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  std::ofstream(binary).put('\n');

  const auto previousValue = readEnvValue("SOX_BIN");
  setEnvValue("SOX_BIN", binary.string());

  const auto result = automix::renderers::SoxDiscovery().find();
  REQUIRE(result.has_value());
  REQUIRE(result->executablePath == std::filesystem::absolute(binary));

  setEnvValue("SOX_BIN", previousValue);
  std::filesystem::remove_all(root);
}

TEST_CASE("Bundled tool discovery resolves RSGAIN_BIN override", "[renderer][discovery]") {
  const auto root = std::filesystem::temp_directory_path() / "automix_rsgain_discovery_env";
  const auto binary = root / executableName("rsgain");

  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  std::ofstream(binary).put('\n');

  const auto previousValue = readEnvValue("RSGAIN_BIN");
  setEnvValue("RSGAIN_BIN", binary.string());

  const auto result = automix::renderers::RsgainDiscovery().find();
  REQUIRE(result.has_value());
  REQUIRE(result->executablePath == std::filesystem::absolute(binary));

  setEnvValue("RSGAIN_BIN", previousValue);
  std::filesystem::remove_all(root);
}
