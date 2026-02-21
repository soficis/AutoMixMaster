#include <cstdlib>
#include <cctype>
#include <filesystem>
#include <fstream>

#include <catch2/catch_test_macros.hpp>

#include "renderers/PhaseLimiterDiscovery.h"

namespace {

std::string binaryNameForPlatform() {
#if defined(_WIN32)
  return "phase_limiter.exe";
#else
  return "phase_limiter";
#endif
}

void setPhaseLimiterEnv(const std::string& value) {
#if defined(_WIN32)
  _putenv_s("PHASELIMITER_BIN", value.c_str());
#else
  if (value.empty()) {
    unsetenv("PHASELIMITER_BIN");
  } else {
    setenv("PHASELIMITER_BIN", value.c_str(), 1);
  }
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

std::string lowerPath(std::filesystem::path path) {
  std::string text = std::filesystem::weakly_canonical(path).lexically_normal().string();
  for (char& c : text) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return text;
}

} // namespace

TEST_CASE("PhaseLimiter discovery finds binary inside assets folder", "[phaselimiter][discovery]") {
  const std::filesystem::path root = std::filesystem::temp_directory_path() / "automix_phaselimiter_discovery_assets";
  const std::filesystem::path binDir = root / "assets" / "PhaseLimiter" / "bin";
  const std::filesystem::path binary = binDir / binaryNameForPlatform();

  std::filesystem::remove_all(root);
  std::filesystem::create_directories(binDir);
  std::ofstream(binary).put('\n');

  automix::renderers::PhaseLimiterDiscovery discovery;
  const auto result = discovery.findInRoots({root});
  REQUIRE(result.has_value());
  REQUIRE(lowerPath(result->executablePath) == lowerPath(binary));
  REQUIRE(lowerPath(result->installRoot) == lowerPath(root / "assets" / "PhaseLimiter"));

  std::filesystem::remove_all(root);
}

TEST_CASE("PhaseLimiter discovery supports PHASELIMITER_BIN override", "[phaselimiter][discovery]") {
  const std::filesystem::path root = std::filesystem::temp_directory_path() / "automix_phaselimiter_discovery_env";
  const std::filesystem::path binDir = root / "custom_bin";
  const std::filesystem::path binary = binDir / binaryNameForPlatform();

  std::filesystem::remove_all(root);
  std::filesystem::create_directories(binDir);
  std::ofstream(binary).put('\n');

  std::string previousEnv;
#if defined(_WIN32)
  char* oldValue = nullptr;
  size_t oldLength = 0;
  if (_dupenv_s(&oldValue, &oldLength, "PHASELIMITER_BIN") == 0 && oldValue != nullptr) {
    previousEnv.assign(oldValue, oldLength > 0 ? oldLength - 1 : 0);
    free(oldValue);
  }
#else
  if (const char* existing = std::getenv("PHASELIMITER_BIN"); existing != nullptr) {
    previousEnv = existing;
  }
#endif

  setPhaseLimiterEnv(binary.string());

  automix::renderers::PhaseLimiterDiscovery discovery;
  const auto result = discovery.find();
  REQUIRE(result.has_value());
  REQUIRE(result->executablePath == std::filesystem::absolute(binary));

  setPhaseLimiterEnv(previousEnv);
  std::filesystem::remove_all(root);
}

TEST_CASE("PhaseLimiter discovery supports AUTOMIX_ASSET_ROOT override", "[phaselimiter][discovery]") {
  const std::filesystem::path root = std::filesystem::temp_directory_path() / "automix_phaselimiter_discovery_asset_root";
  const std::filesystem::path binDir = root / "assets" / "phaselimiter" / "bin";
  const std::filesystem::path binary = binDir / binaryNameForPlatform();

  std::filesystem::remove_all(root);
  std::filesystem::create_directories(binDir);
  std::ofstream(binary).put('\n');

  std::string previousEnv;
#if defined(_WIN32)
  char* oldValue = nullptr;
  size_t oldLength = 0;
  if (_dupenv_s(&oldValue, &oldLength, "AUTOMIX_ASSET_ROOT") == 0 && oldValue != nullptr) {
    previousEnv.assign(oldValue, oldLength > 0 ? oldLength - 1 : 0);
    free(oldValue);
  }
#else
  if (const char* existing = std::getenv("AUTOMIX_ASSET_ROOT"); existing != nullptr) {
    previousEnv = existing;
  }
#endif

  setEnvValue("AUTOMIX_ASSET_ROOT", root.string());

  automix::renderers::PhaseLimiterDiscovery discovery;
  const auto result = discovery.find();
  REQUIRE(result.has_value());
  REQUIRE(lowerPath(result->executablePath) == lowerPath(binary));

  setEnvValue("AUTOMIX_ASSET_ROOT", previousEnv);
  std::filesystem::remove_all(root);
}
