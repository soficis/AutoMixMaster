#include <cstdlib>
#include <cctype>
#include <filesystem>
#include <fstream>

#include <catch2/catch_test_macros.hpp>

#include "renderers/PhaseLimiterDiscovery.h"
#include "renderers/PhaseLimiterPins.h"
#include <regex>
#include <set>

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

TEST_CASE("PhaseLimiter discovery finds binary inside macOS bundle layout", "[phaselimiter][discovery]") {
  const std::filesystem::path appBundle = std::filesystem::temp_directory_path() / "AutoMixMaster.app";
  const std::filesystem::path binDir = appBundle / "Contents" / "MacOS" / "assets" / "phaselimiter" / "bin";
  const std::filesystem::path binary = binDir / binaryNameForPlatform();

  std::filesystem::remove_all(appBundle);
  std::filesystem::create_directories(binDir);
  std::ofstream(binary).put('\n');

  automix::renderers::PhaseLimiterDiscovery discovery;
  const auto result = discovery.findInRoots({appBundle});
  REQUIRE(result.has_value());
  REQUIRE(lowerPath(result->executablePath) == lowerPath(binary));
  REQUIRE(lowerPath(result->installRoot) == lowerPath(appBundle / "Contents" / "MacOS" / "assets" / "phaselimiter"));

  std::filesystem::remove_all(appBundle);
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

TEST_CASE("PhaseLimiter download pin table contains target platforms and valid sha256 digests", "[phaselimiter][discovery]") {
  const auto table = automix::renderers::phaseLimiterDownloadPinTable();
  REQUIRE_FALSE(table.empty());

  const std::vector<std::string> requiredPlatforms = {
      "windows-x64",
      "linux-x64",
      "macos-arm64",
  };

  for (const auto& required : requiredPlatforms) {
    const auto it = std::find_if(table.begin(), table.end(),
                                 [&](const automix::renderers::PhaseLimiterDownloadPin& pin) {
                                   return pin.platformKey == required;
                                 });
    INFO("Checking presence of required platform: " << required);
    REQUIRE(it != table.end());
    REQUIRE(!it->url.empty());
    REQUIRE(it->sha256.size() == 64);
    for (char c : it->sha256) {
      REQUIRE(((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')));
    }
    if (required == "linux-x64") {
      REQUIRE(it->sha256 == "0b382ba78b030926f706345d1b00d5f45890ba384c8a4e960320e3ee992bd163");
    }
  }

  for (const auto& pin : table) {
    REQUIRE(!pin.url.empty());
    REQUIRE(pin.sha256.size() == 64);
  }

  const auto currentKey = automix::renderers::currentPhaseLimiterPlatformKey();
  REQUIRE_FALSE(currentKey.empty());
  REQUIRE(currentKey != "unknown");
}

TEST_CASE("PhaseLimiter auto-download rejects unverified table pin with empty hash", "[phaselimiter][discovery]") {
  automix::renderers::PhaseLimiterDiscovery::resetAttemptedDownloadForTesting();

  bool downloadAttempted = false;
  automix::renderers::PhaseLimiterDiscovery::setDownloadFetcherForTesting(
      [&](const std::string& /*url*/, const std::filesystem::path& destination) {
        downloadAttempted = true;
        std::ofstream out(destination, std::ios::binary);
        out << "test-data";
        return true;
      });

  automix::renderers::PhaseLimiterDownloadPin emptyHashPin{
      "test-empty-hash",
      "https://example.com/test_empty.zip",
      ""
  };

  const auto result = automix::renderers::PhaseLimiterDiscovery::downloadAndInstall(emptyHashPin);
  REQUIRE(!result.has_value());
  REQUIRE(!downloadAttempted);

  automix::renderers::PhaseLimiterDiscovery::resetDownloadFetcherForTesting();
  automix::renderers::PhaseLimiterDiscovery::resetAttemptedDownloadForTesting();
}

TEST_CASE("PhaseLimiter auto-download rejects corrupted archive before extraction and discards file", "[phaselimiter][discovery]") {
  automix::renderers::PhaseLimiterDiscovery::resetAttemptedDownloadForTesting();

  std::filesystem::path interceptedDestination;
  automix::renderers::PhaseLimiterDiscovery::setDownloadFetcherForTesting(
      [&](const std::string& /*url*/, const std::filesystem::path& destination) {
        interceptedDestination = destination;
        std::ofstream out(destination, std::ios::binary);
        out << "corrupted-file-data-that-will-not-match-expected-hash";
        return true;
      });

  automix::renderers::PhaseLimiterDownloadPin testPin{
      "test-platform",
      "https://example.com/test_corrupted.zip",
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  };

  const auto result = automix::renderers::PhaseLimiterDiscovery::downloadAndInstall(testPin);
  REQUIRE(!result.has_value());
  REQUIRE(!interceptedDestination.empty());
  REQUIRE(!std::filesystem::exists(interceptedDestination));

  automix::renderers::PhaseLimiterDiscovery::resetDownloadFetcherForTesting();
  automix::renderers::PhaseLimiterDiscovery::resetAttemptedDownloadForTesting();
}

TEST_CASE("PhaseLimiter defaultPhaseLimiterDownloadPin supports AUTOMIX_PHASELIMITER_DOWNLOAD_URL override", "[phaselimiter][discovery]") {
  std::string previousUrl;
#if defined(_WIN32)
  char* oldValue = nullptr;
  size_t oldLength = 0;
  if (_dupenv_s(&oldValue, &oldLength, "AUTOMIX_PHASELIMITER_DOWNLOAD_URL") == 0 && oldValue != nullptr) {
    previousUrl.assign(oldValue, oldLength > 0 ? oldLength - 1 : 0);
    free(oldValue);
  }
#else
  if (const char* existing = std::getenv("AUTOMIX_PHASELIMITER_DOWNLOAD_URL"); existing != nullptr) {
    previousUrl = existing;
  }
#endif

  setEnvValue("AUTOMIX_PHASELIMITER_DOWNLOAD_URL", "https://example.com/custom_phaselimiter.zip");

  const auto pin = automix::renderers::defaultPhaseLimiterDownloadPin();
  REQUIRE(pin.has_value());
  REQUIRE(pin->url == "https://example.com/custom_phaselimiter.zip");

  setEnvValue("AUTOMIX_PHASELIMITER_DOWNLOAD_URL", previousUrl);
}

TEST_CASE("PhaseLimiter pin table comes from the generated header", "[phaselimiter]") {
  const auto table = automix::renderers::phaseLimiterDownloadPinTable();
  REQUIRE(table.size() == std::size(automix::renderers::kPhaseLimiterPins));

  const std::set<std::string> validKeys = {
      "windows-x64", "macos-arm64", "macos-x86_64", "linux-x64", "linux-arm64"
  };
  const std::regex shaRegex("^[0-9a-f]{64}$");

  std::set<std::string> seenKeys;
  for (const auto& pin : table) {
    REQUIRE(validKeys.find(pin.platformKey) != validKeys.end());
    REQUIRE(pin.url.rfind("https://github.com/", 0) == 0);
    REQUIRE(std::regex_match(pin.sha256, shaRegex));

    REQUIRE(seenKeys.find(pin.platformKey) == seenKeys.end());
    seenKeys.insert(pin.platformKey);
  }
}

