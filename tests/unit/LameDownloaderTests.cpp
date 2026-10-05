#include <filesystem>
#include <fstream>
#include <set>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "util/LameDownloader.h"
#include "util/Sha256.h"

using automix::util::LameDownloader;

TEST_CASE("Every LAME download source is pinned to a SHA-256 over HTTPS", "[util][lame]") {
  const auto pins = LameDownloader::pinnedSources();
  REQUIRE(!pins.empty());

  std::set<std::string> platforms;
  for (const auto& pin : pins) {
    INFO(pin.platformKey << " " << pin.url);
    CHECK(automix::util::isSha256Hex(pin.sha256));
    CHECK(pin.url.rfind("https://", 0) == 0);
    platforms.insert(pin.platformKey);
  }
  for (const char* key : {"win32-x64", "win32-ia32", "win32-arm64", "linux-x64", "linux-arm64", "linux-arm",
                          "darwin-x64", "darwin-arm64"}) {
    INFO(key);
    CHECK(platforms.count(key) == 1);
  }
}

TEST_CASE("A LAME download that does not match its pin is rejected and deleted", "[util][lame]") {
  const auto dir = std::filesystem::temp_directory_path() / "automix_lame_pin_test";
  std::filesystem::create_directories(dir);
  const auto file = dir / "lame.zip";
  const auto write = [&file]() { std::ofstream(file, std::ios::binary) << "not really lame"; };

  write();
  const auto actual = automix::util::fileSha256(file);
  std::string detail;
  CHECK(LameDownloader::verifyDownload(file, actual, &detail));
  CHECK(std::filesystem::exists(file));

  std::string wrong = actual;
  wrong[0] = wrong[0] == '0' ? '1' : '0';
  CHECK_FALSE(LameDownloader::verifyDownload(file, wrong, &detail));
  CHECK(detail.find("mismatch") != std::string::npos);
  CHECK_FALSE(std::filesystem::exists(file));

  write();
  CHECK_FALSE(LameDownloader::verifyDownload(file, "", &detail));
  CHECK_FALSE(std::filesystem::exists(file));

  std::filesystem::remove_all(dir);
}
