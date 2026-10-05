#include <filesystem>
#include <fstream>
#include <iterator>
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

TEST_CASE("The published LAME pin list in the repo is valid and covers every platform", "[util][lame]") {
  std::ifstream file(std::filesystem::path(AUTOMIX_SOURCE_DIR) / "assets" / "lame-pins.json", std::ios::binary);
  REQUIRE(file.is_open());
  const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

  std::string detail;
  const auto pins = LameDownloader::parsePinManifest(text, &detail);
  INFO(detail);
  REQUIRE(!pins.empty());

  std::set<std::string> platforms;
  for (const auto& pin : pins) {
    platforms.insert(pin.platformKey);
  }
  for (const auto& builtIn : LameDownloader::pinnedSources()) {
    INFO(builtIn.platformKey);
    CHECK(platforms.count(builtIn.platformKey) == 1);
  }
}

TEST_CASE("A LAME pin list that could redirect the download is rejected whole", "[util][lame]") {
  const std::string sha(64, 'a');
  const auto list = [](const std::string& entry) { return R"({"schema":1,"sources":[)" + entry + "]}"; };
  const auto zip = [&sha](const std::string& url) {
    return R"({"platform":"win32-x64","type":"zip","url":")" + url + R"(","sha256":")" + sha + R"("})";
  };
  const std::string good = zip("https://www.rarewares.org/files/mp3/lame4.0-x64.zip");

  REQUIRE(LameDownloader::parsePinManifest(list(good), nullptr).size() == 1);
  CHECK(LameDownloader::parsePinManifest(list(R"({"platform":"darwin-arm64","type":"ghcr","sha256":")" + sha + R"("})"),
                                         nullptr)
            .size() == 1);

  std::string detail;
  for (const auto& bad : {
           zip("https://evil.example/lame.zip"),
           zip("http://www.rarewares.org/files/mp3/lame.zip"),
           zip("https://www.rarewares.org.evil.example/files/mp3/lame.zip"),
           zip("https://www.rarewares.org/files/mp3/../../x/lame.zip"),
           zip("https://www.rarewares.org/files/mp3/lame.exe"),
           zip("https://www.rarewares.org/files/mp3/lame.zip?x=.zip"),
           std::string(R"({"platform":"win32-x64","type":"zip","url":"https://www.rarewares.org/files/mp3/l.zip","sha256":"abc"})"),
           std::string(R"({"platform":"win32-x64","type":"exe","url":"https://www.rarewares.org/files/mp3/l.zip","sha256":")") +
               sha + R"("})",
       }) {
    INFO(bad);
    // One bad entry poisons the list even next to a good one.
    CHECK(LameDownloader::parsePinManifest(list(good + "," + bad), &detail).empty());
    CHECK(!detail.empty());
  }
  CHECK(LameDownloader::parsePinManifest("not json", &detail).empty());
  CHECK(LameDownloader::parsePinManifest(R"({"schema":2,"sources":[]})", &detail).empty());
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
