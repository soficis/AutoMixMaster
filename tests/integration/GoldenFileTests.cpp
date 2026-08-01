#include <filesystem>

#include <catch2/catch_test_macros.hpp>

#include "GoldenFileTest.h"

TEST_CASE("Golden-file manifest loads valid entries", "[integration][golden]") {
  const std::filesystem::path manifestPath =
      std::filesystem::path(AUTOMIX_SOURCE_DIR) / "tests/integration/golden_files.json";

  const auto entries = automix::integration::loadGoldenFileManifest(manifestPath);
  REQUIRE_FALSE(entries.empty());
  REQUIRE(entries.front().name == "synthetic_dual_tone");
}

TEST_CASE("Golden-file test passes for synthetic dual tone", "[integration][golden]") {
  const std::filesystem::path manifestPath =
      std::filesystem::path(AUTOMIX_SOURCE_DIR) / "tests/integration/golden_files.json";
  const std::filesystem::path workDir = std::filesystem::temp_directory_path() / "automix_golden";

  const auto entries = automix::integration::loadGoldenFileManifest(manifestPath);
  REQUIRE_FALSE(entries.empty());

  const auto result = automix::integration::runGoldenFileTest(entries.front(), workDir);

  INFO("entry=" << result.entryName
       << " lufs=" << result.actualIntegratedLufs
       << " truePeak=" << result.actualTruePeakDbtp
       << " reason=" << result.failureReason);
  REQUIRE(result.passed);

  std::filesystem::remove_all(workDir);
}
