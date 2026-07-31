#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace automix::integration {

struct GoldenFileEntry {
  std::string name;
  std::string description;
  std::filesystem::path inputPath;
  std::filesystem::path expectedOutputPath;
  double expectedIntegratedLufs = -14.0;
  double expectedTruePeakDbtp = -1.0;
  double lufsTolerance = 1.0;
  double truePeakTolerance = 1.5;
};

struct GoldenFileResult {
  bool passed = false;
  std::string entryName;
  double actualIntegratedLufs = -120.0;
  double actualTruePeakDbtp = 0.0;
  std::string failureReason;
};

// Run golden-file regression test: render input, compare output metrics against expected.
GoldenFileResult runGoldenFileTest(const GoldenFileEntry& entry,
                                   const std::filesystem::path& workDir);

// Load golden-file entries from a JSON manifest.
std::vector<GoldenFileEntry> loadGoldenFileManifest(const std::filesystem::path& manifestPath);

} // namespace automix::integration
