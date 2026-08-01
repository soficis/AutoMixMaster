#include "GoldenFileTest.h"

#include <cmath>
#include <numbers>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

#include "automaster/HeuristicAutoMasterStrategy.h"
#include "automix/HeuristicAutoMixStrategy.h"
#include "domain/Session.h"
#include "domain/Stem.h"
#include "engine/AudioBuffer.h"

namespace automix::integration {

namespace {

engine::AudioBuffer loadTestInput(const std::filesystem::path& /*inputPath*/) {
  const double sampleRate = 44100.0;
  const int samples = static_cast<int>(sampleRate * 2.0);
  engine::AudioBuffer buffer(2, samples, sampleRate);
  for (int i = 0; i < samples; ++i) {
    const double t = static_cast<double>(i) / sampleRate;
    const float sample = static_cast<float>(
        1.35 * (0.7 * std::sin(2.0 * std::numbers::pi_v<double> * 440.0 * t) +
                0.3 * std::sin(2.0 * std::numbers::pi_v<double> * 880.0 * t) +
                0.1 * std::sin(2.0 * std::numbers::pi_v<double> * 1760.0 * t)));
    buffer.setSample(0, i, sample);
    buffer.setSample(1, i, sample * 0.95f);
  }
  return buffer;
}

} // namespace

GoldenFileResult runGoldenFileTest(const GoldenFileEntry& entry,
                                   const std::filesystem::path& workDir) {
  GoldenFileResult result;
  result.entryName = entry.name;

  try {
    // Load or generate input
    auto inputBuffer = loadTestInput(entry.inputPath);
    if (inputBuffer.getNumSamples() == 0) {
      result.failureReason = "Failed to load input: " + entry.inputPath.string();
      return result;
    }

    // Apply auto-mix strategy
    automix::HeuristicAutoMixStrategy mixStrategy;
    domain::Session session;
    auto mixPlan = mixStrategy.buildPlan(session, {}, 1.0);
    const auto mixedBuffer = inputBuffer;

    // Apply auto-master strategy
    automaster::HeuristicAutoMasterStrategy masterStrategy;
    auto masterPlan = masterStrategy.buildPlan(domain::MasterPreset::DefaultStreaming, mixedBuffer);

    automaster::MasteringReport report;
    auto outputBuffer = masterStrategy.applyPlan(mixedBuffer, masterPlan, &report);

    // Measure output metrics
    result.actualIntegratedLufs = masterStrategy.measureIntegratedLufs(outputBuffer);
    result.actualTruePeakDbtp = masterStrategy.estimateTruePeakDbtp(outputBuffer, 4);

    // Compare against expected values
    const bool lufsOk = std::abs(result.actualIntegratedLufs - entry.expectedIntegratedLufs) <= entry.lufsTolerance;
    const bool peakOk = std::abs(result.actualTruePeakDbtp - entry.expectedTruePeakDbtp) <= entry.truePeakTolerance;

    result.passed = lufsOk && peakOk;
    if (!result.passed) {
      std::ostringstream oss;
      if (!lufsOk) {
        oss << "LUFS mismatch: actual=" << result.actualIntegratedLufs
            << " expected=" << entry.expectedIntegratedLufs
            << " tol=" << entry.lufsTolerance << "; ";
      }
      if (!peakOk) {
        oss << "TruePeak mismatch: actual=" << result.actualTruePeakDbtp
            << " expected=" << entry.expectedTruePeakDbtp
            << " tol=" << entry.truePeakTolerance;
      }
      result.failureReason = oss.str();
    }
  } catch (const std::exception& e) {
    result.failureReason = std::string("Exception: ") + e.what();
  }

  return result;
}

std::vector<GoldenFileEntry> loadGoldenFileManifest(const std::filesystem::path& manifestPath) {
  std::vector<GoldenFileEntry> entries;
  if (!std::filesystem::exists(manifestPath)) return entries;

  std::ifstream ifs(manifestPath);
  if (!ifs.is_open()) return entries;

  try {
    nlohmann::json j;
    ifs >> j;

    for (const auto& item : j.value("golden_files", std::vector<nlohmann::json>{})) {
      GoldenFileEntry entry;
      entry.name = item.value("name", "");
      entry.description = item.value("description", "");
      entry.inputPath = item.value("input_path", "");
      entry.expectedOutputPath = item.value("expected_output_path", "");
      entry.expectedIntegratedLufs = item.value("expected_lufs", -14.0);
      entry.expectedTruePeakDbtp = item.value("expected_true_peak", -1.0);
      entry.lufsTolerance = item.value("lufs_tolerance", 1.0);
      entry.truePeakTolerance = item.value("true_peak_tolerance", 1.5);
      entries.push_back(entry);
    }
  } catch (const std::exception&) {
    // Return empty on parse failure
  }

  return entries;
}

} // namespace automix::integration
