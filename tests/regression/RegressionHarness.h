#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace automix::regression {

struct RenderMetrics {
  double integratedLufs = -120.0;
  double truePeakDbtp = 0.0;
  double monoCorrelation = 1.0;
  double spectrumLow = 0.0;
  double spectrumMid = 0.0;
  double spectrumHigh = 0.0;
  double stereoCorrelation = 1.0;
};

struct PipelineMetrics {
  std::string fixtureName;
  std::string pipelineName;
  RenderMetrics metrics;
};

struct RegressionFailure {
  std::string fixtureName;
  std::string pipelineName;
  std::string metricName;
  double expected = 0.0;
  double actual = 0.0;
  double tolerance = 0.0;
};

struct RegressionRunResult {
  bool success = false;
  std::vector<PipelineMetrics> rendered;
  std::vector<RegressionFailure> failures;
};

RegressionRunResult runRegressionSuite(const std::filesystem::path& baselinePath,
                                       const std::filesystem::path& workRoot);

} // namespace automix::regression
