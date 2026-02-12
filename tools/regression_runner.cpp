#include <exception>
#include <filesystem>
#include <iostream>
#include <string>

#include "RegressionHarness.h"

namespace {

std::filesystem::path baselinePathFromArgs(const int argc, const char* const* argv) {
  if (argc >= 3 && std::string(argv[1]) == "--baseline") {
    return argv[2];
  }
  return std::filesystem::path(AUTOMIX_SOURCE_DIR) / "tests/regression/baselines.json";
}

} // namespace

int main(const int argc, const char* const* argv) {
  try {
    const auto baselinePath = baselinePathFromArgs(argc, argv);
    const auto workRoot = std::filesystem::temp_directory_path() / "automix_regression_cli";
    const auto result = automix::regression::runRegressionSuite(baselinePath, workRoot);

    std::cout << "Rendered pipelines: " << result.rendered.size() << "\n";
    for (const auto& rendered : result.rendered) {
      std::cout << rendered.fixtureName << " / " << rendered.pipelineName
                << " LUFS=" << rendered.metrics.integratedLufs
                << " TP=" << rendered.metrics.truePeakDbtp << "\n";
    }

    if (!result.success) {
      std::cerr << "Regression failures: " << result.failures.size() << "\n";
      for (const auto& failure : result.failures) {
        std::cerr << failure.fixtureName << " / " << failure.pipelineName << " / " << failure.metricName
                  << " expected=" << failure.expected
                  << " actual=" << failure.actual
                  << " tolerance=" << failure.tolerance << "\n";
      }
      return 1;
    }

    std::cout << "Regression suite passed.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Regression runner error: " << error.what() << "\n";
    return 2;
  } catch (...) {
    std::cerr << "Regression runner error: unknown exception\n";
    return 2;
  }
}
