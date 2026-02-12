#include <filesystem>

#include <catch2/catch_test_macros.hpp>

#include "RegressionHarness.h"

TEST_CASE("Regression harness validates heuristic and AI metrics against baselines", "[regression]") {
  const std::filesystem::path baselinePath = std::filesystem::path(AUTOMIX_SOURCE_DIR) / "tests/regression/baselines.json";
  const std::filesystem::path workRoot = std::filesystem::temp_directory_path() / "automix_regression";

  const auto result = automix::regression::runRegressionSuite(baselinePath, workRoot);
  for (const auto& failure : result.failures) {
    INFO("fixture=" << failure.fixtureName << " pipeline=" << failure.pipelineName << " metric=" << failure.metricName
                    << " expected=" << failure.expected << " actual=" << failure.actual
                    << " tolerance=" << failure.tolerance);
  }

  REQUIRE(result.success);
  std::filesystem::remove_all(workRoot);
}
