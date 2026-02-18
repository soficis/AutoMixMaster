#include "RegressionHarness.h"

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

#include "ai/AutoMasterStrategyAI.h"
#include "ai/AutoMixStrategyAI.h"
#include "analysis/StemAnalyzer.h"
#include "automaster/HeuristicAutoMasterStrategy.h"
#include "automix/HeuristicAutoMixStrategy.h"
#include "domain/MasterPlan.h"
#include "domain/Session.h"
#include "domain/Stem.h"
#include "engine/OfflineRenderPipeline.h"
#include "renderers/BuiltInRenderer.h"
#include "util/WavWriter.h"

namespace automix::regression {
namespace {

constexpr double kPi = 3.14159265358979323846;

class DeterministicInference final : public ai::IModelInference {
 public:
  bool isAvailable() const override { return loaded_; }

  bool loadModel(const std::filesystem::path&) override {
    loaded_ = true;
    return true;
  }

  ai::InferenceResult run(const ai::InferenceRequest& request) const override {
    ai::InferenceResult result;
    if (!loaded_) {
      result.usedModel = false;
      result.logMessage = "deterministic inference not loaded";
      return result;
    }

    result.usedModel = true;
    result.logMessage = "deterministic inference";
    if (request.task == "mix_parameters") {
      result.outputs = {
          {"confidence", 0.65},
          {"global_gain_db", -0.7},
          {"global_pan_bias", 0.02},
      };
      return result;
    }

    if (request.task == "master_parameters") {
      result.outputs = {
          {"confidence", 0.70},
          {"target_lufs", -13.3},
          {"pre_gain_db", 0.8},
          {"limiter_ceiling_db", -1.1},
          {"glue_ratio", 2.4},
          {"glue_threshold_db", -19.5},
      };
      return result;
    }

    result.usedModel = false;
    result.logMessage = "unsupported task";
    return result;
  }

 private:
  bool loaded_ = false;
};

engine::AudioBuffer makeStemTone(const double sampleRate,
                                 const int samples,
                                 const double freq,
                                 const double amplitude) {
  engine::AudioBuffer buffer(2, samples, sampleRate);
  for (int i = 0; i < samples; ++i) {
    const float sample = static_cast<float>(amplitude * std::sin(2.0 * kPi * freq * i / sampleRate));
    buffer.setSample(0, i, sample);
    buffer.setSample(1, i, sample);
  }
  return buffer;
}

domain::Session buildFixtureSession(const std::filesystem::path& fixtureRoot) {
  std::filesystem::create_directories(fixtureRoot);

  const auto stemA = makeStemTone(44100.0, 44100, 220.0, 0.40);
  const auto stemB = makeStemTone(44100.0, 44100, 880.0, 0.25);
  util::WavWriter writer;
  const auto stemPathA = fixtureRoot / "bass.wav";
  const auto stemPathB = fixtureRoot / "lead.wav";
  writer.write(stemPathA, stemA, 24);
  writer.write(stemPathB, stemB, 24);

  domain::Session session;
  session.sessionName = "regression_fixture";

  domain::Stem bass;
  bass.id = "s_bass";
  bass.name = "Bass";
  bass.filePath = stemPathA.string();

  domain::Stem lead;
  lead.id = "s_lead";
  lead.name = "Lead";
  lead.filePath = stemPathB.string();

  session.stems.push_back(bass);
  session.stems.push_back(lead);
  return session;
}

RenderMetrics metricsFromReport(const std::filesystem::path& reportPath) {
  std::ifstream reportFile(reportPath);
  if (!reportFile.is_open()) {
    throw std::runtime_error("Failed to open report: " + reportPath.string());
  }

  nlohmann::json reportJson;
  reportFile >> reportJson;

  RenderMetrics metrics;
  metrics.integratedLufs = reportJson.value("integratedLufs", -120.0);
  metrics.truePeakDbtp = reportJson.value("truePeakDbtp", 0.0);
  metrics.monoCorrelation = reportJson.value("monoCorrelation", 1.0);
  metrics.spectrumLow = reportJson.value("spectrumLow", 0.0);
  metrics.spectrumMid = reportJson.value("spectrumMid", 0.0);
  metrics.spectrumHigh = reportJson.value("spectrumHigh", 0.0);
  metrics.stereoCorrelation = reportJson.value("stereoCorrelation", 1.0);
  return metrics;
}

double defaultToleranceFor(const std::string& metricName) {
  if (metricName == "integratedLufs") {
    return 1.5;
  }
  if (metricName == "truePeakDbtp") {
    return 0.35;
  }
  if (metricName == "monoCorrelation") {
    return 0.15;
  }
  if (metricName == "stereoCorrelation") {
    return 0.20;
  }
  return 0.35;
}

double toleranceForMetric(const nlohmann::json& fixture,
                          const std::string& pipeline,
                          const std::string& metricName) {
  double tolerance = defaultToleranceFor(metricName);
  if (fixture.contains("tolerances")) {
    const auto& tolerances = fixture.at("tolerances");
    if (tolerances.contains(metricName)) {
      tolerance = tolerances.at(metricName).get<double>();
    }
  }

  if (fixture.contains("overrides")) {
    const auto& overrides = fixture.at("overrides");
    if (overrides.contains(pipeline)) {
      const auto& pipelineOverrides = overrides.at(pipeline);
      if (pipelineOverrides.contains(metricName)) {
        tolerance = pipelineOverrides.at(metricName).get<double>();
      }
    }
  }

  return tolerance;
}

double metricValue(const RenderMetrics& metrics, const std::string& metricName) {
  if (metricName == "integratedLufs") {
    return metrics.integratedLufs;
  }
  if (metricName == "truePeakDbtp") {
    return metrics.truePeakDbtp;
  }
  if (metricName == "monoCorrelation") {
    return metrics.monoCorrelation;
  }
  if (metricName == "spectrumLow") {
    return metrics.spectrumLow;
  }
  if (metricName == "spectrumMid") {
    return metrics.spectrumMid;
  }
  if (metricName == "spectrumHigh") {
    return metrics.spectrumHigh;
  }
  if (metricName == "stereoCorrelation") {
    return metrics.stereoCorrelation;
  }
  throw std::runtime_error("Unknown metric name: " + metricName);
}

PipelineMetrics renderPipeline(const std::string& fixtureName,
                               const std::string& pipelineName,
                               const std::filesystem::path& fixtureRoot) {
  domain::Session session = buildFixtureSession(fixtureRoot);

  domain::RenderSettings settings;
  settings.outputSampleRate = 44100;
  settings.blockSize = 1024;
  settings.outputBitDepth = 24;
  settings.outputPath = (fixtureRoot / (pipelineName + "_master.wav")).string();

  if (pipelineName == "ai") {
    analysis::StemAnalyzer analyzer;
    automix::HeuristicAutoMixStrategy heuristicMix;
    const auto analysisEntries = analyzer.analyzeSession(session);
    const auto heuristicMixPlan = heuristicMix.buildPlan(session, analysisEntries, 1.0);

    DeterministicInference inference;
    inference.loadModel("deterministic");

    ai::AutoMixStrategyAI mixStrategyAi;
    session.mixPlan = mixStrategyAi.buildPlan(session, analysisEntries, heuristicMixPlan, &inference);

    engine::OfflineRenderPipeline offlineRender;
    const auto rawResult = offlineRender.renderRawMix(session, settings, {}, nullptr);
    if (rawResult.cancelled) {
      throw std::runtime_error("AI pipeline render was unexpectedly cancelled.");
    }

    automaster::HeuristicAutoMasterStrategy heuristicMaster;
    const auto heuristicMasterPlan = heuristicMaster.buildPlan(domain::MasterPreset::DefaultStreaming, rawResult.mixBuffer);
    const auto mixMetrics = analyzer.analyzeBuffer(rawResult.mixBuffer);

    ai::AutoMasterStrategyAI masterStrategyAi;
    session.masterPlan = masterStrategyAi.buildPlan(mixMetrics, heuristicMasterPlan, &inference);
  }

  renderers::BuiltInRenderer renderer;
  const auto renderResult = renderer.render(session, settings, {}, nullptr);
  if (!renderResult.success) {
    throw std::runtime_error("Render failed for pipeline: " + pipelineName);
  }

  PipelineMetrics output;
  output.fixtureName = fixtureName;
  output.pipelineName = pipelineName;
  output.metrics = metricsFromReport(renderResult.reportPath);
  return output;
}

void comparePipelineMetrics(const nlohmann::json& fixture,
                            const PipelineMetrics& actual,
                            std::vector<RegressionFailure>* failures) {
  const auto& expectedMetrics = fixture.at("pipelines").at(actual.pipelineName);
  constexpr std::array<const char*, 7> metricNames = {
      "integratedLufs",
      "truePeakDbtp",
      "monoCorrelation",
      "spectrumLow",
      "spectrumMid",
      "spectrumHigh",
      "stereoCorrelation",
  };

  for (const auto* metric : metricNames) {
    const std::string metricName(metric);
    if (!expectedMetrics.contains(metricName)) {
      continue;
    }
    const double expected = expectedMetrics.at(metricName).get<double>();
    const double observed = metricValue(actual.metrics, metricName);
    const double tolerance = toleranceForMetric(fixture, actual.pipelineName, metricName);
    if (std::abs(observed - expected) > tolerance) {
      failures->push_back(RegressionFailure{
          .fixtureName = actual.fixtureName,
          .pipelineName = actual.pipelineName,
          .metricName = metricName,
          .expected = expected,
          .actual = observed,
          .tolerance = tolerance,
      });
    }
  }
}

} // namespace

RegressionRunResult runRegressionSuite(const std::filesystem::path& baselinePath,
                                       const std::filesystem::path& workRoot) {
  std::ifstream baselineFile(baselinePath);
  if (!baselineFile.is_open()) {
    throw std::runtime_error("Failed to open baseline file: " + baselinePath.string());
  }

  nlohmann::json baselineJson;
  baselineFile >> baselineJson;
  if (!baselineJson.contains("fixtures") || !baselineJson.at("fixtures").is_array()) {
    throw std::runtime_error("Baseline JSON missing fixtures array.");
  }

  std::filesystem::create_directories(workRoot);
  RegressionRunResult runResult;

  for (const auto& fixture : baselineJson.at("fixtures")) {
    const std::string fixtureName = fixture.value("name", "fixture");
    if (!fixture.contains("pipelines")) {
      throw std::runtime_error("Fixture missing pipelines object: " + fixtureName);
    }

    const std::filesystem::path fixtureRoot = workRoot / fixtureName;
    std::filesystem::remove_all(fixtureRoot);
    std::filesystem::create_directories(fixtureRoot);

    const auto heuristicMetrics = renderPipeline(fixtureName, "heuristic", fixtureRoot / "heuristic");
    runResult.rendered.push_back(heuristicMetrics);
    comparePipelineMetrics(fixture, heuristicMetrics, &runResult.failures);

    const auto aiMetrics = renderPipeline(fixtureName, "ai", fixtureRoot / "ai");
    runResult.rendered.push_back(aiMetrics);
    comparePipelineMetrics(fixture, aiMetrics, &runResult.failures);
  }

  runResult.success = runResult.failures.empty();
  return runResult;
}

} // namespace automix::regression
