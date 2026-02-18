#include "commands/CommandRegistry.h"
#include "commands/DevToolsUtils.h"

#include <filesystem>
#include <iostream>
#include <map>
#include <set>

#include "ai/AutoMasterStrategyAI.h"
#include "ai/AutoMixStrategyAI.h"
#include "analysis/StemAnalyzer.h"
#include "automaster/HeuristicAutoMasterStrategy.h"
#include "automix/HeuristicAutoMixStrategy.h"
#include "domain/JsonSerialization.h"
#include "domain/MasterPlan.h"
#include "domain/Session.h"
#include "engine/OfflineRenderPipeline.h"
#include "engine/SessionRepository.h"
#include "RegressionHarness.h"

namespace {

using namespace automix::devtools;

int commandGoldenEval(const CommandArgs& args) {
  const auto baselineArg = argValue(args, "--baseline");
  const auto baselinePath = baselineArg.has_value()
                                ? std::filesystem::path(*baselineArg)
                                : findRepoPath("tests/regression/baselines.json")
                                      .value_or(std::filesystem::path("tests/regression/baselines.json"));
  const auto workDir = std::filesystem::path(argValue(args, "--work-dir")
                                                 .value_or((std::filesystem::temp_directory_path() / "automix_golden_eval").string()));

  const auto result = automix::regression::runRegressionSuite(baselinePath, workDir);
  nlohmann::json rendered = nlohmann::json::array();
  for (const auto& metrics : result.rendered) {
    rendered.push_back({
        {"fixtureName", metrics.fixtureName},
        {"pipelineName", metrics.pipelineName},
        {"integratedLufs", metrics.metrics.integratedLufs},
        {"truePeakDbtp", metrics.metrics.truePeakDbtp},
        {"monoCorrelation", metrics.metrics.monoCorrelation},
        {"spectrumLow", metrics.metrics.spectrumLow},
        {"spectrumMid", metrics.metrics.spectrumMid},
        {"spectrumHigh", metrics.metrics.spectrumHigh},
        {"stereoCorrelation", metrics.metrics.stereoCorrelation},
    });
  }

  nlohmann::json failures = nlohmann::json::array();
  for (const auto& failure : result.failures) {
    failures.push_back({
        {"fixtureName", failure.fixtureName},
        {"pipelineName", failure.pipelineName},
        {"metricName", failure.metricName},
        {"expected", failure.expected},
        {"actual", failure.actual},
        {"tolerance", failure.tolerance},
    });
  }

  const nlohmann::json payload = {
      {"generatedAtUtc", iso8601NowUtc()},
      {"baselinePath", baselinePath.string()},
      {"workDir", workDir.string()},
      {"success", result.success},
      {"rendered", rendered},
      {"failures", failures},
  };

  const auto outPath =
      std::filesystem::path(argValue(args, "--out").value_or((workDir / "golden_eval_report.json").string()));
  writeJsonFile(outPath, payload);

  if (hasFlag(args, "--json")) {
    std::cout << payload.dump(2) << "\n";
  } else {
    std::cout << "Golden corpus evaluation\n";
    std::cout << "  Baseline: " << baselinePath.string() << "\n";
    std::cout << "  Rendered: " << result.rendered.size() << "\n";
    std::cout << "  Failures: " << result.failures.size() << "\n";
    std::cout << "  Report: " << outPath.string() << "\n";
  }

  return result.success ? 0 : 1;
}

int commandPlanDiff(const CommandArgs& args) {
  const auto sessionArg = argValue(args, "--session");
  if (!sessionArg.has_value()) {
    std::cerr << "plan-diff requires --session <session.json>\n";
    return 2;
  }

  const auto mixModelArg = argValue(args, "--mix-model");
  const auto masterModelArg = argValue(args, "--master-model");
  const auto outPath =
      std::filesystem::path(argValue(args, "--out")
                                .value_or((std::filesystem::path(*sessionArg).replace_extension(".plan_diff.json")).string()));

  automix::engine::SessionRepository repository;
  auto session = repository.load(*sessionArg);

  automix::analysis::StemAnalyzer analyzer;
  const auto analysisEntries = analyzer.analyzeSession(session);

  const double dryWet = session.mixPlan.has_value() ? session.mixPlan->dryWet : 1.0;
  automix::automix::HeuristicAutoMixStrategy heuristicMixStrategy;
  const auto heuristicMixPlan = heuristicMixStrategy.buildPlan(session, analysisEntries, dryWet);

  std::vector<std::string> inferenceNotes;
  auto mixInference = buildPlanDiffInference(mixModelArg, "mix", &inferenceNotes);
  automix::ai::AutoMixStrategyAI mixStrategyAi;
  const auto aiMixPlan = mixStrategyAi.buildPlan(session, analysisEntries, heuristicMixPlan, mixInference.get());

  auto renderSettings = session.renderSettings;
  renderSettings.outputSampleRate = renderSettings.outputSampleRate > 0 ? renderSettings.outputSampleRate : 44100;
  renderSettings.blockSize = renderSettings.blockSize > 0 ? renderSettings.blockSize : 1024;
  renderSettings.outputBitDepth = renderSettings.outputBitDepth > 0 ? renderSettings.outputBitDepth : 24;

  automix::engine::OfflineRenderPipeline pipeline;
  auto heuristicSession = session;
  heuristicSession.mixPlan = heuristicMixPlan;
  const auto heuristicRaw = pipeline.renderRawMix(heuristicSession, renderSettings, {}, nullptr);
  if (heuristicRaw.cancelled) {
    std::cerr << "plan-diff aborted: heuristic render cancelled.\n";
    return 1;
  }

  auto aiSession = session;
  aiSession.mixPlan = aiMixPlan;
  const auto aiRaw = pipeline.renderRawMix(aiSession, renderSettings, {}, nullptr);
  if (aiRaw.cancelled) {
    std::cerr << "plan-diff aborted: model render cancelled.\n";
    return 1;
  }

  automix::automaster::HeuristicAutoMasterStrategy heuristicMasterStrategy;
  const auto masterPreset = session.masterPlan.has_value() ? session.masterPlan->preset
                                                           : automix::domain::MasterPreset::DefaultStreaming;
  const auto heuristicMasterPlan = heuristicMasterStrategy.buildPlan(masterPreset, heuristicRaw.mixBuffer);
  const auto heuristicMixMetrics = analyzer.analyzeBuffer(heuristicRaw.mixBuffer);

  auto masterInference = buildPlanDiffInference(masterModelArg, "master", &inferenceNotes);
  automix::ai::AutoMasterStrategyAI masterStrategyAi;
  const auto aiMasterPlan = masterStrategyAi.buildPlan(heuristicMixMetrics, heuristicMasterPlan, masterInference.get());

  nlohmann::json stemDeltas = nlohmann::json::array();
  std::map<std::string, automix::domain::StemMixDecision> heuristicByStem;
  std::map<std::string, automix::domain::StemMixDecision> aiByStem;
  for (const auto& decision : heuristicMixPlan.stemDecisions) {
    heuristicByStem[decision.stemId] = decision;
  }
  for (const auto& decision : aiMixPlan.stemDecisions) {
    aiByStem[decision.stemId] = decision;
  }

  std::set<std::string> stemIds;
  for (const auto& [id, _] : heuristicByStem) {
    stemIds.insert(id);
  }
  for (const auto& [id, _] : aiByStem) {
    stemIds.insert(id);
  }
  for (const auto& stemId : stemIds) {
    const auto heurIt = heuristicByStem.find(stemId);
    const auto aiIt = aiByStem.find(stemId);
    const double heurGain = heurIt == heuristicByStem.end() ? 0.0 : heurIt->second.gainDb;
    const double aiGain = aiIt == aiByStem.end() ? 0.0 : aiIt->second.gainDb;
    const double heurPan = heurIt == heuristicByStem.end() ? 0.0 : heurIt->second.pan;
    const double aiPan = aiIt == aiByStem.end() ? 0.0 : aiIt->second.pan;
    const double heurHighPass = heurIt == heuristicByStem.end() ? 0.0 : heurIt->second.highPassHz;
    const double aiHighPass = aiIt == aiByStem.end() ? 0.0 : aiIt->second.highPassHz;

    stemDeltas.push_back({
        {"stemId", stemId},
        {"heuristicGainDb", heurGain},
        {"modelGainDb", aiGain},
        {"deltaGainDb", aiGain - heurGain},
        {"heuristicPan", heurPan},
        {"modelPan", aiPan},
        {"deltaPan", aiPan - heurPan},
        {"heuristicHighPassHz", heurHighPass},
        {"modelHighPassHz", aiHighPass},
        {"deltaHighPassHz", aiHighPass - heurHighPass},
    });
  }

  const double heuristicIntegrated = heuristicMasterStrategy.measureIntegratedLufs(heuristicRaw.mixBuffer);
  const double aiIntegrated = heuristicMasterStrategy.measureIntegratedLufs(aiRaw.mixBuffer);

  const nlohmann::json payload = {
      {"generatedAtUtc", iso8601NowUtc()},
      {"sessionPath", *sessionArg},
      {"inferenceNotes", inferenceNotes},
      {"mixPlan",
       {
           {"heuristic", automix::domain::Json(heuristicMixPlan)},
           {"model", automix::domain::Json(aiMixPlan)},
           {"patch", nlohmann::json::diff(automix::domain::Json(heuristicMixPlan), automix::domain::Json(aiMixPlan))},
           {"stemDeltas", stemDeltas},
       }},
      {"masterPlan",
       {
           {"heuristic", automix::domain::Json(heuristicMasterPlan)},
           {"model", automix::domain::Json(aiMasterPlan)},
           {"patch", nlohmann::json::diff(automix::domain::Json(heuristicMasterPlan), automix::domain::Json(aiMasterPlan))},
           {"deltaTargetLufs", aiMasterPlan.targetLufs - heuristicMasterPlan.targetLufs},
           {"deltaPreGainDb", aiMasterPlan.preGainDb - heuristicMasterPlan.preGainDb},
           {"deltaLimiterCeilingDb", aiMasterPlan.limiterCeilingDb - heuristicMasterPlan.limiterCeilingDb},
           {"deltaGlueRatio", aiMasterPlan.glueRatio - heuristicMasterPlan.glueRatio},
       }},
      {"mixBusComparison",
       {
           {"heuristicIntegratedLufs", heuristicIntegrated},
           {"modelIntegratedLufs", aiIntegrated},
           {"deltaIntegratedLufs", aiIntegrated - heuristicIntegrated},
       }},
  };

  writeJsonFile(outPath, payload);
  if (hasFlag(args, "--json")) {
    std::cout << payload.dump(2) << "\n";
  } else {
    std::cout << "Generated heuristic-vs-model plan diff: " << outPath.string() << "\n";
    std::cout << "Mix patch ops: "
              << nlohmann::json::diff(automix::domain::Json(heuristicMixPlan), automix::domain::Json(aiMixPlan)).size()
              << ", master patch ops: "
              << nlohmann::json::diff(automix::domain::Json(heuristicMasterPlan), automix::domain::Json(aiMasterPlan)).size()
              << "\n";
  }

  return 0;
}

int commandEvalTrend(const CommandArgs& args) {
  const auto baselinePath = std::filesystem::path(argValue(args, "--baseline").value_or(
      findRepoPath("tests/regression/baselines.json").value_or(std::filesystem::path("tests/regression/baselines.json")).string()));
  const auto workDir = std::filesystem::path(argValue(args, "--work-dir")
                                                 .value_or((std::filesystem::temp_directory_path() / "automix_golden_eval").string()));
  const auto trendPath = std::filesystem::path(argValue(args, "--trend").value_or("artifacts/eval/golden_trend.json"));

  const auto result = automix::regression::runRegressionSuite(baselinePath, workDir);
  const nlohmann::json current = {
      {"timestampUtc", iso8601NowUtc()},
      {"baselinePath", baselinePath.string()},
      {"workDir", workDir.string()},
      {"success", result.success},
      {"renderedCount", result.rendered.size()},
      {"failureCount", result.failures.size()},
  };

  auto trend = loadJsonFile(trendPath).value_or(nlohmann::json::array());
  if (!trend.is_array()) {
    trend = nlohmann::json::array();
  }
  trend.push_back(current);
  writeJsonFile(trendPath, trend);

  nlohmann::json payload = {
      {"current", current},
      {"trendPath", trendPath.string()},
      {"historySize", trend.size()},
  };

  if (const auto outArg = argValue(args, "--out"); outArg.has_value()) {
    writeJsonFile(*outArg, payload);
  }

  if (hasFlag(args, "--json")) {
    std::cout << payload.dump(2) << "\n";
  } else {
    std::cout << "Eval trend updated: " << trendPath.string()
              << " entries=" << trend.size()
              << " failures=" << result.failures.size() << "\n";
  }

  return result.success ? 0 : 1;
}

} // namespace

void registerEvalCommands(automix::devtools::CommandRegistry& registry) {
  registry.add("golden-eval", commandGoldenEval);
  registry.add("plan-diff", commandPlanDiff);
  registry.add("eval-trend", commandEvalTrend);
}
