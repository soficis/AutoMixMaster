#include "renderers/BuiltInRenderer.h"

#include <fstream>
#include <exception>

#include <nlohmann/json.hpp>

#include "analysis/StemAnalyzer.h"
#include "automaster/HeuristicAutoMasterStrategy.h"
#include "automaster/OriginalMixReference.h"
#include "engine/AudioFileIO.h"
#include "engine/AudioResampler.h"
#include "engine/OfflineRenderPipeline.h"
#include "util/WavWriter.h"

namespace automix::renderers {

bool BuiltInRenderer::isAvailable() const { return true; }

RenderResult BuiltInRenderer::render(const domain::Session& session,
                                     const domain::RenderSettings& settings,
                                     const ProgressCallback& onProgress,
                                     std::atomic_bool* cancelFlag) const {
  RenderResult result;
  result.rendererName = "BuiltIn";

  engine::OfflineRenderPipeline pipeline;
  auto renderState = pipeline.renderRawMix(
      session, settings,
      [&](const engine::RenderProgress& progress) {
        result.logs.push_back(progress.stage + " " + std::to_string(progress.fraction));
        if (onProgress) {
          onProgress(progress.fraction, progress.stage);
        }
      },
      cancelFlag);

  if (renderState.cancelled) {
    result.cancelled = true;
    result.logs.insert(result.logs.end(), renderState.logs.begin(), renderState.logs.end());
    return result;
  }

  automaster::HeuristicAutoMasterStrategy strategy;
  analysis::StemAnalyzer analyzer;

  domain::MasterPlan plan =
      session.masterPlan.has_value() ? session.masterPlan.value()
                                     : strategy.buildPlan(domain::MasterPreset::DefaultStreaming, renderState.mixBuffer);

  if (!session.masterPlan.has_value() && session.originalMixPath.has_value()) {
    try {
      engine::AudioFileIO fileIO;
      engine::AudioResampler resampler;
      auto originalMix = fileIO.readAudioFile(session.originalMixPath.value());
      if (originalMix.getSampleRate() != renderState.mixBuffer.getSampleRate()) {
        originalMix = resampler.resampleLinear(originalMix, renderState.mixBuffer.getSampleRate());
      }

      automaster::OriginalMixReference referenceTarget;
      plan = referenceTarget.applySoftTarget(plan, renderState.mixBuffer, originalMix, strategy, analyzer);
    } catch (const std::exception& error) {
      result.logs.push_back("Original mix reference skipped: " + std::string(error.what()));
    }
  }

  automaster::MasteringReport masteringReport;
  auto mastered = strategy.applyPlan(renderState.mixBuffer, plan, &masteringReport);
  const auto spectrumMetrics = analyzer.analyzeBuffer(mastered);

  util::WavWriter writer;
  const std::filesystem::path outputPath = settings.outputPath.empty() ? "export_master.wav" : settings.outputPath;
  writer.write(outputPath, mastered, settings.outputBitDepth);

  const std::filesystem::path reportPath = outputPath.string() + ".report.json";
  nlohmann::json report = {
      {"renderer", "BuiltIn"},
      {"outputAudioPath", outputPath.string()},
      {"integratedLufs", masteringReport.integratedLufs},
      {"truePeakDbtp", masteringReport.truePeakDbtp},
      {"spectrumLow", spectrumMetrics.lowEnergy},
      {"spectrumMid", spectrumMetrics.midEnergy},
      {"spectrumHigh", spectrumMetrics.highEnergy},
      {"stereoCorrelation", spectrumMetrics.stereoCorrelation},
      {"targetLufs", plan.targetLufs},
      {"targetTruePeakDbtp", plan.truePeakDbtp},
      {"decisionLog", plan.decisionLog},
      {"renderLogs", renderState.logs},
  };

  std::ofstream out(reportPath);
  out << report.dump(2);

  result.success = true;
  result.outputAudioPath = outputPath.string();
  result.reportPath = reportPath.string();
  result.logs.insert(result.logs.end(), renderState.logs.begin(), renderState.logs.end());
  result.logs.push_back("Built-in renderer completed.");
  return result;
}

} // namespace automix::renderers
