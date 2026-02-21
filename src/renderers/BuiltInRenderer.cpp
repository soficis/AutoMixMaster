#include "renderers/BuiltInRenderer.h"

#include <fstream>
#include <exception>
#include <map>
#include <optional>

#include <nlohmann/json.hpp>

#include "analysis/StemAnalyzer.h"
#include "automaster/HeuristicAutoMasterStrategy.h"
#include "automaster/OriginalMixReference.h"
#include "engine/AudioFileIO.h"
#include "engine/AudioResampler.h"
#include "engine/OfflineRenderPipeline.h"
#include "util/FileUtils.h"
#include "util/MetadataPolicy.h"
#include "util/MetadataSourceResolver.h"
#include "util/WavWriter.h"

namespace automix::renderers {
namespace {

using ::automix::util::metadataSourcePath;
using ::automix::util::pathFromUtf8;
using ::automix::util::pathToUtf8;

} // namespace

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
  const bool usedSessionMasterPlan = session.masterPlan.has_value();

  if (!usedSessionMasterPlan && session.originalMixPath.has_value()) {
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
  std::map<std::string, std::string> sourceMetadata;
  if (const auto sourcePath = metadataSourcePath(session); sourcePath.has_value()) {
    try {
      engine::AudioFileIO fileIO;
      sourceMetadata = fileIO.readMetadata(sourcePath.value());
    } catch (const std::exception& error) {
      result.logs.push_back("Metadata copy skipped: " + std::string(error.what()));
    }
  }
  std::vector<std::string> metadataPolicyNotes;
  const auto exportMetadata =
      util::applyMetadataPolicy(sourceMetadata, settings.metadataPolicy, settings.metadataTemplate, &metadataPolicyNotes);
  for (const auto& note : metadataPolicyNotes) {
    result.logs.push_back(note);
  }
  const std::filesystem::path outputPath =
      settings.outputPath.empty() ? std::filesystem::path("export_master.wav")
                                  : pathFromUtf8(settings.outputPath);
  writer.write(outputPath,
               mastered,
               settings.outputBitDepth,
               settings.outputFormat,
               settings.lossyBitrateKbps,
               settings.lossyQuality,
               settings.mp3UseVbr,
               settings.mp3VbrQuality,
               exportMetadata);

  std::filesystem::path reportPath;
  if (settings.writePerExportReportJson) {
    reportPath = pathToUtf8(outputPath) + ".report.json";
    nlohmann::json report = {
        {"renderer", "BuiltIn"},
        {"outputAudioPath", pathToUtf8(outputPath)},
        {"integratedLufs", masteringReport.integratedLufs},
        {"shortTermLufs", masteringReport.shortTermLufs},
        {"loudnessRange", masteringReport.loudnessRange},
        {"samplePeakDbfs", masteringReport.samplePeakDbfs},
        {"truePeakDbtp", masteringReport.truePeakDbtp},
        {"crestDb", masteringReport.crestDb},
        {"monoCorrelation", masteringReport.monoCorrelation},
        {"spectrumLow", spectrumMetrics.lowEnergy},
        {"spectrumMid", spectrumMetrics.midEnergy},
        {"spectrumHigh", spectrumMetrics.highEnergy},
        {"stereoCorrelation", spectrumMetrics.stereoCorrelation},
        {"masterPreset", plan.presetName},
        {"masterPlanSource", usedSessionMasterPlan ? "session" : "heuristic"},
        {"mixPlanSource", session.mixPlan.has_value() ? "session" : "heuristic"},
        {"exportSpeedMode", settings.exportSpeedMode},
        {"outputFormat", settings.outputFormat},
        {"lossyBitrateKbps", settings.lossyBitrateKbps},
        {"lossyQuality", settings.lossyQuality},
        {"mp3Mode", settings.mp3UseVbr ? "vbr" : "cbr"},
        {"mp3VbrQuality", settings.mp3VbrQuality},
        {"metadataPolicy", settings.metadataPolicy},
        {"targetLufs", plan.targetLufs},
        {"targetTruePeakDbtp", plan.truePeakDbtp},
        {"limiterCeilingDb", plan.limiterCeilingDb},
        {"activeModules", masteringReport.activeModules},
        {"decisionLog", plan.decisionLog},
        {"renderLogs", renderState.logs},
    };

    std::ofstream out(reportPath);
    out << report.dump(2);
  }

  result.success = true;
  result.outputAudioPath = pathToUtf8(outputPath);
  result.reportPath = reportPath.empty() ? std::string {} : pathToUtf8(reportPath);
  result.logs.insert(result.logs.end(), renderState.logs.begin(), renderState.logs.end());
  if (!settings.writePerExportReportJson) {
    result.logs.push_back("Report sidecar disabled (.report.json not written).");
  }
  result.logs.push_back("Built-in renderer completed.");
  return result;
}

} // namespace automix::renderers
