#include "app/ui/VerificationEngine.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <sstream>

#include <juce_gui_basics/juce_gui_basics.h>

#include "ai/FeatureSchema.h"
#include "ai/OnnxModelInference.h"
#include "analysis/StemAnalyzer.h"
#include "automix/HeuristicAutoMixStrategy.h"
#include "engine/AudioFileIO.h"
#include "engine/AudioResampler.h"
#include "engine/BatchQueueRunner.h"
#include "engine/LoudnessMeter.h"
#include "engine/OfflineRenderPipeline.h"
#include "util/FileUtils.h"
#include "util/StringUtils.h"
#include "util/WavWriter.h"

#include "app/ui/MainLayoutInternal.h"

namespace automix::app {
namespace {

using namespace detail;

// ── Verify Export Job ──────────────────────────────────────────

class VerifyExportJob final : public juce::ThreadPoolJob {
 public:
  VerificationEngine::ExportContext ctx;
  VerificationEngine::HistorySink onHistory;

  VerifyExportJob(VerificationEngine::ExportContext context,
                  VerificationEngine::HistorySink sink)
      : juce::ThreadPoolJob("VerifyExportJob"),
        ctx(std::move(context)),
        onHistory(std::move(sink)) {}

  JobStatus runJob() override {
    juce::String reportText;
    try {
      engine::AudioFileIO fileIO;
      auto exported = fileIO.readAudioFile(ctx.outputAudioPath);
      if (exported.getNumSamples() <= 0 || exported.getNumChannels() <= 0) {
        throw std::runtime_error("Exported file contained no decodable audio samples.");
      }

      engine::OfflineRenderPipeline pipeline;
      auto mixedRawResult = pipeline.renderRawMix(ctx.session, ctx.settings, {}, nullptr);
      if (mixedRawResult.cancelled || mixedRawResult.mixBuffer.getNumSamples() <= 0) {
        throw std::runtime_error("Unable to render pre-master reference mix for verification.");
      }

      auto exportedComparable = exported;
      if (std::abs(exportedComparable.getSampleRate() - mixedRawResult.mixBuffer.getSampleRate()) > 1.0e-6) {
        engine::AudioResampler resampler;
        exportedComparable = resampler.resampleLinear(exportedComparable, mixedRawResult.mixBuffer.getSampleRate());
      }

      const auto masteringDiff = analyzeDifference(mixedRawResult.mixBuffer, exportedComparable);

      std::optional<DifferenceMetrics> mixDiff;
      if (ctx.session.mixPlan.has_value()) {
        auto baselineSession = ctx.session;
        baselineSession.mixPlan.reset();
        auto baselineRawResult = pipeline.renderRawMix(baselineSession, ctx.settings, {}, nullptr);
        if (!baselineRawResult.cancelled && baselineRawResult.mixBuffer.getNumSamples() > 0) {
          mixDiff = analyzeDifference(baselineRawResult.mixBuffer, mixedRawResult.mixBuffer);
        }
      }

      engine::LoudnessMeter loudnessMeter;
      const auto preMasterMetrics = loudnessMeter.analyze(mixedRawResult.mixBuffer);
      const auto postMasterMetrics = loudnessMeter.analyze(exportedComparable);
      analysis::StemAnalyzer analyzer;
      const auto preMasterAnalysisMetrics = analyzer.analyzeBuffer(mixedRawResult.mixBuffer);
      const auto postMasterAnalysisMetrics = analyzer.analyzeBuffer(exportedComparable);

      std::optional<ai::InferenceResult> preMasterAnalysis;
      std::optional<ai::InferenceResult> postMasterAnalysis;
      std::string analysisDetails;
      if (ctx.analysisPack.has_value()) {
        const auto modelPath = ctx.analysisPack->rootPath / ctx.analysisPack->modelFile;
        if (util::toLower(modelPath.extension().string()) != ".onnx") {
          analysisDetails = "Analysis pack '" + ctx.analysisPack->id +
                            "' skipped: only ONNX analysis packs are supported in verification.";
        } else {
          ai::OnnxModelInference inference;
          configureInferenceBackend(inference, ctx.analysisPack.value(), ctx.settings.gpuExecutionProvider);
          if (!inference.loadModel(modelPath)) {
            analysisDetails = "Analysis pack '" + ctx.analysisPack->id + "' failed to load: " + modelPath.string();
          } else {
            const auto task = ctx.analysisPack->type.empty() ? std::string("analysis_model") : ctx.analysisPack->type;
            preMasterAnalysis = inference.run(ai::InferenceRequest{
                .task = task,
                .features = ai::FeatureSchemaV1::extract(preMasterAnalysisMetrics),
            });
            postMasterAnalysis = inference.run(ai::InferenceRequest{
                .task = task,
                .features = ai::FeatureSchemaV1::extract(postMasterAnalysisMetrics),
            });
            analysisDetails = inference.backendDiagnostics();
          }
        }
      }

      std::ostringstream report;
      report << std::fixed << std::setprecision(2);
      report << "Verification report for " << ctx.outputAudioPath << "\n";
      report << "Mastering applied: " << (masteringDiff.changed ? "yes" : "no") << "\n";
      report << "Audible difference (proxy): " << (masteringDiff.audiblyDifferent ? "likely yes" : "subtle/none") << "\n";
      report << "Residual vs pre-master: " << masteringDiff.residualRelativeDb << " dB\n";
      report << "Pre-master LUFS: " << preMasterMetrics.integratedLufs
             << " | Post-master LUFS: " << postMasterMetrics.integratedLufs << "\n";
      report << "Pre-master peak: " << preMasterMetrics.samplePeakDbfs
             << " dBFS | Post-master peak: " << postMasterMetrics.samplePeakDbfs << " dBFS\n";
      if (mixDiff.has_value()) {
        report << "Mixing applied: " << (mixDiff->changed ? "yes" : "no") << "\n";
        report << "Mix residual vs baseline: " << mixDiff->residualRelativeDb << " dB\n";
      } else {
        report << "Mixing applied: skipped (no mix plan available)\n";
      }
      if (ctx.analysisPack.has_value()) {
        report << "Analysis pack: " << ctx.analysisPack->id << "\n";
        if (!analysisDetails.empty()) {
          report << "Analysis backend: " << analysisDetails << "\n";
        }
        if (preMasterAnalysis.has_value() && postMasterAnalysis.has_value() &&
            preMasterAnalysis->usedModel && postMasterAnalysis->usedModel) {
          report << "Analysis outputs (pre-master): "
                 << summarizeInferenceOutputs(preMasterAnalysis.value()) << "\n";
          report << "Analysis outputs (post-master): "
                 << summarizeInferenceOutputs(postMasterAnalysis.value()) << "\n";
          report << "Analysis output deltas (post - pre): "
                 << summarizeInferenceDelta(preMasterAnalysis.value(), postMasterAnalysis.value()) << "\n";
        } else {
          report << "Analysis outputs: unavailable (model rejected task or returned no usable outputs)\n";
        }
      }
      report << "Note: audible difference uses an objective residual-energy proxy, not a psychoacoustic AB test.";
      reportText = report.str();
    } catch (const std::exception& error) {
      reportText = juce::String("Verification failed: ") + error.what();
    } catch (...) {
      reportText = "Verification failed: unknown error";
    }

    if (onHistory) {
      onHistory(reportText);
    }

    return jobHasFinished;
  }
};

// ── Verify Batch Job ───────────────────────────────────────────

class VerifyBatchJob final : public juce::ThreadPoolJob {
 public:
  VerificationEngine::BatchContext ctx;
  VerificationEngine::HistorySink onHistory;

  VerifyBatchJob(VerificationEngine::BatchContext context,
                 VerificationEngine::HistorySink sink)
      : juce::ThreadPoolJob("VerifyBatchJob"),
        ctx(std::move(context)),
        onHistory(std::move(sink)) {}

  JobStatus runJob() override {
    juce::String reportText;
    try {
      engine::BatchQueueRunner runner;
      auto items = runner.buildItemsFromFolder(ctx.inputFolder, ctx.outputFolder, ctx.recursiveScan);
      if (items.empty()) {
        throw std::runtime_error("No batch items available for verification.");
      }

      const auto resolvedFormat = util::WavWriter::resolveFormat(std::filesystem::path{}, ctx.settings.outputFormat);
      const auto requiredExtension = util::extensionForFormat(resolvedFormat);
      for (auto& item : items) {
        if (util::toLower(util::pathToUtf8(item.outputPath.extension())) != requiredExtension) {
          item.outputPath.replace_extension(requiredExtension);
        }
      }

      analysis::StemAnalyzer analyzer;
      automix::HeuristicAutoMixStrategy autoMix;
      engine::OfflineRenderPipeline pipeline;
      engine::AudioFileIO fileIO;
      engine::AudioResampler resampler;
      engine::LoudnessMeter meter;

      int verified = 0;
      int missingOutputs = 0;
      int masteringApplied = 0;
      int masteringAudible = 0;
      int mixingApplied = 0;
      int mixingAudible = 0;
      double masteringResidualSumDb = 0.0;
      double mixingResidualSumDb = 0.0;
      double loudnessDeltaSum = 0.0;
      int analysisEvaluated = 0;
      int analysisConfidenceCount = 0;
      double analysisConfidencePreSum = 0.0;
      double analysisConfidencePostSum = 0.0;
      std::string analysisPackDiagnostics;
      std::optional<std::string> firstAnalysisPreview;
      ai::OnnxModelInference analysisInference;
      bool analysisInferenceReady = false;
      std::string analysisTask = "analysis_model";

      if (ctx.analysisPack.has_value()) {
        const auto modelPath = ctx.analysisPack->rootPath / ctx.analysisPack->modelFile;
        if (util::toLower(util::pathToUtf8(modelPath.extension())) == ".onnx") {
          configureInferenceBackend(analysisInference, ctx.analysisPack.value(), ctx.settings.gpuExecutionProvider);
          if (analysisInference.loadModel(modelPath)) {
            analysisInferenceReady = true;
            analysisTask = ctx.analysisPack->type.empty() ? std::string("analysis_model") : ctx.analysisPack->type;
            analysisPackDiagnostics = analysisInference.backendDiagnostics();
          } else {
            analysisPackDiagnostics = "failed to load analysis model at " + util::pathToUtf8(modelPath);
          }
        } else {
          analysisPackDiagnostics = "analysis pack is not ONNX and was skipped";
        }
      }

      for (auto& item : items) {
        try {
          if (!std::filesystem::exists(item.outputPath)) {
            ++missingOutputs;
            continue;
          }

          auto outputBuffer = fileIO.readAudioFile(item.outputPath);
          if (outputBuffer.getNumSamples() <= 0 || outputBuffer.getNumChannels() <= 0) {
            ++missingOutputs;
            continue;
          }

          auto mixedSession = item.session;
          const auto analysisEntries = analyzer.analyzeSession(mixedSession);
          mixedSession.mixPlan = autoMix.buildPlan(mixedSession, analysisEntries, 1.0);

          auto mixedRawResult = pipeline.renderRawMix(mixedSession, ctx.settings, {}, nullptr);
          if (mixedRawResult.cancelled || mixedRawResult.mixBuffer.getNumSamples() <= 0) {
            continue;
          }

          auto baselineSession = mixedSession;
          baselineSession.mixPlan.reset();
          auto baselineRawResult = pipeline.renderRawMix(baselineSession, ctx.settings, {}, nullptr);
          if (baselineRawResult.cancelled || baselineRawResult.mixBuffer.getNumSamples() <= 0) {
            continue;
          }

          auto comparableOutput = outputBuffer;
          if (std::abs(comparableOutput.getSampleRate() - mixedRawResult.mixBuffer.getSampleRate()) > 1.0e-6) {
            comparableOutput = resampler.resampleLinear(comparableOutput, mixedRawResult.mixBuffer.getSampleRate());
          }

          const auto masteringDiff = analyzeDifference(mixedRawResult.mixBuffer, comparableOutput);
          const auto mixingDiff = analyzeDifference(baselineRawResult.mixBuffer, mixedRawResult.mixBuffer);
          const auto preMasterMetrics = meter.analyze(mixedRawResult.mixBuffer);
          const auto postMasterMetrics = meter.analyze(comparableOutput);
          const auto preMasterAnalysisMetrics = analyzer.analyzeBuffer(mixedRawResult.mixBuffer);
          const auto postMasterAnalysisMetrics = analyzer.analyzeBuffer(comparableOutput);

          ++verified;
          masteringApplied += masteringDiff.changed ? 1 : 0;
          masteringAudible += masteringDiff.audiblyDifferent ? 1 : 0;
          mixingApplied += mixingDiff.changed ? 1 : 0;
          mixingAudible += mixingDiff.audiblyDifferent ? 1 : 0;
          masteringResidualSumDb += masteringDiff.residualRelativeDb;
          mixingResidualSumDb += mixingDiff.residualRelativeDb;
          loudnessDeltaSum += (postMasterMetrics.integratedLufs - preMasterMetrics.integratedLufs);

          if (analysisInferenceReady) {
            const auto preAnalysis = analysisInference.run(ai::InferenceRequest{
                .task = analysisTask,
                .features = ai::FeatureSchemaV1::extract(preMasterAnalysisMetrics),
            });
            const auto postAnalysis = analysisInference.run(ai::InferenceRequest{
                .task = analysisTask,
                .features = ai::FeatureSchemaV1::extract(postMasterAnalysisMetrics),
            });
            if (preAnalysis.usedModel && postAnalysis.usedModel) {
              ++analysisEvaluated;
              const auto preConfidenceIt = preAnalysis.outputs.find("confidence");
              const auto postConfidenceIt = postAnalysis.outputs.find("confidence");
              if (preConfidenceIt != preAnalysis.outputs.end() && postConfidenceIt != postAnalysis.outputs.end()) {
                analysisConfidencePreSum += preConfidenceIt->second;
                analysisConfidencePostSum += postConfidenceIt->second;
                ++analysisConfidenceCount;
              }
              if (!firstAnalysisPreview.has_value()) {
                firstAnalysisPreview = "pre: " + summarizeInferenceOutputs(preAnalysis) +
                                       " | post: " + summarizeInferenceOutputs(postAnalysis);
              }
            }
          }
        } catch (...) {
          ++missingOutputs;
        }
      }

      std::ostringstream report;
      report << std::fixed << std::setprecision(2);
      report << "Batch verification summary\n";
      report << "Input folder: " << util::pathToUtf8(ctx.inputFolder) << "\n";
      report << "Output folder: " << util::pathToUtf8(ctx.outputFolder) << "\n";
      report << "Items discovered: " << items.size() << "\n";
      report << "Items verified: " << verified << "\n";
      report << "Missing/undecodable outputs: " << missingOutputs << "\n";
      if (ctx.analysisPack.has_value()) {
        report << "Analysis pack configured: " << ctx.analysisPack->id << "\n";
        if (!analysisPackDiagnostics.empty()) {
          report << "Analysis backend: " << analysisPackDiagnostics << "\n";
        }
      }

      if (verified > 0) {
        const auto count = static_cast<double>(verified);
        report << "Mixing applied: " << mixingApplied << "/" << verified
               << " (audible proxy: " << mixingAudible << "/" << verified << ")\n";
        report << "Mastering applied: " << masteringApplied << "/" << verified
               << " (audible proxy: " << masteringAudible << "/" << verified << ")\n";
        report << "Average mix residual vs baseline: " << (mixingResidualSumDb / count) << " dB\n";
        report << "Average master residual vs pre-master: " << (masteringResidualSumDb / count) << " dB\n";
        report << "Average LUFS delta (post - pre): " << (loudnessDeltaSum / count) << " LU\n";
        if (ctx.analysisPack.has_value()) {
          report << "Analysis evaluations: " << analysisEvaluated << "/" << verified << "\n";
          if (analysisConfidenceCount > 0) {
            const double confidenceCount = static_cast<double>(analysisConfidenceCount);
            report << "Average analysis confidence pre: " << (analysisConfidencePreSum / confidenceCount) << "\n";
            report << "Average analysis confidence post: " << (analysisConfidencePostSum / confidenceCount) << "\n";
          }
          if (firstAnalysisPreview.has_value()) {
            report << "Analysis output sample: " << firstAnalysisPreview.value() << "\n";
          }
        }
      } else {
        report << "No outputs were verified.\n";
      }

      report << "Note: audible difference uses residual-energy proxy, not a psychoacoustic AB test.";
      reportText = report.str();
    } catch (const std::exception& error) {
      reportText = juce::String("Batch verification failed: ") + error.what();
    } catch (...) {
      reportText = "Batch verification failed: unknown error";
    }

    if (onHistory) {
      onHistory(reportText);
    }

    return jobHasFinished;
  }
};

} // namespace

// ── Static entry points ────────────────────────────────────────

void VerificationEngine::runExportVerification(ExportContext context,
                                                juce::ThreadPool& backgroundPool,
                                                HistorySink onHistory) {
  if (context.outputAudioPath.empty()) {
    return;
  }
  backgroundPool.addJob(
      new VerifyExportJob(std::move(context), std::move(onHistory)),
      true);
}

void VerificationEngine::runBatchVerification(BatchContext context,
                                               juce::ThreadPool& backgroundPool,
                                               HistorySink onHistory) {
  if (context.outputFolder.empty()) {
    return;
  }
  backgroundPool.addJob(
      new VerifyBatchJob(std::move(context), std::move(onHistory)),
      true);
}

} // namespace automix::app
