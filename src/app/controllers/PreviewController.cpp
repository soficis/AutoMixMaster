#include "app/controllers/PreviewController.h"

#include <algorithm>
#include <exception>

#include "automaster/HeuristicAutoMasterStrategy.h"
#include "engine/AudioPreviewEngine.h"
#include "engine/OfflineRenderPipeline.h"
#include "util/BackgroundJob.h"
#include "util/CallbackDispatch.h"

namespace automix::app {

PreviewController::PreviewController(juce::ThreadPool& threadPool, Callbacks callbacks)
    : threadPool_(threadPool), callbacks_(std::move(callbacks)) {}

void PreviewController::rebuildPreview(PreviewBuildRequest request) {
  if (request.session.stems.empty()) {
    return;
  }

  threadPool_.addJob(new util::BackgroundJob(
                         [callbacks = callbacks_, request = std::move(request)]() mutable {
                           PreviewBuildResult result;
                           result.generation = request.generation;
                           result.previousProgress = request.previousProgress;

                           try {
                             if (!request.soloStemId.empty()) {
                               for (auto& stem : request.session.stems) {
                                 stem.enabled = stem.id == request.soloStemId;
                               }
                             }
                             if (!request.muteStemId.empty()) {
                               for (auto& stem : request.session.stems) {
                                 if (stem.id == request.muteStemId) {
                                   stem.enabled = false;
                                 }
                               }
                             }

                             auto settings = request.session.renderSettings;
                             settings.outputSampleRate = settings.outputSampleRate > 0 ? settings.outputSampleRate : 44100;
                             settings.blockSize = settings.blockSize > 0 ? settings.blockSize : 1024;
                             settings.outputBitDepth = std::clamp(settings.outputBitDepth, 16, 32);
                             settings.rendererName = "BuiltIn";
                             settings.outputPath.clear();
                             settings.outputFormat = "wav";

                             engine::OfflineRenderPipeline pipeline;
                             const auto raw = pipeline.renderRawMix(request.session, settings, {}, nullptr);
                             if (raw.cancelled || raw.mixBuffer.getNumSamples() == 0) {
                               return;
                             }

                             result.rawMix = raw.mixBuffer;
                             result.mastered = raw.mixBuffer;
                             if (request.session.masterPlan.has_value()) {
                               automaster::HeuristicAutoMasterStrategy strategy;
                               result.mastered = strategy.applyPlan(raw.mixBuffer, request.session.masterPlan.value(), nullptr);
                             }

                             engine::AudioPreviewEngine previewEngine;
                             previewEngine.setBuffers(result.rawMix, result.mastered);
                             result.preview = previewEngine.buildCrossfadedPreview(1024);
                             result.success = true;
                           } catch (const std::exception& error) {
                             result.errorText = error.what();
                           } catch (...) {
                             result.errorText = "Preview rebuild skipped: unknown error";
                           }

                           if (!result.success && result.errorText.isEmpty()) {
                             return;
                           }

                           util::dispatchCallback([callbacks, result = std::move(result)]() mutable {
                             if (callbacks.onPreviewReady) {
                               callbacks.onPreviewReady(std::move(result));
                             }
                           });
                         }),
                     true);
}

void PreviewController::applyTransportBuffer(const engine::AudioBuffer& buffer,
                                             const domain::TimelineState& timeline,
                                             engine::TransportController& transportController,
                                             std::atomic<int64_t>& playbackCursorSamples) {
  playbackCursorSamples.store(0);
  transportController.setTimeline(buffer.getNumSamples(), buffer.getSampleRate());
  transportController.setLoopRangeSeconds(timeline.loopInSeconds,
                                          timeline.loopOutSeconds,
                                          timeline.loopEnabled);
  transportController.stop();
}

} // namespace automix::app
