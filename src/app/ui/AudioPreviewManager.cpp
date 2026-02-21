#include "app/ui/AudioPreviewManager.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace automix::app {

namespace {

domain::StemMixDecision* findOrCreateStemDecision(domain::Session& session,
                                                  const std::string& stemId) {
  if (!session.mixPlan.has_value()) {
    session.mixPlan = domain::MixPlan{};
  }

  for (auto& decision : session.mixPlan->stemDecisions) {
    if (decision.stemId == stemId) {
      return &decision;
    }
  }

  auto& decision = session.mixPlan->stemDecisions.emplace_back();
  decision.stemId = stemId;
  return &decision;
}

double trimVolumeToGainDb(const float volume) {
  constexpr double kMinLinearGain = 0.0001;
  const double linearGain = std::clamp(static_cast<double>(volume), kMinLinearGain, 1.5);
  return 20.0 * std::log10(linearGain);
}

void applyStemDisplayStateToPreviewSession(
    domain::Session& previewSession,
    const std::vector<StemPanel::StemDisplay>& displays) {
  if (displays.empty()) {
    return;
  }

  std::unordered_map<std::string, const StemPanel::StemDisplay*> displayByStemId;
  displayByStemId.reserve(displays.size());
  bool anySoloed = false;
  for (const auto& display : displays) {
    displayByStemId.emplace(display.id, &display);
    anySoloed = anySoloed || display.solo;
  }

  for (auto& stem : previewSession.stems) {
    const auto displayIt = displayByStemId.find(stem.id);
    if (displayIt == displayByStemId.end()) {
      continue;
    }

    const auto& display = *displayIt->second;
    stem.enabled = display.enabled && !display.mute && (!anySoloed || display.solo);

    constexpr float kVolumeEpsilon = 0.0001f;
    if (std::abs(display.volume - 1.0f) <= kVolumeEpsilon) {
      continue;
    }

    auto* decision = findOrCreateStemDecision(previewSession, display.id);
    decision->gainDb += trimVolumeToGainDb(display.volume);
  }
}

} // namespace

AudioPreviewManager::AudioPreviewManager(juce::ThreadPool& pool, juce::Component* owner)
    : owner_(owner) {
  PreviewController::Callbacks cb;
  auto ownerSafe = owner_;
  cb.onPreviewReady = [this, ownerSafe](PreviewBuildResult result) {
    juce::MessageManager::callAsync([this, ownerSafe, result = std::move(result)]() mutable {
      if (!ownerSafe)
        return;

      if (result.generation < generation_.load())
        return;

      if (!result.success) {
        if (result.errorText.isNotEmpty() && onPreviewError)
          onPreviewError(result.errorText);
        return;
      }

      {
        std::lock_guard<std::mutex> lock(bufferMutex_);
        buffer_ = result.preview;
      }

      if (onPreviewReady)
        onPreviewReady(result.preview, result.previousProgress);

      if (onHistoryLine)
        onHistoryLine("Preview updated");
    });
  };
  previewController_ = std::make_unique<PreviewController>(pool, std::move(cb));
}

void AudioPreviewManager::rebuildPreview(const domain::Session& session,
                                         const std::vector<StemPanel::StemDisplay>& displays,
                                         double currentProgress) {
  if (session.stems.empty())
    return;

  domain::Session previewSession = session;
  applyStemDisplayStateToPreviewSession(previewSession, displays);

  auto gen = ++generation_;

  PreviewBuildRequest request;
  request.session = std::move(previewSession);
  request.generation = gen;
  request.previousProgress = currentProgress;

  previewController_->rebuildPreview(std::move(request));
}

void AudioPreviewManager::setBuffer(const engine::AudioBuffer& buffer) {
  std::lock_guard<std::mutex> lock(bufferMutex_);
  buffer_ = buffer;
}

std::mutex& AudioPreviewManager::bufferMutex() {
  return bufferMutex_;
}

const engine::AudioBuffer& AudioPreviewManager::buffer() const {
  return buffer_;
}

} // namespace automix::app
