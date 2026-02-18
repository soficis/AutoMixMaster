#pragma once

#include <atomic>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "domain/Session.h"
#include "domain/Stem.h"
#include "domain/ProjectProfile.h"
#include "analysis/StemAnalyzer.h"
#include "renderers/RendererRegistry.h"

namespace automix::app {

/// Centralized UI state container.
/// All UI components read from AppState; controllers update it via callbacks.
/// Changes are observed through the onChange callback.
struct AppState {
  // Session
  domain::Session session;

  // Stems for display
  std::vector<domain::Stem> stems;

  // Analysis
  std::vector<analysis::StemAnalysisEntry> analysisEntries;

  // Renderer info
  std::vector<renderers::RendererInfo> rendererInfos;

  // Task state
  enum class ActiveTask { None, Import, AutoMix, AutoMaster, Batch, Export };
  ActiveTask activeTask = ActiveTask::None;
  bool taskRunning = false;
  std::string taskStatusText;
  double taskProgress = 0.0; // <0 = indeterminate

  // Transport
  bool isPlaying = false;
  double transportProgress = 0.0;
  double currentSeconds = 0.0;
  double totalSeconds = 0.0;

  // Metering
  float meterLeftDb = -60.0f;
  float meterRightDb = -60.0f;
  float peakLeftDb = -60.0f;
  float peakRightDb = -60.0f;
  double integratedLufs = -70.0;
  double shortTermLufs = -70.0;
  double truePeakDbtp = -70.0;

  // Profiles
  std::vector<domain::ProjectProfile> profiles;

  // Change notification
  std::function<void()> onChange;

  void notifyChanged() {
    if (onChange)
      onChange();
  }
};

} // namespace automix::app
