#pragma once

#include <atomic>
#include <map>
#include <functional>
#include <string>
#include <vector>

#include <juce_core/juce_core.h>
#include <juce_events/juce_events.h>

#include "analysis/StemAnalyzer.h"
#include "domain/ProjectProfile.h"
#include "domain/Session.h"
#include "renderers/RendererRegistry.h"
#include "util/WavWriter.h"

namespace automix::app {

struct ExportResult {
  bool success = false;
  bool cancelled = false;
  std::string rendererName;
  std::string outputAudioPath;
  std::string reportPath;
  std::string exportSpeedMode;
  std::vector<std::string> logs;
  std::vector<analysis::StemAnalysisEntry> analysisEntries;
  juce::String healthText;
  bool healthHasCriticalIssues = false;
  size_t healthIssueCount = 0;
  juce::String crashMessage;
};

struct ExportPreflightRequest {
  std::string selectedRendererId;
  std::string safetyPolicyId;
  std::string projectProfileId;
  std::vector<domain::ProjectProfile> projectProfiles;
};

struct ExportPreflightResult {
  bool allowed = true;
  juce::String statusText;
  juce::String taskHistoryText;
};

struct ExportCodecControls {
  bool formatEnabled = true;
  bool bitrateEnabled = false;
  bool mp3ModeEnabled = false;
  bool mp3VbrEnabled = false;
};

struct QuickExportDefaults {
  std::string outputFormat = "wav";
  int lossyBitrateKbps = 320;
  int lossyQuality = 8;
  bool mp3UseVbr = true;
  int mp3VbrQuality = 0;
  bool usedFallbackCodec = false;
};

struct BuildRenderSettingsRequest {
  std::string outputPath;
  std::string exportSpeedMode;
  std::string outputFormat;
  int lossyBitrateKbps = 320;
  bool mp3UseVbr = false;
  int mp3VbrQuality = 4;
  int gpuProviderSelectionId = 1;
  std::string selectedRendererId;
  std::vector<renderers::RendererInfo> rendererInfos;
  std::string metadataPolicy;
  std::map<std::string, std::string> metadataTemplate;
};

struct ExternalRendererValidationResult {
  std::string selectedPath;
  std::string selectedName;
  bool valid = false;
  std::string diagnostics;
};

struct LamePrefetchResult {
  bool success = false;
  std::string executablePath;
  std::string detail;
};

class ExportController {
 public:
  struct Callbacks {
    std::function<void(const std::string&)> onStatus;
    std::function<void(const std::string&)> onTaskHistory;
    std::function<void(ExportResult)> onExportComplete;
    std::function<void(ExternalRendererValidationResult)> onExternalRendererValidated;
    std::function<void(LamePrefetchResult)> onLamePrefetchComplete;
  };

  ExportController(juce::ThreadPool& threadPool, Callbacks callbacks);

  std::vector<util::WavWriter::FormatAvailability> listCodecAvailability() const;
  std::string selectedExportSpeedMode(int selectedId, const std::map<int, std::string>& modeByComboId) const;
  bool isQuickExportMode(const std::string& exportSpeedMode) const;
  QuickExportDefaults quickExportDefaults(const std::map<int, std::string>& codecFormatByComboId) const;
  ExportCodecControls codecControlsFor(const std::string& selectedFormat,
                                       bool mp3UseVbr,
                                       const std::string& exportSpeedMode) const;
  domain::RenderSettings buildRenderSettings(const BuildRenderSettingsRequest& request) const;

  ExportPreflightResult preflight(const ExportPreflightRequest& request) const;

  void runExport(const domain::Session& session,
                 const domain::RenderSettings& settings,
                 const std::vector<analysis::StemAnalysisEntry>& analysisEntries,
                 std::atomic_bool& cancelFlag);
  void validateExternalRenderer(std::string selectedPath, std::string selectedName);
  void prefetchLame();

  static void clearHealthCache();

 private:
  juce::ThreadPool& threadPool_;
  Callbacks callbacks_;
};

} // namespace automix::app
