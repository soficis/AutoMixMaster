#pragma once

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <juce_audio_devices/juce_audio_devices.h>
#include <juce_gui_basics/juce_gui_basics.h>

#include "ai/ModelManager.h"
#include "ai/HuggingFaceModelHub.h"
#include "analysis/StemAnalyzer.h"
#include "app/style/Theme.h"
#include "automaster/HeuristicAutoMasterStrategy.h"
#include "automix/HeuristicAutoMixStrategy.h"
#include "domain/MasterPlan.h"
#include "domain/ProjectProfile.h"
#include "domain/Session.h"
#include "engine/AudioPreviewEngine.h"
#include "engine/SessionRepository.h"
#include "engine/TransportController.h"
#include "renderers/RendererRegistry.h"

#include "app/controllers/ModelController.h"
#include "app/controllers/ImportController.h"
#include "app/controllers/ExportController.h"
#include "app/controllers/ProcessingController.h"
#include "app/controllers/SessionController.h"
#include "app/controllers/PreviewController.h"
#include "app/controllers/OriginalMixController.h"
#include "app/controllers/ProfileController.h"

namespace automix::app {

class HeaderBar;
class HeroWaveform;
class TransportBar;
class ControlDeck;
class TaskCenterPanel;

/// Top-level layout component for the new UI.
/// Owns all backend objects, controllers, and UI components.
class MainLayout final : public juce::Component,
                         private juce::Timer,
                         private juce::ChangeListener,
                         private juce::AudioIODeviceCallback {
public:
  MainLayout();
  ~MainLayout() override;

  void paint(juce::Graphics& g) override;
  void resized() override;
  bool keyPressed(const juce::KeyPress& key) override;

private:
  // Timer / Listener overrides
  void timerCallback() override;
  void changeListenerCallback(juce::ChangeBroadcaster* source) override;
  void audioDeviceIOCallbackWithContext(const float* const* inputChannelData, int numInputChannels,
                                        float* const* outputChannelData, int numOutputChannels,
                                        int numSamples, const juce::AudioIODeviceCallbackContext& context) override;
  void audioDeviceAboutToStart(juce::AudioIODevice* device) override;
  void audioDeviceStopped() override;

  // Wiring helpers
  void wireHeaderCallbacks();
  void wireTransportCallbacks();
  void wireControlDeckCallbacks();
  void wireTaskCenterCallbacks();
  void wireHeroWaveformCallbacks();

  // Action handlers
  void onImport();
  void onAutoMix();
  void onAutoMaster();
  void onBatch();
  void onExport();
  void onCancel();
  void onSaveSession();
  void onLoadSession();
  void onModelsMenu();
  void onSettings();

  // UI update methods
  void updateTransportDisplay();
  void updateTransportLoopAndZoomUI();
  void rebuildPreviewBuffersAsync();
  void updateTransportFromBuffer(const engine::AudioBuffer& buffer);
  void appendTaskHistory(const juce::String& line);
  void refreshRenderers();
  void refreshCodecAvailability();
  void refreshModelPacks();
  void refreshProjectProfiles();
  void refreshStemRoutingSelectors();
  void populateMasterPresetSelectors();
  void updateMeterPanel(const automaster::MasteringReport& report);
  void applyLoadedSession(domain::Session loadedSession, const juce::String& sourcePath);

  // Query helpers
  domain::RenderSettings buildCurrentRenderSettings(const std::string& outputPath) const;
  std::string selectedExportSpeedMode() const;
  std::vector<renderers::ExternalRendererConfig> loadConfiguredExternalRenderers() const;

  // ── UI Components ───────────────────────────────────────────────
  juce::TooltipWindow tooltipWindow_{this, 400};
  std::unique_ptr<HeaderBar> headerBar_;
  std::unique_ptr<HeroWaveform> heroWaveform_;
  std::unique_ptr<TransportBar> transportBar_;
  std::unique_ptr<ControlDeck> controlDeck_;
  std::unique_ptr<TaskCenterPanel> taskCenter_;

  // ── Backend Objects ─────────────────────────────────────────────
  domain::Session session_;
  analysis::StemAnalyzer analyzer_;
  automix::HeuristicAutoMixStrategy autoMixStrategy_;
  automaster::HeuristicAutoMasterStrategy autoMasterStrategy_;
  engine::SessionRepository sessionRepository_;
  engine::AudioPreviewEngine previewEngine_;
  engine::TransportController transportController_;
  ai::ModelManager modelManager_;
  juce::AudioDeviceManager audioDeviceManager_;

  // Threading
  juce::ThreadPool backgroundPool_{3};
  std::atomic_bool cancelRender_{false};
  std::atomic_bool taskRunning_{false};
  std::atomic_uint64_t previewBuildGeneration_{0};
  std::atomic<int64_t> playbackCursorSamples_{0};

  // Audio buffer management
  std::mutex playbackBufferMutex_;
  engine::AudioBuffer playbackBuffer_;

  // Analysis and rendering state
  std::vector<analysis::StemAnalysisEntry> analysisEntries_;
  std::vector<renderers::RendererInfo> rendererInfos_;

  // ID maps for combo box selections
  std::map<int, std::string> rendererIdByComboId_;
  std::map<int, std::string> codecFormatByComboId_;
  std::map<int, std::string> exportSpeedModeByComboId_;
  std::map<int, std::string> projectProfileIdByComboId_;
  std::map<int, domain::MasterPreset> masterPresetByComboId_;
  std::map<int, domain::MasterPreset> platformPresetByComboId_;
  std::map<int, std::string> roleModelIdByComboId_;
  std::map<int, std::string> mixModelIdByComboId_;
  std::map<int, std::string> masterModelIdByComboId_;

  // Task history
  std::vector<juce::String> taskHistoryLines_;
  std::vector<domain::ProjectProfile> projectProfiles_;

  // File choosers
  std::unique_ptr<juce::FileChooser> importChooser_;
  std::unique_ptr<juce::FileChooser> exportChooser_;
  std::unique_ptr<juce::FileChooser> saveSessionChooser_;
  std::unique_ptr<juce::FileChooser> loadSessionChooser_;
  std::unique_ptr<juce::FileChooser> batchImportChooser_;

  // Controllers
  std::unique_ptr<ModelController> modelController_;
  std::unique_ptr<ImportController> importController_;
  std::unique_ptr<ExportController> exportController_;
  std::unique_ptr<ProcessingController> processingController_;
  std::unique_ptr<PreviewController> previewController_;

  // Layout constants
  static constexpr int kHeaderHeight = 48;
  static constexpr int kTransportHeight = 56;
  static constexpr int kTaskCenterHeight = 80;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MainLayout)
};

} // namespace automix::app
