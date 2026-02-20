#pragma once

#include <atomic>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <juce_audio_devices/juce_audio_devices.h>
#include <juce_gui_basics/juce_gui_basics.h>

#include "ai/ModelManager.h"
#include "app/style/Theme.h"
#include "automaster/HeuristicAutoMasterStrategy.h"
#include "domain/MasterPlan.h"
#include "domain/ProjectProfile.h"
#include "engine/TransportController.h"
#include "renderers/RendererRegistry.h"

#include "app/controllers/ExportController.h"
#include "app/controllers/ImportController.h"
#include "app/controllers/ModelController.h"
#include "app/controllers/ProcessingController.h"
#include "app/controllers/ProfileController.h"
#include "app/controllers/SessionController.h"
#include "app/ui/SelectionState.h"
#include "app/ui/SessionManager.h"

namespace automix::app {

class HeaderBar;
class HeroWaveform;
class TransportBar;
class ControlDeck;
class TaskCenterPanel;
class TaskOrchestrator;
class AudioPreviewManager;
class ModelBrowserPanel;

/// Top-level layout component for the UI.
/// Owns backend objects, controllers, coordinators, and UI components.
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
  void wireHeroWaveformCallbacks();

  // Action handlers
  void onImport();
  void onAutoMix();
  void onAutoMaster();
  void onAutoMixMaster(); // Pipeline: Mix → Master → Export
  void onBatch();
  void onExport();
  void onSaveSession();
  void onLoadSession();
  void onModelsDialog();
  void onSettings();

  // Import helper shared by button and drag/drop
  void importFiles(std::vector<juce::File> files);

  // Pipeline export helper (called after Auto Master completes in pipeline mode)
  void triggerPipelineExport();

  // UI update methods
  void updateTransportDisplay();
  void rebuildPreview();
  void updateTransportFromBuffer(const engine::AudioBuffer& buffer);
  void refreshRenderers();
  void refreshCodecAvailability();
  void refreshModelPacks();
  void refreshProjectProfiles();
  void populateMasterPresetSelectors();
  void updateMeterPanel(const automaster::MasteringReport& report);
  void applyLoadedSession(domain::Session loadedSession, const juce::String& sourcePath);
  void startExportVerification(const std::string& outputAudioPath);
  void startBatchVerification(const std::string& outputFolder);

  // Query helpers
  domain::RenderSettings buildCurrentRenderSettings(const std::string& outputPath) const;
  std::vector<renderers::ExternalRendererConfig> loadConfiguredExternalRenderers();

  // ── UI Components ───────────────────────────────────────────────
  juce::TooltipWindow tooltipWindow_{this, 400};
  std::unique_ptr<HeaderBar> headerBar_;
  std::unique_ptr<HeroWaveform> heroWaveform_;
  std::unique_ptr<TransportBar> transportBar_;
  std::unique_ptr<ControlDeck> controlDeck_;
  std::unique_ptr<TaskCenterPanel> taskCenter_;

  // ── Backend Objects ─────────────────────────────────────────────
  engine::TransportController transportController_;
  ai::ModelManager modelManager_;
  juce::AudioDeviceManager audioDeviceManager_;
  juce::ThreadPool backgroundPool_{3};
  std::atomic<float> outputVolume_{1.0f};

  // ── Coordinators (created in constructor body) ──────────────────
  std::unique_ptr<TaskOrchestrator> taskOrchestrator_;
  std::unique_ptr<AudioPreviewManager> previewManager_;
  SessionManager sessionManager_;

  // State
  std::vector<analysis::StemAnalysisEntry> analysisEntries_;
  std::vector<renderers::RendererInfo> rendererInfos_;
  SelectionState selectionState_;
  std::vector<domain::ProjectProfile> projectProfiles_;
  std::optional<domain::Session> exportVerificationSession_;
  std::optional<domain::RenderSettings> exportVerificationSettings_;
  std::optional<std::filesystem::path> batchVerificationInputFolder_;
  std::optional<domain::RenderSettings> batchVerificationSettings_;
  bool batchVerificationRecursiveScan_ = false;

  // File choosers
  std::unique_ptr<juce::FileChooser> importChooser_;
  std::unique_ptr<juce::FileChooser> exportChooser_;
  std::unique_ptr<juce::FileChooser> saveSessionChooser_;
  std::unique_ptr<juce::FileChooser> loadSessionChooser_;
  std::unique_ptr<juce::FileChooser> batchImportChooser_;
  std::unique_ptr<juce::FileChooser> autoMixMasterExportChooser_;

  // Pipeline state: non-empty while Auto Mix+Master pipeline is running.
  // Stores the folder path to export into after mastering completes.
  std::string pendingPipelineExportFolder_;

  // Controllers
  std::unique_ptr<ModelController> modelController_;
  std::unique_ptr<ImportController> importController_;
  std::unique_ptr<ExportController> exportController_;
  std::unique_ptr<ProcessingController> processingController_;
  std::unique_ptr<SessionController> sessionController_;

  // Layout constants
  static constexpr int kHeaderHeight = 48;
  static constexpr int kTransportHeight = 56;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MainLayout)
};

} // namespace automix::app
