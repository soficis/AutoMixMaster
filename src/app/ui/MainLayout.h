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
#include "app/ui/VerificationEngine.h"

namespace automix::app {

// ── Keyboard shortcut table (single source of truth) ────────────────────
// Command ids start at 1000 (0 is reserved as "no command" by JUCE).

enum class ShortcutCommand : juce::CommandID {
  saveSession = 1000,
  loadSession = 1001,
  import = 1002,
  autoMix = 1003,
  autoMaster = 1004,
  autoMixMaster = 1005,
  exportProject = 1006,
  modelsDialog = 1007,
  undo = 1008,
  redo = 1009,
  playPause = 1010,
  shortcutsDialog = 1011,
};

/// One row of the shortcut table: command id, display name, default key.
struct ShortcutEntry {
  ShortcutCommand command;
  const char* name;
  juce::KeyPress defaultKey;
};

/// Every application command and its default key binding.
///
/// Header-only so MainLayout (command registration + cheatsheet dialog) and the
/// unit tests (no-duplicate-keybindings, conflict-resolution invariants) share
/// one table. A command may appear more than once (e.g. Redo has both Ctrl+Y
/// and Ctrl+Shift+Z); the keybindings themselves are unique across the table.
///
/// Conflict resolution (documented in the cheatsheet): Ctrl+Shift+M = Mix +
/// Master (the README-documented one-click pipeline), Ctrl+Shift+A = Auto
/// Master.
inline const std::vector<ShortcutEntry>& shortcutTable() {
  static const std::vector<ShortcutEntry> table = {
      {ShortcutCommand::saveSession, "Save Session",
       juce::KeyPress('s', juce::ModifierKeys::ctrlModifier, 0)},
      {ShortcutCommand::loadSession, "Load Session",
       juce::KeyPress('o', juce::ModifierKeys::ctrlModifier, 0)},
      {ShortcutCommand::import, "Import",
       juce::KeyPress('i', juce::ModifierKeys::ctrlModifier, 0)},
      {ShortcutCommand::autoMix, "Auto Mix",
       juce::KeyPress('m', juce::ModifierKeys::ctrlModifier, 0)},
      {ShortcutCommand::autoMaster, "Auto Master",
       juce::KeyPress('a',
                      juce::ModifierKeys::ctrlModifier | juce::ModifierKeys::shiftModifier, 0)},
      {ShortcutCommand::autoMixMaster, "Mix + Master",
       juce::KeyPress('m',
                      juce::ModifierKeys::ctrlModifier | juce::ModifierKeys::shiftModifier, 0)},
      {ShortcutCommand::exportProject, "Export",
       juce::KeyPress('e', juce::ModifierKeys::ctrlModifier, 0)},
      {ShortcutCommand::modelsDialog, "Models",
       juce::KeyPress('k', juce::ModifierKeys::ctrlModifier, 0)},
      {ShortcutCommand::undo, "Undo",
       juce::KeyPress('z', juce::ModifierKeys::ctrlModifier, 0)},
      {ShortcutCommand::redo, "Redo",
       juce::KeyPress('y', juce::ModifierKeys::ctrlModifier, 0)},
      {ShortcutCommand::redo, "Redo",
       juce::KeyPress('z',
                      juce::ModifierKeys::ctrlModifier | juce::ModifierKeys::shiftModifier, 0)},
      {ShortcutCommand::playPause, "Play / Pause",
       juce::KeyPress(juce::KeyPress::spaceKey, juce::ModifierKeys::noModifiers, 0)},
      {ShortcutCommand::shortcutsDialog, "Keyboard Shortcuts",
       juce::KeyPress('/', juce::ModifierKeys::ctrlModifier, 0)},
  };
  return table;
}

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
                         private juce::AudioIODeviceCallback,
                         private juce::ApplicationCommandTarget {
public:
  MainLayout();
  ~MainLayout() override;

  TaskOrchestrator* getTaskOrchestrator() const { return taskOrchestrator_.get(); }

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

  // Controller factory (extracted from constructor)
  void initControllers();
  void initComboBoxes();

  // Action handlers
  void onImport();
  void onAutoMix();
  void onAutoMaster();
  void onAutoMixMaster(); // Pipeline: Mix -> Master -> Export
  void onBatch();
  void onExport();
  void onSaveSession();
  void onLoadSession();
  void onModelsDialog();
  void onSettings();
  void onUndo();
  void onRedo();
  void onHeaderProfileSelected(const juce::String& profileId);
  bool startAiSeparationBeforeAutoMixIfNeeded();

  // Shortcut helpers (wired through the command manager)
  void onTogglePlayPause();
  void showShortcutsDialog();

  // ── ApplicationCommandTarget (keyboard shortcuts) ──────────────
  juce::ApplicationCommandTarget* getNextCommandTarget() override;
  void getAllCommands(juce::Array<juce::CommandID>& commands) override;
  void getCommandInfo(juce::CommandID commandID, juce::ApplicationCommandInfo& result) override;
  bool perform(const juce::ApplicationCommandTarget::InvocationInfo& info) override;

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
  void updateSeparationModelBadge();
  void refreshProjectProfiles();
  void populateMasterPresetSelectors();
  void updateRendererChainPreview();
  void applySessionUiSelections();
  void syncSessionUiSelections();
  void updateMeterPanel(const automaster::MasteringReport& report);
  void applyLoadedSession(domain::Session loadedSession, const juce::String& sourcePath);
  void startExportVerification(const std::string& outputAudioPath);
  void startBatchVerification(const std::string& outputFolder);

  // Query helpers
  domain::RenderSettings buildCurrentRenderSettings(const std::string& outputPath) const;
  std::optional<ai::ModelPack> resolveActiveModelPackForTask(const std::string& taskScope);
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

  // Live meter targets: written by the audio callback, read by the UI timer
  // (30 Hz, message thread) which copies them into GlowMeters::setLevels/setPeaks.
  // Initialised to the meter floor; only playback publishes real levels.
  std::atomic<float> liveMeterLeftLevel_{-60.0f};
  std::atomic<float> liveMeterRightLevel_{-60.0f};
  std::atomic<float> liveMeterLeftPeak_{-60.0f};
  std::atomic<float> liveMeterRightPeak_{-60.0f};

  // Last play state pushed to the transport bar (message thread only).
  bool transportBarPlaying_ = false;

  // ── Coordinators (created in constructor body) ──────────────────
  std::unique_ptr<TaskOrchestrator> taskOrchestrator_;
  std::unique_ptr<AudioPreviewManager> previewManager_;
  SessionManager sessionManager_;

  // State
  std::vector<analysis::StemAnalysisEntry> analysisEntries_;
  std::vector<renderers::RendererInfo> rendererInfos_;
  SelectionState selectionState_;
  std::vector<domain::ProjectProfile> projectProfiles_;
  juce::Component::SafePointer<ModelBrowserPanel> modelBrowserPanel_;
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
  bool pendingAutoMixAfterSeparationImport_ = false;
  bool skipNextAutoMixSeparationCheck_ = false;

  // Controllers
  std::unique_ptr<ModelController> modelController_;
  std::unique_ptr<ImportController> importController_;
  std::unique_ptr<ExportController> exportController_;
  std::unique_ptr<ProcessingController> processingController_;
  std::unique_ptr<SessionController> sessionController_;

  // Keyboard shortcut dispatch (owns the command/key-mapping registry)
  juce::ApplicationCommandManager commandManager_;

  // Layout constants
  static constexpr int kHeaderHeight = 48;
  static constexpr int kTransportHeight = 56;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MainLayout)
};

} // namespace automix::app
