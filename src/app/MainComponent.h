#pragma once

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include <juce_audio_devices/juce_audio_devices.h>
#include <juce_gui_extra/juce_gui_extra.h>

#include "analysis/StemAnalyzer.h"
#include "ai/ModelManager.h"
#include "ai/HuggingFaceModelHub.h"
#include "app/WaveformPreviewComponent.h"
#include "automaster/HeuristicAutoMasterStrategy.h"
#include "automix/HeuristicAutoMixStrategy.h"
#include "domain/MasterPlan.h"
#include "domain/ProjectProfile.h"
#include "domain/Session.h"
#include "engine/AudioPreviewEngine.h"
#include "engine/TransportController.h"
#include "app/controllers/OriginalMixController.h"
#include "app/controllers/ModelController.h"
#include "app/controllers/ImportController.h"
#include "app/controllers/ExportController.h"
#include "app/controllers/ProfileController.h"
#include "app/controllers/PreviewController.h"
#include "app/controllers/ProcessingController.h"
#include "app/controllers/SessionController.h"
#include "renderers/RendererRegistry.h"

namespace automix::app {

class MainComponent final : public juce::Component,
                            private juce::Button::Listener,
                            private juce::ComboBox::Listener,
                            private juce::Slider::Listener,
                            private juce::Timer,
                            private juce::ChangeListener,
                            private juce::AudioIODeviceCallback {
 public:
  MainComponent();
  ~MainComponent() override;

  void resized() override;

 private:
  enum class ActiveTask {
    None,
    Import,
    Model,
    Session,
    AutoMix,
    AutoMaster,
    Batch,
    Export,
  };

  class AnalysisTableModel final : public juce::TableListBoxModel {
   public:
    void setEntries(const std::vector<analysis::StemAnalysisEntry>* entries);
    int getNumRows() override;
    void paintRowBackground(juce::Graphics& g,
                            int rowNumber,
                            int width,
                            int height,
                            bool rowIsSelected) override;
    void paintCell(juce::Graphics& g,
                   int rowNumber,
                   int columnId,
                   int width,
                   int height,
                   bool rowIsSelected) override;

   private:
    const std::vector<analysis::StemAnalysisEntry>* entries_ = nullptr;
  };

  void buttonClicked(juce::Button* button) override;
  void comboBoxChanged(juce::ComboBox* comboBoxThatHasChanged) override;
  void sliderValueChanged(juce::Slider* slider) override;
  void timerCallback() override;
  void changeListenerCallback(juce::ChangeBroadcaster* source) override;
  void audioDeviceIOCallbackWithContext(const float* const* inputChannelData,
                                        int numInputChannels,
                                        float* const* outputChannelData,
                                        int numOutputChannels,
                                        int numSamples,
                                        const juce::AudioIODeviceCallbackContext& context) override;
  void audioDeviceAboutToStart(juce::AudioIODevice* device) override;
  void audioDeviceStopped() override;

  void onImport();
  void onImportOriginalMix();
  void onClearOriginalMix();
  void onRegenerateCachedRenders();
  void onAutoMix();
  void onAutoMaster();
  void onBatchImport();
  void onExport();
  void onCancel();
  void onSaveSession();
  void onLoadSession();
  void onPreviewOriginal();
  void onPreviewRendered();
  void onAddExternalRenderer();
  void onPrefetchLame();
  void onModelsMenu();

  void updateMeterPanel(const automaster::MasteringReport& report);
  void refreshModelPacks();
  void refreshRenderers();
  void refreshCodecAvailability();
  void updateExportCodecControls();
  std::string selectedExportSpeedMode() const;
  bool isQuickExportModeSelected() const;
  void applyQuickExportDefaults();
  void refreshProjectProfiles();
  void refreshStemRoutingSelectors();
  void applyLoadedSession(domain::Session loadedSession, const juce::String& sourcePath);
  void rebuildPreviewBuffers();
  void rebuildPreviewBuffersAsync();
  void updateTransportFromBuffer(const engine::AudioBuffer& buffer);
  void updateTransportDisplay();
  void updateTransportLoopAndZoomUI();
  void appendTaskHistory(const juce::String& line);
  void applyProjectProfile(const domain::ProjectProfile& profile);
  void populateMasterPresetSelectors();
  void initializeControllers();
  void beginCancelableTask(const juce::String& statusText,
                           const juce::String& historyText,
                           ActiveTask activeTask);
  void finishCancelableTask();
  void requestCancelForActiveTask();

  domain::MasterPreset selectedMasterPreset() const;
  domain::MasterPreset selectedPlatformPreset() const;

  domain::RenderSettings buildCurrentRenderSettings(const std::string& outputPath) const;
  std::vector<renderers::ExternalRendererConfig> loadConfiguredExternalRenderers() const;

  juce::TextButton importButton_ {"Import"};
  juce::TextButton originalMixButton_ {"Original Mix"};
  juce::TextButton clearOriginalMixButton_ {"Clear Original"};
  juce::TextButton regenerateCacheButton_ {"Regenerate Cache"};
  juce::TextButton saveSessionButton_ {"Save Session"};
  juce::TextButton loadSessionButton_ {"Load Session"};
  juce::TextButton modelsMenuButton_ {"Models"};
  juce::TextButton autoMixButton_ {"Auto Mix"};
  juce::TextButton autoMasterButton_ {"Auto Master"};
  juce::TextButton batchImportButton_ {"Batch Folder"};
  juce::TextButton previewOriginalButton_ {"Preview A"};
  juce::TextButton previewRenderedButton_ {"Preview B"};
  juce::TextButton playPauseButton_ {"Play/Pause"};
  juce::TextButton stopButton_ {"Stop"};
  juce::TextButton loopInButton_ {"Set Loop In"};
  juce::TextButton loopOutButton_ {"Set Loop Out"};
  juce::TextButton clearLoopButton_ {"Clear Loop"};
  juce::TextButton addExternalRendererButton_ {"Add External Limiter"};
  juce::TextButton prefetchLameButton_ {"Prefetch LAME"};
  juce::TextButton exportButton_ {"Export"};
  juce::TextButton cancelButton_ {"Cancel"};
  juce::ToggleButton separatedStemsToggle_ {"AI-separated stems"};
  juce::Label residualBlendLabel_ {"residualBlendLabel", "Residual Blend %"};
  juce::Slider residualBlendSlider_;
  juce::ComboBox rendererBox_;
  juce::Label exportFormatLabel_ {"exportFormatLabel", "Export"};
  juce::ComboBox exportFormatBox_;
  juce::Label exportSpeedModeLabel_ {"exportSpeedModeLabel", "Mode"};
  juce::ComboBox exportSpeedModeBox_;
  juce::Label projectProfileLabel_ {"projectProfileLabel", "Profile"};
  juce::ComboBox projectProfileBox_;
  juce::Label exportBitrateLabel_ {"exportBitrateLabel", "Lossy kbps"};
  juce::Slider exportBitrateSlider_;
  juce::Label mp3ModeLabel_ {"mp3ModeLabel", "MP3 Mode"};
  juce::ComboBox mp3ModeBox_;
  juce::Label mp3VbrLabel_ {"mp3VbrLabel", "VBR Q"};
  juce::Slider mp3VbrSlider_;
  juce::Label gpuProviderLabel_ {"gpuProviderLabel", "ML Provider"};
  juce::ComboBox gpuProviderBox_;
  juce::Label masterPresetLabel_ {"masterPresetLabel", "Master Preset"};
  juce::ComboBox masterPresetBox_;
  juce::Label platformPresetLabel_ {"platformPresetLabel", "Platform"};
  juce::ComboBox platformPresetBox_;
  juce::Label soloStemLabel_ {"soloStemLabel", "Solo"};
  juce::ComboBox soloStemBox_;
  juce::Label muteStemLabel_ {"muteStemLabel", "Mute"};
  juce::ComboBox muteStemBox_;
  juce::Slider transportSlider_;
  juce::Label zoomLabel_ {"zoomLabel", "Zoom"};
  juce::Slider zoomSlider_;
  juce::ToggleButton fineScrubToggle_ {"Fine Scrub"};
  juce::Label aiModelsLabel_ {"aiModelsLabel", "AI Models"};
  juce::ComboBox roleModelBox_;
  juce::ComboBox mixModelBox_;
  juce::ComboBox masterModelBox_;
  juce::Label statusLabel_;
  juce::Label meterLufsLabel_ {"meterLufsLabel", "LUFS: --"};
  juce::Label meterShortTermLabel_ {"meterShortTermLabel", "Short-term: --"};
  juce::Label meterTruePeakLabel_ {"meterTruePeakLabel", "True Peak: --"};
  WaveformPreviewComponent waveformPreview_;
  AnalysisTableModel analysisTableModel_;
  juce::TableListBox analysisTable_;
  juce::TextEditor reportEditor_;
  juce::Label taskCenterLabel_ {"taskCenterLabel", "Task Center"};
  juce::TextEditor taskCenterEditor_;

  domain::Session session_;
  analysis::StemAnalyzer analyzer_;
  automix::HeuristicAutoMixStrategy autoMixStrategy_;
  automaster::HeuristicAutoMasterStrategy autoMasterStrategy_;
  engine::AudioPreviewEngine previewEngine_;
  engine::TransportController transportController_;
  ai::ModelManager modelManager_;
  juce::AudioDeviceManager audioDeviceManager_;
  std::vector<analysis::StemAnalysisEntry> analysisEntries_;
  std::vector<renderers::RendererInfo> rendererInfos_;
  std::vector<renderers::ExternalRendererConfig> userExternalRendererConfigs_;
  std::map<int, std::string> rendererIdByComboId_;
  std::map<int, std::string> roleModelIdByComboId_;
  std::map<int, std::string> mixModelIdByComboId_;
  std::map<int, std::string> masterModelIdByComboId_;
  std::map<int, domain::MasterPreset> masterPresetByComboId_;
  std::map<int, domain::MasterPreset> platformPresetByComboId_;
  std::map<int, std::string> codecFormatByComboId_;
  std::map<int, std::string> exportSpeedModeByComboId_;
  std::map<int, std::string> stemIdBySoloComboId_;
  std::map<int, std::string> stemIdByMuteComboId_;
  std::map<int, std::string> projectProfileIdByComboId_;
  juce::ThreadPool backgroundPool_ {3};
  std::atomic_bool cancelImport_ {false};
  std::atomic_bool cancelModel_ {false};
  std::atomic_bool cancelSession_ {false};
  std::atomic_bool cancelMix_ {false};
  std::atomic_bool cancelMaster_ {false};
  std::atomic_bool cancelBatch_ {false};
  std::atomic_bool cancelExport_ {false};
  ActiveTask activeTask_ = ActiveTask::None;
  std::atomic_bool taskRunning_ {false};
  std::atomic_uint64_t previewBuildGeneration_ {0};
  std::atomic<int64_t> playbackCursorSamples_ {0};
  std::mutex playbackBufferMutex_;
  engine::AudioBuffer playbackBuffer_;
  bool ignoreTransportSliderChange_ = false;
  double lastFineScrubProgress_ = 0.0;
  std::unique_ptr<juce::FileChooser> importChooser_;
  std::unique_ptr<juce::FileChooser> originalMixChooser_;
  std::unique_ptr<juce::FileChooser> exportChooser_;
  std::unique_ptr<juce::FileChooser> batchImportChooser_;
  std::unique_ptr<juce::FileChooser> saveSessionChooser_;
  std::unique_ptr<juce::FileChooser> loadSessionChooser_;
  std::unique_ptr<juce::FileChooser> externalRendererChooser_;
  std::vector<juce::String> taskHistoryLines_;
  std::vector<domain::ProjectProfile> projectProfiles_;
  std::unique_ptr<ModelController> modelController_;
  std::unique_ptr<ImportController> importController_;
  std::unique_ptr<ExportController> exportController_;
  std::unique_ptr<ProfileController> profileController_;
  std::unique_ptr<PreviewController> previewController_;
  std::unique_ptr<ProcessingController> processingController_;
  std::unique_ptr<SessionController> sessionController_;
  std::unique_ptr<OriginalMixController> originalMixController_;
};

} // namespace automix::app
