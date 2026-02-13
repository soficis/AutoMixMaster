#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <vector>

#include <juce_gui_extra/juce_gui_extra.h>

#include "analysis/StemAnalyzer.h"
#include "ai/ModelManager.h"
#include "automaster/HeuristicAutoMasterStrategy.h"
#include "automix/HeuristicAutoMixStrategy.h"
#include "domain/Session.h"
#include "engine/AudioPreviewEngine.h"
#include "engine/SessionRepository.h"
#include "renderers/RendererRegistry.h"

namespace automix::app {

class MainComponent final : public juce::Component,
                            private juce::Button::Listener,
                            private juce::ComboBox::Listener,
                            private juce::Slider::Listener {
 public:
  MainComponent();
  ~MainComponent() override;

  void resized() override;

 private:
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

  void onImport();
  void onImportOriginalMix();
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
  void updateMeterPanel(const automaster::MasteringReport& report);
  void refreshModelPacks();
  void refreshRenderers();
  domain::RenderSettings buildCurrentRenderSettings(const std::string& outputPath) const;
  std::vector<renderers::ExternalRendererConfig> loadConfiguredExternalRenderers() const;

  juce::TextButton importButton_ {"Import"};
  juce::TextButton originalMixButton_ {"Original Mix"};
  juce::TextButton saveSessionButton_ {"Save Session"};
  juce::TextButton loadSessionButton_ {"Load Session"};
  juce::TextButton autoMixButton_ {"Auto Mix"};
  juce::TextButton autoMasterButton_ {"Auto Master"};
  juce::TextButton batchImportButton_ {"Batch Folder"};
  juce::TextButton previewOriginalButton_ {"Preview A"};
  juce::TextButton previewRenderedButton_ {"Preview B"};
  juce::TextButton addExternalRendererButton_ {"Add External Limiter"};
  juce::TextButton exportButton_ {"Export"};
  juce::TextButton cancelButton_ {"Cancel"};
  juce::ToggleButton separatedStemsToggle_ {"AI-separated stems"};
  juce::Label residualBlendLabel_ {"residualBlendLabel", "Residual Blend %"};
  juce::Slider residualBlendSlider_;
  juce::ComboBox rendererBox_;
  juce::Label exportFormatLabel_ {"exportFormatLabel", "Export"};
  juce::ComboBox exportFormatBox_;
  juce::Label exportBitrateLabel_ {"exportBitrateLabel", "Lossy kbps"};
  juce::Slider exportBitrateSlider_;
  juce::Label aiModelsLabel_ {"aiModelsLabel", "AI Models"};
  juce::ComboBox roleModelBox_;
  juce::ComboBox mixModelBox_;
  juce::ComboBox masterModelBox_;
  juce::Label statusLabel_;
  juce::Label meterLufsLabel_ {"meterLufsLabel", "LUFS: --"};
  juce::Label meterShortTermLabel_ {"meterShortTermLabel", "Short-term: --"};
  juce::Label meterTruePeakLabel_ {"meterTruePeakLabel", "True Peak: --"};
  AnalysisTableModel analysisTableModel_;
  juce::TableListBox analysisTable_;
  juce::TextEditor reportEditor_;

  domain::Session session_;
  analysis::StemAnalyzer analyzer_;
  automix::HeuristicAutoMixStrategy autoMixStrategy_;
  automaster::HeuristicAutoMasterStrategy autoMasterStrategy_;
  engine::SessionRepository sessionRepository_;
  engine::AudioPreviewEngine previewEngine_;
  ai::ModelManager modelManager_;
  std::vector<analysis::StemAnalysisEntry> analysisEntries_;
  std::vector<renderers::RendererInfo> rendererInfos_;
  std::vector<renderers::ExternalRendererConfig> userExternalRendererConfigs_;
  std::map<int, std::string> rendererIdByComboId_;
  std::map<int, std::string> roleModelIdByComboId_;
  std::map<int, std::string> mixModelIdByComboId_;
  std::map<int, std::string> masterModelIdByComboId_;
  std::atomic_bool cancelRender_ {false};
  std::atomic_bool taskRunning_ {false};
  std::unique_ptr<juce::FileChooser> importChooser_;
  std::unique_ptr<juce::FileChooser> originalMixChooser_;
  std::unique_ptr<juce::FileChooser> exportChooser_;
  std::unique_ptr<juce::FileChooser> batchImportChooser_;
  std::unique_ptr<juce::FileChooser> saveSessionChooser_;
  std::unique_ptr<juce::FileChooser> loadSessionChooser_;
  std::unique_ptr<juce::FileChooser> externalRendererChooser_;
};

} // namespace automix::app
