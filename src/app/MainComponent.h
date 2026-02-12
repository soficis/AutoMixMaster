#pragma once

#include <atomic>
#include <memory>
#include <vector>

#include <juce_gui_extra/juce_gui_extra.h>

#include "analysis/StemAnalyzer.h"
#include "ai/ModelManager.h"
#include "automaster/HeuristicAutoMasterStrategy.h"
#include "automix/HeuristicAutoMixStrategy.h"
#include "domain/Session.h"

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
  void onExport();
  void refreshModelPacks();

  juce::TextButton importButton_ {"Import"};
  juce::TextButton originalMixButton_ {"Original Mix"};
  juce::TextButton autoMixButton_ {"Auto Mix"};
  juce::TextButton autoMasterButton_ {"Auto Master"};
  juce::TextButton exportButton_ {"Export"};
  juce::ToggleButton separatedStemsToggle_ {"AI-separated stems"};
  juce::Label residualBlendLabel_ {"residualBlendLabel", "Residual Blend %"};
  juce::Slider residualBlendSlider_;
  juce::ComboBox rendererBox_;
  juce::Label aiModelsLabel_ {"aiModelsLabel", "AI Models"};
  juce::ComboBox roleModelBox_;
  juce::ComboBox mixModelBox_;
  juce::ComboBox masterModelBox_;
  juce::Label statusLabel_;
  AnalysisTableModel analysisTableModel_;
  juce::TableListBox analysisTable_;
  juce::TextEditor reportEditor_;

  domain::Session session_;
  analysis::StemAnalyzer analyzer_;
  automix::HeuristicAutoMixStrategy autoMixStrategy_;
  automaster::HeuristicAutoMasterStrategy autoMasterStrategy_;
  ai::ModelManager modelManager_;
  std::vector<analysis::StemAnalysisEntry> analysisEntries_;
  std::atomic_bool cancelRender_ {false};
  std::unique_ptr<juce::FileChooser> importChooser_;
  std::unique_ptr<juce::FileChooser> originalMixChooser_;
  std::unique_ptr<juce::FileChooser> exportChooser_;
};

} // namespace automix::app
