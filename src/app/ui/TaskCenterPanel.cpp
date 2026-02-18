#include "app/ui/TaskCenterPanel.h"

namespace automix::app {

using namespace theme;

TaskCenterPanel::TaskCenterPanel() : progressBar_(progressValue_) {
  taskLabel_.setText("Ready", juce::dontSendNotification);
  taskLabel_.setFont(typography::body());
  taskLabel_.setColour(juce::Label::textColourId, colour(colours::text));
  taskLabel_.setJustificationType(juce::Justification::centredLeft);

  historyEditor_.setMultiLine(true);
  historyEditor_.setReadOnly(true);
  historyEditor_.setScrollbarsShown(true);
  historyEditor_.setFont(typography::caption());
  historyEditor_.setText("Task history will appear here.");

  cancelButton_.setEnabled(false);
  cancelButton_.onClick = [this] {
    if (onCancel)
      onCancel();
  };

  addAndMakeVisible(taskLabel_);
  addAndMakeVisible(progressBar_);
  addAndMakeVisible(cancelButton_);
  addAndMakeVisible(historyEditor_);
}

void TaskCenterPanel::paint(juce::Graphics& g) {
  g.fillAll(colour(colours::surface));

  // Top border
  g.setColour(colour(colours::surfaceBorder));
  g.fillRect(0, 0, getWidth(), 1);
}

void TaskCenterPanel::resized() {
  auto area = getLocalBounds().reduced(static_cast<int>(metrics::paddingMedium));

  // Top row: task label + progress + cancel
  auto topRow = area.removeFromTop(24);
  cancelButton_.setBounds(topRow.removeFromRight(64).reduced(1));
  auto progressArea = topRow.removeFromRight(std::min(200, topRow.getWidth() / 3));
  progressBar_.setBounds(progressArea.reduced(2));
  taskLabel_.setBounds(topRow);

  // Remaining: history editor
  area.removeFromTop(4);
  historyEditor_.setBounds(area);
}

void TaskCenterPanel::setCurrentTask(const juce::String& name, const juce::String& status) {
  juce::String text = name;
  if (status.isNotEmpty())
    text += " - " + status;
  taskLabel_.setText(text, juce::dontSendNotification);
}

void TaskCenterPanel::setProgress(double progress) {
  progressValue_ = progress;
  progressBar_.repaint();
}

void TaskCenterPanel::setCanCancel(bool canCancel) {
  cancelButton_.setEnabled(canCancel);
}

void TaskCenterPanel::appendHistory(const juce::String& line) {
  auto timestamp = juce::Time::getCurrentTime().toString(true, true);
  auto entry = "[" + timestamp + "] " + line;

  auto currentText = historyEditor_.getText();
  if (currentText == "Task history will appear here.") {
    historyEditor_.setText(entry + "\n", false);
  } else {
    historyEditor_.moveCaretToEnd(false);
    historyEditor_.insertTextAtCaret(entry + "\n");
  }
}

void TaskCenterPanel::clearHistory() {
  historyEditor_.setText("", false);
}

} // namespace automix::app
