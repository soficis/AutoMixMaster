#include "app/ui/TaskCenterPanel.h"

#include <algorithm>
#include <cmath>

namespace automix::app {

using namespace theme;

TaskCenterPanel::TaskCenterPanel() : progressBar_(progressValue_) {
  taskLabel_.setText("Ready", juce::dontSendNotification);
  taskLabel_.setFont(typography::body());
  taskLabel_.setColour(juce::Label::textColourId, colour(colours::text));
  taskLabel_.setJustificationType(juce::Justification::centredLeft);

  stateBadge_.setText("IDLE", juce::dontSendNotification);
  stateBadge_.setFont(typography::caption());
  stateBadge_.setJustificationType(juce::Justification::centred);
  stateBadge_.setColour(juce::Label::backgroundColourId, stateColour(TaskState::Idle));
  stateBadge_.setColour(juce::Label::textColourId, juce::Colours::white);

  progressLabel_.setText("0%", juce::dontSendNotification);
  progressLabel_.setFont(typography::caption());
  progressLabel_.setColour(juce::Label::textColourId, colour(colours::textMuted));
  progressLabel_.setJustificationType(juce::Justification::centredRight);

  historyEditor_.setMultiLine(true);
  historyEditor_.setReadOnly(true);
  historyEditor_.setCaretVisible(true);
  historyEditor_.setWantsKeyboardFocus(true);
  historyEditor_.setMouseClickGrabsKeyboardFocus(true);
  historyEditor_.setPopupMenuEnabled(true);
  historyEditor_.setScrollbarsShown(true);
  historyEditor_.setFont(juce::Font(juce::FontOptions{}
      .withName(juce::Font::getDefaultMonospacedFontName())
      .withPointHeight(12.0f)));
  historyEditor_.setText("Task history will appear here.");

  copyLogButton_.onClick = [this] {
    juce::SystemClipboard::copyTextToClipboard(historyEditor_.getText());
  };

  cancelButton_.setEnabled(false);
  cancelButton_.onClick = [this] {
    if (onCancel)
      onCancel();
  };

  addAndMakeVisible(taskLabel_);
  addAndMakeVisible(stateBadge_);
  addAndMakeVisible(progressBar_);
  addAndMakeVisible(progressLabel_);
  addAndMakeVisible(copyLogButton_);
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

  // Top row: state badge + task label + progress + cancel
  auto topRow = area.removeFromTop(24);
  cancelButton_.setBounds(topRow.removeFromRight(64).reduced(1));
  copyLogButton_.setBounds(topRow.removeFromRight(84).reduced(1));
  progressLabel_.setBounds(topRow.removeFromRight(48).reduced(1));
  auto progressArea = topRow.removeFromRight(std::min(220, topRow.getWidth() / 2));
  progressBar_.setBounds(progressArea.reduced(2));
  stateBadge_.setBounds(topRow.removeFromLeft(80).reduced(1));
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
  if (progressValue_ < 0.0) {
    progressLabel_.setText("...", juce::dontSendNotification);
  } else {
    const auto percent = static_cast<int>(std::round(std::clamp(progressValue_, 0.0, 1.0) * 100.0));
    progressLabel_.setText(juce::String(percent) + "%", juce::dontSendNotification);
  }
  progressBar_.repaint();
  progressLabel_.repaint();
}

void TaskCenterPanel::setCanCancel(bool canCancel) {
  cancelButton_.setEnabled(canCancel);
}

void TaskCenterPanel::setTaskState(TaskState state) {
  currentState_ = state;
  stateBadge_.setText(juce::String(stateLabel(state)), juce::dontSendNotification);
  stateBadge_.setColour(juce::Label::backgroundColourId, stateColour(state));
  stateBadge_.repaint();
}

void TaskCenterPanel::appendHistory(const juce::String& line) {
  const auto now = juce::Time::getCurrentTime();
  const auto timestamp = juce::String::formatted("%02d:%02d:%02d",
                                                  now.getHours(),
                                                  now.getMinutes(),
                                                  now.getSeconds());
  const auto entry = "[" + timestamp + "] " + line;
  const auto entryWithNewline = entry + "\n";

  if (!hasHistoryEntries_) {
    historyEditor_.setText(entryWithNewline, false);
    hasHistoryEntries_ = true;
    historyLineCount_ = 1;
    return;
  }

  historyEditor_.moveCaretToEnd(false);
  historyEditor_.insertTextAtCaret(entryWithNewline);
  ++historyLineCount_;

  if (historyLineCount_ <= kMaxHistoryLines) {
    return;
  }

  auto text = historyEditor_.getText();
  int scanIndex = 0;
  int removedLines = 0;
  while (removedLines < kHistoryTrimChunkLines) {
    const int newlineIndex = text.indexOfChar(scanIndex, '\n');
    if (newlineIndex < 0) {
      break;
    }
    scanIndex = newlineIndex + 1;
    ++removedLines;
  }

  if (removedLines <= 0 || scanIndex <= 0) {
    return;
  }

  historyEditor_.setText(text.substring(scanIndex), false);
  historyEditor_.moveCaretToEnd(false);
  historyLineCount_ = std::max(0, historyLineCount_ - removedLines);
  hasHistoryEntries_ = historyLineCount_ > 0;
}

void TaskCenterPanel::clearHistory() {
  historyEditor_.setText("", false);
  hasHistoryEntries_ = false;
  historyLineCount_ = 0;
}

juce::Colour TaskCenterPanel::stateColour(TaskState state) {
  switch (state) {
    case TaskState::Idle:      return colour(colours::textMuted);
    case TaskState::Running:   return colour(colours::primary);
    case TaskState::Cancelled: return colour(colours::warning);
    case TaskState::Completed: return colour(colours::success);
    case TaskState::Failed:    return colour(colours::error);
  }
  return colour(colours::textMuted);
}

const char* TaskCenterPanel::stateLabel(TaskState state) {
  switch (state) {
    case TaskState::Idle:      return "IDLE";
    case TaskState::Running:   return "RUNNING";
    case TaskState::Cancelled: return "CANCELLED";
    case TaskState::Completed: return "COMPLETED";
    case TaskState::Failed:    return "FAILED";
  }
  return "UNKNOWN";
}

} // namespace automix::app
