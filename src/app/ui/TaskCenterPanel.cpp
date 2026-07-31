#include "app/ui/TaskCenterPanel.h"

#include <algorithm>
#include <cmath>

// allow: SIZE_OK — batch task/queue/history display is one cohesive panel; splitting
// into an extra compilation unit would require a CMakeLists.txt change (forbidden in T2.5).

namespace automix::app {

using namespace theme;

TaskCenterPanel::TaskCenterPanel() : progressBar_(progressValue_), batchProgressBar_(batchProgressValue_) {
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

  queueHeaderLabel_.setText("Batch Queue", juce::dontSendNotification);
  queueHeaderLabel_.setFont(typography::caption());
  queueHeaderLabel_.setColour(juce::Label::textColourId, colour(colours::textMuted));
  queueHeaderLabel_.setJustificationType(juce::Justification::centredLeft);

  queueEtaLabel_.setText("ETA: --", juce::dontSendNotification);
  queueEtaLabel_.setFont(typography::caption());
  queueEtaLabel_.setColour(juce::Label::textColourId, colour(colours::textMuted));
  queueEtaLabel_.setJustificationType(juce::Justification::centredRight);

  queueThroughputLabel_.setText("--/s", juce::dontSendNotification);
  queueThroughputLabel_.setFont(typography::caption());
  queueThroughputLabel_.setColour(juce::Label::textColourId, colour(colours::textMuted));
  queueThroughputLabel_.setJustificationType(juce::Justification::centredRight);

  etaLabel_.setFont(typography::caption());
  etaLabel_.setColour(juce::Label::textColourId, colour(colours::textMuted));
  etaLabel_.setJustificationType(juce::Justification::centredLeft);

  batchCountsLabel_.setFont(typography::caption());
  batchCountsLabel_.setColour(juce::Label::textColourId, colour(colours::textMuted));
  batchCountsLabel_.setJustificationType(juce::Justification::centredRight);

  addAndMakeVisible(taskLabel_);
  addAndMakeVisible(stateBadge_);
  addAndMakeVisible(progressBar_);
  addAndMakeVisible(progressLabel_);
  addAndMakeVisible(copyLogButton_);
  addAndMakeVisible(cancelButton_);
  addAndMakeVisible(queueHeaderLabel_);
  addAndMakeVisible(queueEtaLabel_);
  addAndMakeVisible(queueThroughputLabel_);
  addAndMakeVisible(etaLabel_);
  addAndMakeVisible(batchCountsLabel_);
  addAndMakeVisible(batchProgressBar_);
  addAndMakeVisible(historyEditor_);

  updateBatchSummary();
}

void TaskCenterPanel::paint(juce::Graphics& g) {
  g.fillAll(colour(colours::surface));

  // Top border
  g.setColour(colour(colours::surfaceBorder));
  g.fillRect(0, 0, getWidth(), 1);

  if (!queueItems_.empty()) {
    auto area = getLocalBounds().reduced(static_cast<int>(metrics::paddingMedium));
    auto topRow = area.removeFromTop(24);
    auto queueArea = area.removeFromTop(static_cast<int>(queueItems_.size()) * kQueueItemHeight + 4);

    queueArea.removeFromTop(topRow.getHeight() + 24 + 4 + 24 + 4);

    g.setColour(colour(colours::surfaceBorder).withAlpha(0.3f));
    g.fillRoundedRectangle(queueArea.toFloat(), 4.0f);

    for (size_t i = 0; i < queueItems_.size() && i < static_cast<size_t>(kMaxVisibleQueueItems); ++i) {
      auto itemBounds = juce::Rectangle<float>(
          static_cast<float>(queueArea.getX()),
          static_cast<float>(queueArea.getY() + static_cast<int>(i) * kQueueItemHeight),
          static_cast<float>(queueArea.getWidth()),
          static_cast<float>(kQueueItemHeight));
      drawQueueItem(g, itemBounds, queueItems_[i], static_cast<int>(i));
    }
  }
}

void TaskCenterPanel::drawQueueItem(juce::Graphics& g, juce::Rectangle<float> bounds, const BatchQueueItem& item, int /*index*/) {
  g.setColour(colour(colours::surfaceBorder).withAlpha(0.1f));
  g.fillRect(bounds);

  if (item.progress > 0.0) {
    auto fillBounds = bounds.withWidth(bounds.getWidth() * static_cast<float>(item.progress));
    g.setColour(stateColour(item.state).withAlpha(0.2f));
    g.fillRect(fillBounds);
  }

  auto dotBounds = bounds.removeFromLeft(12).withSizeKeepingCentre(6.0f, 6.0f);
  g.setColour(stateColour(item.state));
  g.fillEllipse(dotBounds);

  bounds.removeFromLeft(4);
  g.setFont(typography::caption());
  g.setColour(colour(colours::text));
  g.drawText(item.fileName, bounds.removeFromLeft(bounds.getWidth() - 48.0f), juce::Justification::centredLeft);

  g.setColour(colour(colours::textMuted));
  g.drawText(juce::String(static_cast<int>(item.progress * 100.0f)) + "%",
             bounds, juce::Justification::centredRight);
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

  // Batch queue header row
  auto queueHeaderRow = area.removeFromTop(18);
  queueEtaLabel_.setBounds(queueHeaderRow.removeFromRight(100).reduced(1));
  queueThroughputLabel_.setBounds(queueHeaderRow.removeFromRight(80).reduced(1));
  queueHeaderLabel_.setBounds(queueHeaderRow.reduced(1));

  // Queue items
  if (!queueItems_.empty()) {
    int queueHeight = std::min(static_cast<int>(queueItems_.size()), kMaxVisibleQueueItems) * kQueueItemHeight;
    area.removeFromTop(queueHeight + 4);
  }

  // Batch ETA / summary row: ETA (left) + overall progress bar (middle) + counts (right)
  auto summaryRow = area.removeFromTop(22);
  etaLabel_.setBounds(summaryRow.removeFromLeft(92).reduced(1));
  batchCountsLabel_.setBounds(summaryRow.removeFromRight(190).reduced(1));
  batchProgressBar_.setBounds(summaryRow.reduced(2, 4));

  // Remaining: history editor
  area.removeFromTop(4);
  historyEditor_.setBounds(area);
}

void TaskCenterPanel::setQueueItems(const std::vector<BatchQueueItem>& items) {
  queueItems_ = items;
  resized();
  repaint();
}

void TaskCenterPanel::setQueueEta(const juce::String& eta) {
  queueEtaLabel_.setText("ETA: " + eta, juce::dontSendNotification);
}

void TaskCenterPanel::setQueueThroughput(const juce::String& throughput) {
  queueThroughputLabel_.setText(throughput + "/s", juce::dontSendNotification);
}

void TaskCenterPanel::moveQueueItem(int fromIndex, int toIndex) {
  if (fromIndex < 0 || fromIndex >= static_cast<int>(queueItems_.size()) ||
      toIndex < 0 || toIndex >= static_cast<int>(queueItems_.size()))
    return;
  auto item = std::move(queueItems_[static_cast<size_t>(fromIndex)]);
  queueItems_.erase(queueItems_.begin() + fromIndex);
  queueItems_.insert(queueItems_.begin() + toIndex, std::move(item));
  for (size_t i = 0; i < queueItems_.size(); ++i)
    queueItems_[i].orderIndex = static_cast<int>(i);
  resized();
  repaint();
  if (onQueueItemMoved) onQueueItemMoved(fromIndex, toIndex);
}

void TaskCenterPanel::removeQueueItem(int index) {
  if (index < 0 || index >= static_cast<int>(queueItems_.size()))
    return;
  queueItems_.erase(queueItems_.begin() + index);
  resized();
  repaint();
  if (onQueueItemRemoved) onQueueItemRemoved(index);
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

  // Live overall-fraction path: mirror into the batch summary bar while a batch is active.
  if (batchDetail_.totalCount > 0) {
    batchProgressValue_ = std::clamp(progress, 0.0, 1.0);
    batchProgressBar_.repaint();
  }
}

void TaskCenterPanel::setBatchProgress(const engine::BatchQueueRunner::ProgressDetail& detail, double elapsedSeconds) {
  batchDetail_ = detail;
  batchElapsedSeconds_ = std::max(0.0, elapsedSeconds);
  batchProgressValue_ = std::clamp(detail.overallFraction, 0.0, 1.0);
  updateBatchSummary();
  batchProgressBar_.repaint();
}

juce::String TaskCenterPanel::batchEtaText() const {
  return etaLabel_.getText();
}

juce::String TaskCenterPanel::batchCountsText() const {
  return batchCountsLabel_.getText();
}

juce::RelativeTime TaskCenterPanel::etaEstimate(int currentIndex, int total, double elapsedSeconds) {
  if (total <= 0 || currentIndex <= 0 || currentIndex >= total || elapsedSeconds <= 0.0) {
    return juce::RelativeTime::seconds(0.0);
  }
  const double avgItemSeconds = elapsedSeconds / static_cast<double>(currentIndex);
  const int remainingItems = total - currentIndex;
  return juce::RelativeTime::seconds(avgItemSeconds * static_cast<double>(remainingItems));
}

juce::String TaskCenterPanel::formatEta(const juce::RelativeTime& eta) {
  const int totalSeconds = static_cast<int>(std::ceil(eta.inSeconds()));
  const int minutes = totalSeconds / 60;
  const int seconds = totalSeconds % 60;
  return juce::String::formatted("ETA %02d:%02d", minutes, seconds);
}

void TaskCenterPanel::updateBatchSummary() {
  const size_t total = batchDetail_.totalCount;
  const size_t completed = batchDetail_.completedCount;
  const size_t failed = batchDetail_.failedCount;

  if (total == 0) {
    // No batch items: em dash "—".
    etaLabel_.setText(juce::String(static_cast<juce::juce_wchar>(0x2014)), juce::dontSendNotification);
  } else {
    const int currentIndex = static_cast<int>(batchDetail_.itemIndex) + 1;
    if (currentIndex >= static_cast<int>(total)) {
      etaLabel_.setText("Done", juce::dontSendNotification);
    } else {
      const auto eta = etaEstimate(currentIndex, static_cast<int>(total), batchElapsedSeconds_);
      etaLabel_.setText(formatEta(eta), juce::dontSendNotification);
    }
  }

  batchCountsLabel_.setText(juce::String(completed) + " completed, " + juce::String(failed) + " failed / " +
                                juce::String(total) + " total",
                            juce::dontSendNotification);
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
