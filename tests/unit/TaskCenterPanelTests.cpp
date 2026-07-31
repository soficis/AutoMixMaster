#include <string>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <juce_events/juce_events.h>

#include "app/ui/MainLayout.h"
#include "app/ui/TaskCenterPanel.h"
#include "engine/BatchQueueRunner.h"

namespace {

juce::TextEditor* findHistoryEditor(automix::app::TaskCenterPanel& panel) {
  for (int i = 0; i < panel.getNumChildComponents(); ++i) {
    if (auto* editor = dynamic_cast<juce::TextEditor*>(panel.getChildComponent(i)); editor != nullptr) {
      return editor;
    }
  }
  return nullptr;
}

int countNewlines(const juce::String& text) {
  int count = 0;
  const auto asStd = text.toStdString();
  for (const char ch : asStd) {
    if (ch == '\n') {
      ++count;
    }
  }
  return count;
}

} // namespace

TEST_CASE("TaskCenterPanel trims old history entries for large logs", "[ui][taskcenter]") {
  juce::ScopedJuceInitialiser_GUI juceInit;

  automix::app::TaskCenterPanel panel;
  auto* editor = findHistoryEditor(panel);
  REQUIRE(editor != nullptr);

  for (int i = 0; i < 2200; ++i) {
    panel.appendHistory("line " + std::to_string(i));
  }

  const auto text = editor->getText();
  REQUIRE(text.contains("line 2199"));
  REQUIRE_FALSE(text.contains("line 0"));
  REQUIRE(countNewlines(text) <= 1500);
}

TEST_CASE("etaEstimate computes remaining time from throughput", "[ui][taskcenter][eta]") {
  using automix::app::TaskCenterPanel;

  SECTION("zero elapsed means zero ETA") {
    REQUIRE(TaskCenterPanel::etaEstimate(1, 2, 0.0).inSeconds() == Catch::Approx(0.0));
  }

  SECTION("half done at T leaves ~T remaining") {
    REQUIRE(TaskCenterPanel::etaEstimate(1, 2, 120.0).inSeconds() == Catch::Approx(120.0));
    REQUIRE(TaskCenterPanel::etaEstimate(3, 6, 90.0).inSeconds() == Catch::Approx(90.0));
  }

  SECTION("currentIndex == total has no remaining time") {
    REQUIRE(TaskCenterPanel::etaEstimate(2, 2, 100.0).inSeconds() == Catch::Approx(0.0));
  }

  SECTION("total == 0 does not divide by zero") {
    REQUIRE(TaskCenterPanel::etaEstimate(0, 0, 50.0).inSeconds() == Catch::Approx(0.0));
    REQUIRE(TaskCenterPanel::etaEstimate(1, 0, 50.0).inSeconds() == Catch::Approx(0.0));
  }

  SECTION("no completed items yet has zero ETA") {
    REQUIRE(TaskCenterPanel::etaEstimate(0, 5, 40.0).inSeconds() == Catch::Approx(0.0));
  }
}

TEST_CASE("TaskCenterPanel batch ETA row state transitions", "[ui][taskcenter][eta]") {
  juce::ScopedJuceInitialiser_GUI juceInit;

  automix::app::TaskCenterPanel panel;
  const juce::String emDash(static_cast<juce::juce_wchar>(0x2014));

  SECTION("no batch items shows dash") {
    REQUIRE(panel.batchEtaText() == emDash);
    REQUIRE(panel.batchCountsText() == "0 completed, 0 failed / 0 total");
  }

  SECTION("running batch with elapsed shows formatted ETA") {
    automix::engine::BatchQueueRunner::ProgressDetail detail;
    detail.itemIndex = 1;
    detail.overallFraction = 0.4;
    detail.completedCount = 1;
    detail.failedCount = 0;
    detail.totalCount = 5;
    panel.setBatchProgress(detail, 120.0);

    // currentIndex = itemIndex + 1 = 2 of 5; remaining 3 items at 60 s/item = 180 s.
    REQUIRE(panel.batchEtaText() == "ETA 03:00");
    REQUIRE(panel.batchCountsText() == "1 completed, 0 failed / 5 total");
  }

  SECTION("complete batch shows Done") {
    automix::engine::BatchQueueRunner::ProgressDetail detail;
    detail.itemIndex = 5;
    detail.overallFraction = 1.0;
    detail.completedCount = 5;
    detail.failedCount = 0;
    detail.totalCount = 5;
    panel.setBatchProgress(detail, 300.0);

    REQUIRE(panel.batchEtaText() == "Done");
    REQUIRE(panel.batchCountsText() == "5 completed, 0 failed / 5 total");
  }
}

// ── Shortcut table (ApplicationCommandManager key bindings) ─────────────

namespace {

using automix::app::ShortcutCommand;

std::optional<ShortcutCommand> commandForKey(
    const std::vector<automix::app::ShortcutEntry>& table, const juce::KeyPress& key) {
  for (const auto& entry : table)
    if (entry.defaultKey == key)
      return entry.command;
  return std::nullopt;
}

} // namespace

TEST_CASE("Shortcut table has no duplicate keybindings", "[ui][shortcuts]") {
  juce::ScopedJuceInitialiser_GUI juceInit;
  const auto& table = automix::app::shortcutTable();
  REQUIRE(table.size() >= 12);

  juce::StringArray seenKeys;
  int spaceBindings = 0;
  for (const auto& entry : table) {
    const juce::String keyDesc = entry.defaultKey.getTextDescription();
    REQUIRE_FALSE(seenKeys.contains(keyDesc));
    seenKeys.add(keyDesc);
    if (entry.defaultKey.getKeyCode() == juce::KeyPress::spaceKey)
      ++spaceBindings;
  }

  // Space must be bound exactly once (to Play / Pause).
  REQUIRE(spaceBindings == 1);
}

TEST_CASE("Shortcut table resolves the Ctrl+Shift+M / Ctrl+Shift+A conflict", "[ui][shortcuts]") {
  juce::ScopedJuceInitialiser_GUI juceInit;
  const auto& table = automix::app::shortcutTable();

  const auto ctrlShift = juce::ModifierKeys::ctrlModifier | juce::ModifierKeys::shiftModifier;
  const auto ctrl = juce::ModifierKeys::ctrlModifier;

  // Ctrl+Shift+M = Mix + Master (README-documented one-click pipeline).
  REQUIRE(commandForKey(table, juce::KeyPress('m', ctrlShift, 0)) == ShortcutCommand::autoMixMaster);
  // Ctrl+Shift+A = Auto Master.
  REQUIRE(commandForKey(table, juce::KeyPress('a', ctrlShift, 0)) == ShortcutCommand::autoMaster);
  // Ctrl+M stays Auto Mix.
  REQUIRE(commandForKey(table, juce::KeyPress('m', ctrl, 0)) == ShortcutCommand::autoMix);
  // Ctrl+S is Save; Space is Play / Pause.
  REQUIRE(commandForKey(table, juce::KeyPress('s', ctrl, 0)) == ShortcutCommand::saveSession);
  REQUIRE(commandForKey(table, juce::KeyPress(juce::KeyPress::spaceKey,
                                              juce::ModifierKeys::noModifiers, 0)) ==
          ShortcutCommand::playPause);
}

TEST_CASE("Shortcut table covers every command exactly once", "[ui][shortcuts]") {
  juce::ScopedJuceInitialiser_GUI juceInit;
  const auto& table = automix::app::shortcutTable();

  const ShortcutCommand allCommands[] = {
      ShortcutCommand::saveSession,     ShortcutCommand::loadSession,
      ShortcutCommand::import,          ShortcutCommand::autoMix,
      ShortcutCommand::autoMaster,      ShortcutCommand::autoMixMaster,
      ShortcutCommand::exportProject,   ShortcutCommand::modelsDialog,
      ShortcutCommand::undo,            ShortcutCommand::redo,
      ShortcutCommand::playPause,       ShortcutCommand::shortcutsDialog,
  };

  int seen = 0;
  for (const auto cmd : allCommands)
    for (const auto& entry : table)
      if (entry.command == cmd) {
        ++seen;
        break;
      }
  REQUIRE(seen == static_cast<int>(std::size(allCommands)));
}
