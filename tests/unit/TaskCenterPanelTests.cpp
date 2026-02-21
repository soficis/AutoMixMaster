#include <string>

#include <catch2/catch_test_macros.hpp>

#include <juce_events/juce_events.h>

#include "app/ui/TaskCenterPanel.h"

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
