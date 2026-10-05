#include <catch2/catch_test_macros.hpp>

#include "app/ui/HeroWaveform.h"
#include "app/ui/MainLayout.h"

using automix::app::HeroWaveform;
using automix::app::QuitStep;
using automix::app::nextQuitStep;

TEST_CASE("Quit flow confirms a running task before anything else", "[ui][quit]") {
  // No task: modified -> save prompt, unmodified -> quit now.
  REQUIRE(nextQuitStep(false, false, false) == QuitStep::QuitNow);
  REQUIRE(nextQuitStep(false, false, true) == QuitStep::PromptSave);
  REQUIRE(nextQuitStep(false, true, false) == QuitStep::QuitNow);
  REQUIRE(nextQuitStep(false, true, true) == QuitStep::PromptSave);
  // Task running and not yet confirmed: always ask first.
  REQUIRE(nextQuitStep(true, false, false) == QuitStep::ConfirmRunningTask);
  REQUIRE(nextQuitStep(true, false, true) == QuitStep::ConfirmRunningTask);
  // Confirmed: continue into the unsaved-changes flow.
  REQUIRE(nextQuitStep(true, true, false) == QuitStep::QuitNow);
  REQUIRE(nextQuitStep(true, true, true) == QuitStep::PromptSave);
}

TEST_CASE("Click-to-import only fires for a left click inside an empty session", "[ui][import]") {
  for (int bits = 0; bits < 16; ++bits) {
    const bool hasStems = (bits & 1) != 0;
    const bool left = (bits & 2) != 0;
    const bool clicked = (bits & 4) != 0;
    const bool inside = (bits & 8) != 0;
    const bool expected = !hasStems && left && clicked && inside;
    INFO("hasStems=" << hasStems << " left=" << left << " clicked=" << clicked << " inside=" << inside);
    REQUIRE(HeroWaveform::shouldOpenImportOnClick(hasStems, left, clicked, inside) == expected);
  }
}
