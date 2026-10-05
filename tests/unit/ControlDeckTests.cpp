#include <catch2/catch_test_macros.hpp>
#include <limits>

#include <juce_events/juce_events.h>
#include <juce_gui_basics/juce_gui_basics.h>

#include "app/ui/ControlDeck.h"
#include "app/ui/GlowMeters.h"

namespace {

juce::Button* findButtonByText(juce::Component& parent, const juce::String& text) {
  for (auto* child : parent.getChildren()) {
    if (auto* button = dynamic_cast<juce::Button*>(child); button != nullptr && button->getButtonText() == text)
      return button;
    if (child != nullptr) {
      if (auto* nested = findButtonByText(*child, text))
        return nested;
    }
  }
  return nullptr;
}

} // namespace

TEST_CASE("ControlDeck gates stem-dependent actions on setHasStems", "[ui][controldeck]") {
  juce::ScopedJuceInitialiser_GUI juceInit;

  automix::app::ControlDeck deck;
  deck.setSize(1000, 300);

  const auto enabled = [&](const char* name) {
    auto* button = findButtonByText(deck, name);
    REQUIRE(button != nullptr);
    return button->isEnabled();
  };

  deck.setHasStems(false);
  REQUIRE_FALSE(enabled("Auto Mix"));
  REQUIRE_FALSE(enabled("Auto Master"));
  REQUIRE_FALSE(enabled("Mix + Master"));
  REQUIRE_FALSE(enabled("Export"));
  REQUIRE(enabled("Batch"));
  REQUIRE(enabled("Import"));

  deck.setHasStems(true);
  REQUIRE(enabled("Auto Mix"));
  REQUIRE(enabled("Auto Master"));
  REQUIRE(enabled("Mix + Master"));
  REQUIRE(enabled("Export"));
  REQUIRE(enabled("Batch"));
  REQUIRE(enabled("Import"));
}

TEST_CASE("ControlDeck shows the Vocal Model toggle only with AI Stem Separation", "[ui][controldeck]") {
  juce::ScopedJuceInitialiser_GUI juceInit;

  automix::app::ControlDeck deck;
  deck.setSize(1000, 300);

  deck.setSeparationControlsVisible(false);
  REQUIRE_FALSE(deck.getTensorSeparationToggle().isVisible());

  deck.setSeparationControlsVisible(true);
  REQUIRE(deck.getTensorSeparationToggle().isVisible());
}

TEST_CASE("GlowMeters readouts show -- for no signal", "[ui][glowmeters]") {
  using automix::app::GlowMeters;
  REQUIRE(GlowMeters::formatReadout("M: ", -70.0, " LUFS") == "M: -- LUFS");
  REQUIRE(GlowMeters::formatReadout("TP: ", -69.95, " dBTP") == "TP: -- dBTP");
  REQUIRE(GlowMeters::formatReadout("I: ", -std::numeric_limits<double>::infinity(), " LUFS") == "I: -- LUFS");
  REQUIRE(GlowMeters::formatReadout("S: ", -14.0, " LUFS") == "S: -14.0 LUFS");
}

TEST_CASE("ControlDeck settings row keeps Platform combo visible and non-overlapping", "[ui][controldeck]") {
  juce::ScopedJuceInitialiser_GUI juceInit;

  for (const int width : {940, 1500}) {
    automix::app::ControlDeck deck;
    deck.setSize(width, 420);

    const auto platform = deck.getPlatformPresetBox().getBounds();
    const auto profile = deck.getProfileBox().getBounds();
    const auto master = deck.getMasterPresetBox().getBounds();
    REQUIRE_FALSE(platform.isEmpty());
    REQUIRE(deck.getLocalBounds().contains(platform));
    REQUIRE(platform.getRight() <= deck.getWidth());
    REQUIRE_FALSE(platform.intersects(profile));
    REQUIRE_FALSE(platform.intersects(master));

    auto* autoMaster = findButtonByText(deck, "Auto Master");
    auto* import = findButtonByText(deck, "Import");
    REQUIRE(autoMaster != nullptr);
    REQUIRE(import != nullptr);
    REQUIRE(autoMaster->getHeight() == import->getHeight());
  }
}
