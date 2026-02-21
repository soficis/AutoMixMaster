#include <juce_gui_extra/juce_gui_extra.h>

#include "app/style/AutoMixLookAndFeel.h"
#include "app/ui/MainLayout.h"

namespace automix::app {

class MainWindow final : public juce::DocumentWindow {
public:
  explicit MainWindow(juce::String name)
      : juce::DocumentWindow(name,
                             juce::Desktop::getInstance().getDefaultLookAndFeel().findColour(
                                 juce::ResizableWindow::backgroundColourId),
                             juce::DocumentWindow::allButtons) {
    setUsingNativeTitleBar(true);
    setContentOwned(new MainLayout(), true);
    setResizable(true, true);

    // Size to 85% of the primary display's usable area, with reasonable minimum.
    const auto& displays = juce::Desktop::getInstance().getDisplays();
    const auto* primary = displays.getPrimaryDisplay();
    const auto userArea = primary ? primary->userArea : juce::Rectangle<int>(0, 0, 1920, 1080);
    const int initW = juce::jmax(1280, userArea.getWidth()  * 85 / 100);
    const int initH = juce::jmax(800,  userArea.getHeight() * 85 / 100);
    centreWithSize(initW, initH);
    setResizeLimits(960, 640, 8192, 4320);
    setVisible(true);
  }

  void closeButtonPressed() override {
    juce::JUCEApplication::getInstance()->systemRequestedQuit();
  }
};

class AutoMixMasterApplication final : public juce::JUCEApplication {
public:
  const juce::String getApplicationName() override {
    return "AutoMixMaster";
  }
  const juce::String getApplicationVersion() override {
    return APP_VERSION;
  }

  void initialise(const juce::String&) override {
    lookAndFeel_ = std::make_unique<AutoMixLookAndFeel>();
    juce::LookAndFeel::setDefaultLookAndFeel(lookAndFeel_.get());
    mainWindow_ = std::make_unique<MainWindow>(getApplicationName());
  }

  void shutdown() override {
    mainWindow_.reset();
    juce::LookAndFeel::setDefaultLookAndFeel(nullptr);
    lookAndFeel_.reset();
  }

private:
  std::unique_ptr<AutoMixLookAndFeel> lookAndFeel_;
  std::unique_ptr<MainWindow> mainWindow_;
};

} // namespace automix::app

START_JUCE_APPLICATION(automix::app::AutoMixMasterApplication)
