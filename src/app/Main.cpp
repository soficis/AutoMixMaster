#include <juce_gui_extra/juce_gui_extra.h>

#include "app/MainComponent.h"

namespace automix::app {

class MainWindow final : public juce::DocumentWindow {
 public:
  explicit MainWindow(juce::String name)
      : juce::DocumentWindow(name,
                             juce::Desktop::getInstance().getDefaultLookAndFeel()
                                 .findColour(juce::ResizableWindow::backgroundColourId),
                             juce::DocumentWindow::allButtons) {
    setUsingNativeTitleBar(true);
    setContentOwned(new MainComponent(), true);
    setResizable(true, true);
    centreWithSize(1280, 720);
    setVisible(true);
  }

  void closeButtonPressed() override { juce::JUCEApplication::getInstance()->systemRequestedQuit(); }
};

class AutoMixMasterApplication final : public juce::JUCEApplication {
 public:
  const juce::String getApplicationName() override { return "AutoMixMaster"; }
  const juce::String getApplicationVersion() override { return APP_VERSION; }

  void initialise(const juce::String&) override { mainWindow_ = std::make_unique<MainWindow>(getApplicationName()); }

  void shutdown() override { mainWindow_.reset(); }

 private:
  std::unique_ptr<MainWindow> mainWindow_;
};

} // namespace automix::app

START_JUCE_APPLICATION(automix::app::AutoMixMasterApplication)
