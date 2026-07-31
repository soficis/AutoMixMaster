#pragma once

#include <functional>

#include <juce_gui_basics/juce_gui_basics.h>

namespace automix::app::ui {

/// Global keyboard shortcut handler.
/// Install via juce::Component::addKeyListener() on the main window.
class KeyboardShortcuts final : public juce::KeyListener {
public:
  using Action = std::function<void()>;

  /// All action callbacks are optional — leave default-constructed to ignore.
  struct Actions {
    Action togglePlay;            // Ctrl+Space
    Action exportFile;            // Ctrl+E
    Action autoMix;               // Ctrl+M
    Action autoMaster;            // Ctrl+Shift+M
    Action mixAndMaster;          // Ctrl+Shift+A
    Action openSettings;          // Ctrl+,
    Action toggleBatchRecursive;  // Ctrl+B

    Action stop;                  // Escape
    Action toggleBatchMode;       // Ctrl+Shift+B
    Action openModelManager;      // Ctrl+K
    Action toggleTheme;           // Ctrl+T
    Action saveSession;           // Ctrl+S
    Action loadSession;           // Ctrl+O
  };

  explicit KeyboardShortcuts(Actions actions)
      : actions_(std::move(actions)) {}

  bool keyPressed(const juce::KeyPress& key, juce::Component* /*originatingComponent*/) override {
    return processKeyPress(key.getKeyCode(), key.getModifiers());
  }

  bool keyStateChanged(bool /*isKeyDown*/, juce::Component* /*originatingComponent*/) override {
    return false; // we only react on press, not release
  }

private:
  Actions actions_;

  bool processKeyPress(const int keyCode, const juce::KeyPress::ModifierKeys& modifiers) {
    const bool ctrl = modifiers.isCtrlDown();
    const bool shift = modifiers.isShiftDown();
    // const bool alt = modifiers.isAltDown();  // reserved for future use

    // ── Ctrl+ combinations ──────────────────────────────────────
    if (ctrl && !shift) {
      switch (keyCode) {
        case ' ': // Ctrl+Space = toggle play
          if (actions_.togglePlay) { actions_.togglePlay(); return true; }
          break;
        case 'E':
          if (actions_.exportFile) { actions_.exportFile(); return true; }
          break;
        case 'M':
          if (actions_.autoMix) { actions_.autoMix(); return true; }
          break;
        case ',':
          if (actions_.openSettings) { actions_.openSettings(); return true; }
          break;
        case 'B':
          if (actions_.toggleBatchRecursive) { actions_.toggleBatchRecursive(); return true; }
          break;
        case 'K':
          if (actions_.openModelManager) { actions_.openModelManager(); return true; }
          break;
        case 'T':
          if (actions_.toggleTheme) { actions_.toggleTheme(); return true; }
          break;
        case 'S':
          if (actions_.saveSession) { actions_.saveSession(); return true; }
          break;
        case 'O':
          if (actions_.loadSession) { actions_.loadSession(); return true; }
          break;
        default:
          break;
      }
    }

    // ── Ctrl+Shift+ combinations ────────────────────────────────
    if (ctrl && shift) {
      switch (keyCode) {
        case 'M':
          if (actions_.autoMaster) { actions_.autoMaster(); return true; }
          break;
        case 'A':
          if (actions_.mixAndMaster) { actions_.mixAndMaster(); return true; }
          break;
        case 'B':
          if (actions_.toggleBatchMode) { actions_.toggleBatchMode(); return true; }
          break;
        default:
          break;
      }
    }

    // ── Escape ──────────────────────────────────────────────────
    if (keyCode == juce::KeyPress::escapeKey && !ctrl && !shift) {
      if (actions_.stop) { actions_.stop(); return true; }
    }

    return false; // unhandled
  }

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(KeyboardShortcuts)
};

} // namespace automix::app::ui
