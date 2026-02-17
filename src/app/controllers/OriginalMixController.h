#pragma once

#include <optional>
#include <string>

#include <juce_core/juce_core.h>

namespace automix::app {

struct OriginalMixSelectionResult {
  bool applied = false;
  std::string path;
  juce::String statusText;
  juce::String reportLine;
  juce::String taskHistoryLine;
};

struct OriginalMixClearResult {
  bool cleared = false;
  juce::String statusText;
  juce::String reportLine;
  juce::String taskHistoryLine;
};

class OriginalMixController {
 public:
  OriginalMixSelectionResult applySelectedPath(const std::string& path,
                                               const std::string& displayName) const;
  OriginalMixClearResult clear(const std::optional<std::string>& currentPath) const;
};

} // namespace automix::app
