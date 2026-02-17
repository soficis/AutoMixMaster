#include "app/controllers/OriginalMixController.h"

namespace automix::app {

OriginalMixSelectionResult OriginalMixController::applySelectedPath(const std::string& path,
                                                                    const std::string& displayName) const {
  OriginalMixSelectionResult result;
  if (path.empty()) {
    return result;
  }

  result.applied = true;
  result.path = path;
  result.statusText = "Original mix loaded";
  result.reportLine = "Original mix: " + juce::String(path);
  result.taskHistoryLine = "Original mix loaded: " + juce::String(displayName);
  return result;
}

OriginalMixClearResult OriginalMixController::clear(const std::optional<std::string>& currentPath) const {
  OriginalMixClearResult result;
  if (!currentPath.has_value()) {
    result.statusText = "No original mix is configured";
    return result;
  }

  result.cleared = true;
  result.statusText = "Original mix cleared";
  result.reportLine = "Original mix cleared";
  result.taskHistoryLine = "Original mix configuration cleared";
  return result;
}

} // namespace automix::app
