#include "util/StringUtils.h"

namespace automix::util {

juce::String toJuceText(const std::vector<std::string>& lines) {
  juce::String output;
  for (const auto& line : lines) {
    output += juce::String(line);
    output += "\n";
  }
  return output;
}

} // namespace automix::util
