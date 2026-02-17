#pragma once

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

#include <juce_core/juce_core.h>

namespace automix::util {

inline std::string toLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

inline std::string trim(std::string value) {
  const auto first = std::find_if_not(value.begin(), value.end(), [](const unsigned char c) {
    return std::isspace(c) != 0;
  });
  const auto last = std::find_if_not(value.rbegin(), value.rend(), [](const unsigned char c) {
    return std::isspace(c) != 0;
  }).base();

  if (first >= last) {
    return "";
  }

  value = std::string(first, last);
  if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
    value = value.substr(1, value.size() - 2);
  }
  return value;
}

inline std::string extensionForFormat(const std::string& format) {
  const auto normalized = toLower(format);
  if (normalized == "wav") {
    return ".wav";
  }
  if (normalized == "flac") {
    return ".flac";
  }
  if (normalized == "aiff" || normalized == "aif") {
    return ".aiff";
  }
  if (normalized == "ogg" || normalized == "vorbis") {
    return ".ogg";
  }
  if (normalized == "mp3") {
    return ".mp3";
  }
  return ".wav";
}

inline juce::String toJuceText(const std::vector<std::string>& lines) {
  juce::String output;
  for (const auto& line : lines) {
    output += juce::String(line);
    output += "\n";
  }
  return output;
}

} // namespace automix::util
