#include "util/WavWriter.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <juce_audio_formats/juce_audio_formats.h>

namespace automix::util {
namespace {

std::string toLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

int qualityIndexFromRequested(const juce::StringArray& options, const int requestedQuality) {
  if (options.isEmpty()) {
    return 0;
  }
  return std::clamp(requestedQuality, 0, options.size() - 1);
}

std::optional<std::filesystem::path> findLameExecutable() {
  if (const char* env = std::getenv("LAME_BIN"); env != nullptr && *env != '\0') {
    const std::filesystem::path candidate(env);
    std::error_code error;
    if (std::filesystem::is_regular_file(candidate, error) && !error) {
      return candidate;
    }
  }

  const char* rawPath = std::getenv("PATH");
  if (rawPath == nullptr || *rawPath == '\0') {
    return std::nullopt;
  }

#if defined(_WIN32)
  constexpr char delimiter = ';';
  const std::vector<std::string> names = {"lame.exe", "lame"};
#else
  constexpr char delimiter = ':';
  const std::vector<std::string> names = {"lame"};
#endif

  std::stringstream stream(rawPath);
  std::string token;
  while (std::getline(stream, token, delimiter)) {
    if (token.empty()) {
      continue;
    }
    for (const auto& name : names) {
      const auto candidate = std::filesystem::path(token) / name;
      std::error_code error;
      if (std::filesystem::is_regular_file(candidate, error) && !error) {
        return candidate;
      }
    }
  }

  return std::nullopt;
}

std::string extensionFormat(const std::filesystem::path& path) {
  const auto ext = toLower(path.extension().string());
  if (ext == ".wav" || ext == ".wave") {
    return "wav";
  }
  if (ext == ".aif" || ext == ".aiff") {
    return "aiff";
  }
  if (ext == ".flac") {
    return "flac";
  }
  if (ext == ".ogg" || ext == ".vorbis") {
    return "ogg";
  }
  if (ext == ".mp3") {
    return "mp3";
  }
  return "";
}

} // namespace

std::string WavWriter::resolveFormat(const std::filesystem::path& path, const std::string& preferredFormat) {
  const auto preferred = toLower(preferredFormat);
  if (!preferred.empty() && preferred != "auto") {
    if (preferred == "aif") {
      return "aiff";
    }
    if (preferred == "vorbis") {
      return "ogg";
    }
    return preferred;
  }

  const auto fromExtension = extensionFormat(path);
  return fromExtension.empty() ? "wav" : fromExtension;
}

bool WavWriter::isLossyFormat(const std::string& format) {
  const auto normalized = toLower(format);
  return normalized == "mp3" || normalized == "ogg";
}

void WavWriter::write(const std::filesystem::path& path,
                      const engine::AudioBuffer& buffer,
                      const int bitDepth,
                      const std::string& preferredFormat,
                      const int lossyBitrateKbps,
                      const int lossyQuality) const {
  juce::File outputFile(path.string());
  outputFile.deleteFile();

  std::unique_ptr<juce::FileOutputStream> stream(outputFile.createOutputStream());
  if (stream == nullptr || !stream->openedOk()) {
    throw std::runtime_error("Failed to open output audio file: " + path.string());
  }

  const auto format = resolveFormat(path, preferredFormat);
  const int outputBitDepth = std::clamp(bitDepth, 16, 32);
  const int bitrate = std::clamp(lossyBitrateKbps, 48, 512);
  const int quality = std::clamp(lossyQuality, 0, 10);

  juce::StringPairArray metadata;
  metadata.set("bitrate", juce::String(bitrate));

  std::unique_ptr<juce::AudioFormatWriter> writer;
  if (format == "wav") {
    juce::WavAudioFormat wav;
    writer.reset(wav.createWriterFor(stream.get(),
                                     buffer.getSampleRate(),
                                     static_cast<unsigned int>(buffer.getNumChannels()),
                                     outputBitDepth,
                                     metadata,
                                     0));
  } else if (format == "aiff") {
    juce::AiffAudioFormat aiff;
    writer.reset(aiff.createWriterFor(stream.get(),
                                      buffer.getSampleRate(),
                                      static_cast<unsigned int>(buffer.getNumChannels()),
                                      outputBitDepth,
                                      metadata,
                                      0));
  } else if (format == "flac") {
    juce::FlacAudioFormat flac;
    writer.reset(flac.createWriterFor(stream.get(),
                                      buffer.getSampleRate(),
                                      static_cast<unsigned int>(buffer.getNumChannels()),
                                      std::clamp(outputBitDepth, 16, 24),
                                      metadata,
                                      0));
  } else if (format == "ogg") {
#if defined(JUCE_USE_OGGVORBIS) && JUCE_USE_OGGVORBIS
    juce::OggVorbisAudioFormat ogg;
    writer.reset(ogg.createWriterFor(stream.get(),
                                     buffer.getSampleRate(),
                                     static_cast<unsigned int>(buffer.getNumChannels()),
                                     0,
                                     metadata,
                                     qualityIndexFromRequested(ogg.getQualityOptions(), quality)));
#endif
  } else if (format == "mp3") {
#if defined(JUCE_USE_MP3AUDIOFORMAT) && JUCE_USE_MP3AUDIOFORMAT
    juce::MP3AudioFormat mp3;
    writer.reset(mp3.createWriterFor(stream.get(),
                                     buffer.getSampleRate(),
                                     static_cast<unsigned int>(buffer.getNumChannels()),
                                     0,
                                     metadata,
                                     qualityIndexFromRequested(mp3.getQualityOptions(), quality)));

    if (writer == nullptr) {
#if defined(JUCE_USE_LAME_AUDIO_FORMAT) && JUCE_USE_LAME_AUDIO_FORMAT
      const auto lamePath = findLameExecutable();
      if (lamePath.has_value()) {
        juce::LAMEEncoderAudioFormat lame(mp3, juce::File(lamePath->string()));
        writer.reset(lame.createWriterFor(stream.get(),
                                          buffer.getSampleRate(),
                                          static_cast<unsigned int>(buffer.getNumChannels()),
                                          outputBitDepth,
                                          metadata,
                                          qualityIndexFromRequested(lame.getQualityOptions(), quality)));
      }
#endif
    }
#endif
  } else {
    throw std::runtime_error("Unsupported output format '" + format + "' for: " + path.string());
  }

  if (writer == nullptr) {
    throw std::runtime_error("Failed to create audio writer (format=" + format + ") for: " + path.string());
  }
  stream.release();

  juce::AudioBuffer<float> juceBuffer(buffer.getNumChannels(), buffer.getNumSamples());
  for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
    float* dst = juceBuffer.getWritePointer(ch);
    const float* src = buffer.getReadPointer(ch);
    std::copy(src, src + buffer.getNumSamples(), dst);
  }

  writer->writeFromAudioSampleBuffer(juceBuffer, 0, juceBuffer.getNumSamples());
}

} // namespace automix::util
