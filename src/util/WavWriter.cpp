#include "util/WavWriter.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_core/juce_core.h>

#include "util/LameDownloader.h"

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

std::vector<std::string> lameExecutableNames() {
#if defined(_WIN32)
  return {"lame.exe", "lame"};
#else
  return {"lame"};
#endif
}

bool isRegularFile(const std::filesystem::path& path) {
  std::error_code error;
  return std::filesystem::is_regular_file(path, error) && !error;
}

std::string trimPathToken(std::string token) {
  const auto first = std::find_if_not(token.begin(), token.end(), [](const unsigned char c) {
    return std::isspace(c) != 0;
  });
  const auto last = std::find_if_not(token.rbegin(), token.rend(), [](const unsigned char c) {
    return std::isspace(c) != 0;
  }).base();

  if (first >= last) {
    return "";
  }

  token = std::string(first, last);
  if (token.size() >= 2 && token.front() == '"' && token.back() == '"') {
    token = token.substr(1, token.size() - 2);
  }
  return token;
}

void appendAncestorPaths(const std::filesystem::path& seed,
                         std::vector<std::filesystem::path>& roots,
                         std::unordered_set<std::string>& seen) {
  if (seed.empty()) {
    return;
  }

  std::error_code error;
  auto current = std::filesystem::absolute(seed, error);
  if (error) {
    return;
  }

  for (int depth = 0; depth < 8; ++depth) {
    const auto key = toLower(current.lexically_normal().string());
    if (seen.insert(key).second) {
      roots.push_back(current);
    }

    if (!current.has_parent_path()) {
      break;
    }
    const auto parent = current.parent_path();
    if (parent == current) {
      break;
    }
    current = parent;
  }
}

std::vector<std::filesystem::path> ancestorPaths() {
  std::vector<std::filesystem::path> roots;
  std::unordered_set<std::string> seen;
  roots.reserve(24);

  std::error_code error;
  appendAncestorPaths(std::filesystem::current_path(error), roots, seen);

  const juce::File executable = juce::File::getSpecialLocation(juce::File::currentExecutableFile);
  if (executable.existsAsFile()) {
    const auto executableDir = std::filesystem::path(executable.getParentDirectory().getFullPathName().toStdString());
    appendAncestorPaths(executableDir, roots, seen);
    appendAncestorPaths(executableDir / ".." / "Resources", roots, seen);
  }

  return roots;
}

std::optional<std::filesystem::path> findLameInDirectory(const std::filesystem::path& directory) {
  if (directory.empty()) {
    return std::nullopt;
  }

  for (const auto& name : lameExecutableNames()) {
    const auto candidate = directory / name;
    if (isRegularFile(candidate)) {
      return candidate;
    }
  }
  return std::nullopt;
}

std::optional<std::filesystem::path> resolveBundledLameExecutable() {
#if defined(ENABLE_BUNDLED_LAME) && ENABLE_BUNDLED_LAME
  const auto names = lameExecutableNames();
#if defined(_WIN32)
  const std::vector<std::filesystem::path> dirs = {
      "assets/lame",
      "assets/lame/bin",
      "assets/lame/windows",
      "assets/codecs/lame",
      "third_party/lame/bin",
  };
#elif defined(__APPLE__)
  const std::vector<std::filesystem::path> dirs = {
      "assets/lame",
      "assets/lame/bin",
      "assets/lame/mac",
      "assets/codecs/lame",
      "third_party/lame/bin",
  };
#else
  const std::vector<std::filesystem::path> dirs = {
      "assets/lame",
      "assets/lame/bin",
      "assets/lame/linux",
      "assets/codecs/lame",
      "third_party/lame/bin",
  };
#endif

  for (const auto& root : ancestorPaths()) {
    for (const auto& dir : dirs) {
      for (const auto& name : names) {
        const auto candidate = root / dir / name;
        if (isRegularFile(candidate)) {
          return candidate;
        }
      }
    }
  }
#endif

  return std::nullopt;
}

std::optional<std::filesystem::path> findLameExecutable() {
  if (const auto downloaded = LameDownloader::cacheBinaryPath(); isRegularFile(downloaded)) {
    return downloaded;
  }

  if (const auto bundled = resolveBundledLameExecutable(); bundled.has_value()) {
    return bundled;
  }

  const juce::File executable = juce::File::getSpecialLocation(juce::File::currentExecutableFile);
  if (executable.existsAsFile()) {
    const auto executableDir = std::filesystem::path(executable.getParentDirectory().getFullPathName().toStdString());
    if (const auto besideExecutable = findLameInDirectory(executableDir); besideExecutable.has_value()) {
      return besideExecutable;
    }
    if (const auto inExecutableBin = findLameInDirectory(executableDir / "bin"); inExecutableBin.has_value()) {
      return inExecutableBin;
    }
    if (const auto inExecutableLame = findLameInDirectory(executableDir / "lame"); inExecutableLame.has_value()) {
      return inExecutableLame;
    }
  }

  if (const char* env = std::getenv("LAME_BIN"); env != nullptr && *env != '\0') {
    const std::filesystem::path candidate(trimPathToken(env));
    if (isRegularFile(candidate)) {
      return candidate;
    }
  }

  const char* rawPath = std::getenv("PATH");
  if (rawPath == nullptr || *rawPath == '\0') {
    return std::nullopt;
  }

#if defined(_WIN32)
  constexpr char delimiter = ';';
#else
  constexpr char delimiter = ':';
#endif
  const auto names = lameExecutableNames();

  std::stringstream stream(rawPath);
  std::string token;
  while (std::getline(stream, token, delimiter)) {
    const auto entry = trimPathToken(token);
    if (entry.empty()) {
      continue;
    }
    for (const auto& name : names) {
      const auto candidate = std::filesystem::path(entry) / name;
      if (isRegularFile(candidate)) {
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

bool formatExistsInManager(const juce::AudioFormatManager& manager, const std::string& extension) {
  for (int i = 0; i < manager.getNumKnownFormats(); ++i) {
    if (const auto* format = manager.getKnownFormat(i); format != nullptr) {
      const auto exts = format->getFileExtensions();
      for (const auto& ext : exts) {
        if (toLower(ext.toStdString()) == toLower(extension)) {
          return true;
        }
      }
    }
  }
  return false;
}

std::unique_ptr<juce::AudioFormatWriter> createWriterForFormat(const std::string& format,
                                                               juce::OutputStream* stream,
                                                               const double sampleRate,
                                                               const int channels,
                                                               const int outputBitDepth,
                                                               const int bitrate,
                                                               const int quality,
                                                               std::string* detail) {
  juce::StringPairArray metadata;
  metadata.set("bitrate", juce::String(bitrate));

  std::unique_ptr<juce::AudioFormatWriter> writer;
  const auto normalized = toLower(format);

  if (normalized == "wav") {
    juce::WavAudioFormat wav;
    writer.reset(wav.createWriterFor(stream,
                                     sampleRate,
                                     static_cast<unsigned int>(channels),
                                     outputBitDepth,
                                     metadata,
                                     0));
    if (writer == nullptr && detail != nullptr) {
      *detail = "WAV writer creation failed.";
    }
    return writer;
  }

  if (normalized == "aiff") {
    juce::AiffAudioFormat aiff;
    writer.reset(aiff.createWriterFor(stream,
                                      sampleRate,
                                      static_cast<unsigned int>(channels),
                                      outputBitDepth,
                                      metadata,
                                      0));
    if (writer == nullptr && detail != nullptr) {
      *detail = "AIFF writer creation failed.";
    }
    return writer;
  }

  if (normalized == "flac") {
    juce::FlacAudioFormat flac;
    writer.reset(flac.createWriterFor(stream,
                                      sampleRate,
                                      static_cast<unsigned int>(channels),
                                      std::clamp(outputBitDepth, 16, 24),
                                      metadata,
                                      0));
    if (writer == nullptr && detail != nullptr) {
      *detail = "FLAC writer creation failed.";
    }
    return writer;
  }

  if (normalized == "ogg") {
#if defined(JUCE_USE_OGGVORBIS) && JUCE_USE_OGGVORBIS
    juce::OggVorbisAudioFormat ogg;
    writer.reset(ogg.createWriterFor(stream,
                                     sampleRate,
                                     static_cast<unsigned int>(channels),
                                     0,
                                     metadata,
                                     qualityIndexFromRequested(ogg.getQualityOptions(), quality)));
    if (writer == nullptr && detail != nullptr) {
      *detail = "OGG writer creation failed.";
    }
#else
    if (detail != nullptr) {
      *detail = "JUCE OGG/Vorbis codec support is disabled in this build.";
    }
#endif
    return writer;
  }

  if (normalized == "mp3") {
#if defined(JUCE_USE_MP3AUDIOFORMAT) && JUCE_USE_MP3AUDIOFORMAT
    juce::MP3AudioFormat mp3;
    writer.reset(mp3.createWriterFor(stream,
                                     sampleRate,
                                     static_cast<unsigned int>(channels),
                                     0,
                                     metadata,
                                     qualityIndexFromRequested(mp3.getQualityOptions(), quality)));

    if (writer == nullptr) {
#if defined(JUCE_USE_LAME_AUDIO_FORMAT) && JUCE_USE_LAME_AUDIO_FORMAT
      const auto lamePath = findLameExecutable();
      if (lamePath.has_value()) {
        juce::LAMEEncoderAudioFormat lame(mp3, juce::File(lamePath->string()));
        writer.reset(lame.createWriterFor(stream,
                                          sampleRate,
                                          static_cast<unsigned int>(channels),
                                          outputBitDepth,
                                          metadata,
                                          qualityIndexFromRequested(lame.getQualityOptions(), quality)));
      }
#endif
    }

    if (writer == nullptr && detail != nullptr) {
      *detail = "MP3 writer unavailable (JUCE MP3 codec or LAME fallback not found).";
    }
#else
    if (detail != nullptr) {
      *detail = "JUCE MP3 codec support is disabled in this build.";
    }
#endif
    return writer;
  }

  if (detail != nullptr) {
    *detail = "Unsupported format '" + format + "'.";
  }
  return writer;
}

void writeAudioBufferToWriter(juce::AudioFormatWriter& writer, const engine::AudioBuffer& buffer) {
  juce::AudioBuffer<float> juceBuffer(buffer.getNumChannels(), buffer.getNumSamples());
  for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
    float* dst = juceBuffer.getWritePointer(ch);
    const float* src = buffer.getReadPointer(ch);
    std::copy(src, src + buffer.getNumSamples(), dst);
  }

  writer.writeFromAudioSampleBuffer(juceBuffer, 0, juceBuffer.getNumSamples());
}

int lameQualityPreset(const int quality) {
  const int clamped = std::clamp(quality, 0, 10);
  return std::clamp(9 - clamped, 0, 9);
}

bool encodeMp3WithExternalLame(const std::filesystem::path& lamePath,
                               const std::filesystem::path& inputWavPath,
                               const std::filesystem::path& outputMp3Path,
                               const int bitrateKbps,
                               const int quality,
                               std::string* detail) {
  juce::StringArray command;
  command.add(lamePath.string());
  command.add("--silent");
  command.add("-b");
  command.add(std::to_string(std::clamp(bitrateKbps, 48, 320)));
  command.add("-q");
  command.add(std::to_string(lameQualityPreset(quality)));
  command.add(inputWavPath.string());
  command.add(outputMp3Path.string());

  juce::ChildProcess process;
  if (!process.start(command)) {
    if (detail != nullptr) {
      *detail = "Failed to start external LAME process.";
    }
    return false;
  }

  if (!process.waitForProcessToFinish(120000)) {
    process.kill();
    if (detail != nullptr) {
      *detail = "External LAME process timed out.";
    }
    return false;
  }

  const int exitCode = process.getExitCode();
  const auto output = process.readAllProcessOutput().toStdString();
  std::error_code error;
  const bool hasOutput = std::filesystem::is_regular_file(outputMp3Path, error) && !error;
  if (exitCode != 0 || !hasOutput) {
    if (detail != nullptr) {
      std::ostringstream os;
      os << "External LAME failed (exit=" << exitCode << ")";
      if (!output.empty()) {
        os << ": " << output;
      }
      *detail = os.str();
    }
    return false;
  }

  if (detail != nullptr) {
    *detail = "External LAME encode succeeded.";
  }
  return true;
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

std::vector<WavWriter::FormatAvailability> WavWriter::getAvailableFormats() {
  juce::AudioFormatManager manager;
  manager.registerBasicFormats();

  const std::vector<std::pair<std::string, std::string>> formatDescriptors = {
      {"wav", ".wav"},
      {"aiff", ".aiff"},
      {"flac", ".flac"},
      {"ogg", ".ogg"},
      {"mp3", ".mp3"},
  };

  std::vector<FormatAvailability> availability;
  availability.reserve(formatDescriptors.size());

  for (const auto& [format, extension] : formatDescriptors) {
    auto stream = std::make_unique<juce::MemoryOutputStream>();
    std::string detail;
    auto writer = createWriterForFormat(format, stream.get(), 44100.0, 2, 24, 192, 7, &detail);
    const bool knownByManager = formatExistsInManager(manager, extension);

    bool available = writer != nullptr;
    if (!available && format == "mp3") {
      if (const auto lamePath = findLameExecutable(); lamePath.has_value()) {
        available = true;
        detail = "MP3 available via external LAME binary: " + lamePath->string();
      } else if (LameDownloader::isSupportedOnCurrentPlatform()) {
        available = true;
        detail = "MP3 available via on-demand LAME downloader (attempted during export).";
      }
    }

    if (available) {
      if (writer != nullptr) {
        stream.release();
        detail = knownByManager ? "Writer and AudioFormatManager support are available."
                                : "Writer available; not listed by AudioFormatManager::getKnownFormats().";
      } else if (detail.empty()) {
        detail = "Format available through external encoder.";
      }
    } else if (detail.empty()) {
      detail = knownByManager ? "Format known by AudioFormatManager but writer creation failed."
                              : "Format not available in this build.";
    }
    if (!available && format == "mp3") {
      const std::string guidance =
          "Set LAME_BIN or add 'lame' to PATH. Automatic downloader uses AUTOMIX_LAME_DOWNLOAD_URL/AUTOMIX_LAME_VERSION.";
      detail = detail.empty() ? guidance : (detail + " " + guidance);
    }

    availability.push_back(FormatAvailability{
        .format = format,
        .available = available,
        .detail = detail,
    });
  }

  return availability;
}

bool WavWriter::isFormatAvailable(const std::string& format) {
  const auto normalized = toLower(format);
  const auto availability = getAvailableFormats();
  const auto it = std::find_if(availability.begin(), availability.end(), [&](const FormatAvailability& entry) {
    return entry.format == normalized;
  });
  return it != availability.end() && it->available;
}

void WavWriter::write(const std::filesystem::path& path,
                      const engine::AudioBuffer& buffer,
                      const int bitDepth,
                      const std::string& preferredFormat,
                      const int lossyBitrateKbps,
                      const int lossyQuality) const {
  const auto format = resolveFormat(path, preferredFormat);
  const auto normalizedFormat = toLower(format);
  const int outputBitDepth = std::clamp(bitDepth, 16, 32);
  const int bitrate = std::clamp(lossyBitrateKbps, 48, 512);
  const int quality = std::clamp(lossyQuality, 0, 10);

  juce::File outputFile(path.string());
  outputFile.deleteFile();

  std::unique_ptr<juce::FileOutputStream> stream(outputFile.createOutputStream());
  if (stream == nullptr || !stream->openedOk()) {
    throw std::runtime_error("Failed to open output audio file: " + path.string());
  }

  std::string detail;
  auto writer = createWriterForFormat(format,
                                      stream.get(),
                                      buffer.getSampleRate(),
                                      buffer.getNumChannels(),
                                      outputBitDepth,
                                      bitrate,
                                      quality,
                                      &detail);

  if (writer == nullptr && normalizedFormat == "mp3") {
    stream.reset();
    outputFile.deleteFile();
    auto lamePath = findLameExecutable();
    if (!lamePath.has_value()) {
      const auto download = LameDownloader::ensureAvailable();
      if (download.success && !download.executablePath.empty()) {
        lamePath = download.executablePath;
      } else if (!download.detail.empty()) {
        detail = detail.empty() ? download.detail : (detail + " " + download.detail);
      }
    }
    if (lamePath.has_value()) {
      const auto nonce = std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count());
      const auto tempWavPath = path.parent_path() / (path.stem().string() + "_lame_input_" + nonce + ".wav");
      std::string lameDetail;
      std::error_code error;

      try {
        write(tempWavPath, buffer, outputBitDepth, "wav", bitrate, quality);
        if (encodeMp3WithExternalLame(*lamePath, tempWavPath, path, bitrate, quality, &lameDetail)) {
          std::filesystem::remove(tempWavPath, error);
          return;
        }
      } catch (const std::exception& errorException) {
        lameDetail = errorException.what();
      }

      std::filesystem::remove(tempWavPath, error);
      if (!lameDetail.empty()) {
        detail = detail.empty() ? lameDetail : detail + " " + lameDetail;
      }
    }
  }

  if (writer == nullptr) {
    const auto availability = getAvailableFormats();
    std::ostringstream os;
    os << "Failed to create audio writer (format=" << format << ") for: " << path.string();
    if (!detail.empty()) {
      os << " [" << detail << "]";
    }
    os << " Available formats:";
    for (const auto& entry : availability) {
      os << ' ' << entry.format << '=' << (entry.available ? "yes" : "no");
    }
    throw std::runtime_error(os.str());
  }
  stream.release();
  writeAudioBufferToWriter(*writer, buffer);
}

} // namespace automix::util
