#include "engine/BatchQueueRunner.h"

#include <algorithm>
#include <cctype>
#include <ctime>
#include <exception>
#include <iomanip>
#include <mutex>
#include <optional>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "analysis/StemAnalyzer.h"
#include "automix/HeuristicAutoMixStrategy.h"
#include "renderers/RendererPipeline.h"
#include "util/FileUtils.h"
#include "util/StringUtils.h"
#include "util/WavWriter.h"

namespace automix::engine {
namespace {

using ::automix::util::toLower;
using ::automix::util::extensionForFormat;
using ::automix::util::pathFromUtf8;
using ::automix::util::pathToGenericUtf8;
using ::automix::util::pathToUtf8;
using ::automix::util::trim;

bool hasAudioExtension(const std::filesystem::path& path) {
  const std::string ext = toLower(pathToUtf8(path.extension()));
  return ext == ".wav" || ext == ".aif" || ext == ".aiff" || ext == ".flac" || ext == ".mp3" || ext == ".ogg";
}

void runAnalysisForItem(domain::BatchItem& item) {
  item.status = domain::BatchItemStatus::Analyzing;
  analysis::StemAnalyzer analyzer;
  automix::HeuristicAutoMixStrategy autoMix;
  const auto analysisEntries = analyzer.analyzeSession(item.session);
  item.session.mixPlan = autoMix.buildPlan(item.session, analysisEntries, 1.0);
}

bool isTrimSeparator(const char value) {
  return std::isspace(static_cast<unsigned char>(value)) != 0 || value == '_' || value == '-' || value == '.';
}

std::string trimTokenSeparators(std::string value) {
  value = trim(std::move(value));
  while (!value.empty() && isTrimSeparator(value.front())) {
    value.erase(value.begin());
  }
  while (!value.empty() && isTrimSeparator(value.back())) {
    value.pop_back();
  }
  return value;
}

std::string sanitizeOutputStem(std::string value) {
  static constexpr size_t kMaxStemLength = 80;
  value = trimTokenSeparators(std::move(value));
  if (value.empty()) {
    return "song";
  }

  for (char& ch : value) {
    switch (ch) {
      case '<':
      case '>':
      case ':':
      case '"':
      case '/':
      case '\\':
      case '|':
      case '?':
      case '*':
        ch = '_';
        break;
      default:
        break;
    }
  }

  if (value.size() > kMaxStemLength) {
    size_t truncationIndex = kMaxStemLength;
    while (truncationIndex > 0 &&
           (static_cast<unsigned char>(value[truncationIndex]) & 0xC0) == 0x80) {
      --truncationIndex;
    }
    value.resize(truncationIndex);
    value = trimTokenSeparators(std::move(value));
  }

  if (value.empty()) {
    return "song";
  }

  return value;
}

std::string currentDateStamp() {
  const std::time_t now = std::time(nullptr);
  std::tm localTime{};
#if defined(_WIN32)
  localtime_s(&localTime, &now);
#else
  localtime_r(&now, &localTime);
#endif

  std::ostringstream output;
  output << std::put_time(&localTime, "%Y%m%d");
  return output.str();
}

std::string buildOutputFilename(const std::string& sessionName, const std::string& extension) {
  const auto safeTitle = sanitizeOutputStem(sessionName);
  return safeTitle + "_AutoMixMaster_" + currentDateStamp() + "_01" + extension;
}

std::string toPathKey(const std::filesystem::path& path) {
  return toLower(pathToGenericUtf8(path.lexically_normal()));
}

std::filesystem::path buildUniqueOutputPath(const std::filesystem::path& outputFolder,
                                            const std::string& sessionName,
                                            const std::string& extension,
                                            std::unordered_set<std::string>& reservedPaths) {
  const auto safeTitle = sanitizeOutputStem(sessionName);
  const auto dateStamp = currentDateStamp();

  for (int index = 1; index <= 9999; ++index) {
    std::ostringstream sequence;
    sequence << std::setw(2) << std::setfill('0') << index;
    const auto fileName = safeTitle + "_AutoMixMaster_" + dateStamp + "_" + sequence.str() + extension;
    const auto candidate = outputFolder / pathFromUtf8(fileName);
    const auto key = toPathKey(candidate);
    if (reservedPaths.contains(key)) {
      continue;
    }
    std::error_code existsError;
    if (!std::filesystem::exists(candidate, existsError) && !existsError) {
      reservedPaths.insert(key);
      return candidate;
    }
  }

  const auto fallback = outputFolder / pathFromUtf8(buildOutputFilename(sessionName, extension));
  reservedPaths.insert(toPathKey(fallback));
  return fallback;
}

struct ParsedStemName {
  std::string groupKey;
  std::string groupDisplay;
  std::string roleToken;
};

std::optional<ParsedStemName> parseParenthesizedStemName(const std::string& originalStemName) {
  const auto lowerStemName = toLower(originalStemName);
  if (lowerStemName.size() < 4 || lowerStemName.back() != ')') {
    return std::nullopt;
  }

  const auto openPos = lowerStemName.find_last_of('(');
  if (openPos == std::string::npos || openPos == 0 || openPos + 1 >= lowerStemName.size() - 1) {
    return std::nullopt;
  }

  const auto roleToken = trimTokenSeparators(lowerStemName.substr(openPos + 1, lowerStemName.size() - openPos - 2));
  auto groupDisplay = trimTokenSeparators(originalStemName.substr(0, openPos));
  if (roleToken.empty() || groupDisplay.empty()) {
    return std::nullopt;
  }

  return ParsedStemName{
      .groupKey = toLower(groupDisplay),
      .groupDisplay = groupDisplay,
      .roleToken = roleToken,
  };
}

std::optional<ParsedStemName> parseSuffixedStemName(const std::string& originalStemName) {
  static const std::vector<std::string> roleTokens = {
      "vocals", "vocal", "vox",
      "bass",
      "drums", "drum", "kick", "snare",
      "guitar", "gtr",
      "piano", "keys", "key", "synth",
      "fx", "effects", "sfx",
      "other", "music",
  };

  const auto lowerStemName = toLower(originalStemName);
  static constexpr char separators[] = {'_', '-', ' '};

  for (const auto& role : roleTokens) {
    for (const auto separator : separators) {
      const std::string suffix = std::string(1, separator) + role;
      if (lowerStemName.size() <= suffix.size() || !lowerStemName.ends_with(suffix)) {
        continue;
      }

      auto groupDisplay = trimTokenSeparators(originalStemName.substr(0, originalStemName.size() - suffix.size()));
      if (groupDisplay.empty()) {
        continue;
      }

      return ParsedStemName{
          .groupKey = toLower(groupDisplay),
          .groupDisplay = groupDisplay,
          .roleToken = role,
      };
    }
  }

  return std::nullopt;
}

ParsedStemName parseStemName(const std::filesystem::path& filePath) {
  const auto originalStemName = trimTokenSeparators(pathToUtf8(filePath.stem()));
  if (originalStemName.empty()) {
    return ParsedStemName{.groupKey = "song", .groupDisplay = "song", .roleToken = "mix"};
  }

  if (const auto parsed = parseParenthesizedStemName(originalStemName); parsed.has_value()) {
    return parsed.value();
  }

  if (const auto parsed = parseSuffixedStemName(originalStemName); parsed.has_value()) {
    return parsed.value();
  }

  return ParsedStemName{
      .groupKey = toLower(originalStemName),
      .groupDisplay = originalStemName,
      .roleToken = "mix",
  };
}

domain::StemRole roleFromSuffix(const std::string& suffix) {
  const auto normalized = toLower(trimTokenSeparators(suffix));

  if (normalized == "vocals" || normalized == "vocal" || normalized == "vox") {
    return domain::StemRole::Vocals;
  }
  if (normalized == "bass") {
    return domain::StemRole::Bass;
  }
  if (normalized == "drums" || normalized == "drum") {
    return domain::StemRole::Drums;
  }
  if (normalized == "kick") {
    return domain::StemRole::Kick;
  }
  if (normalized == "snare") {
    return domain::StemRole::Snare;
  }
  if (normalized == "guitar" || normalized == "gtr") {
    return domain::StemRole::Guitar;
  }
  if (normalized == "piano" || normalized == "keys" || normalized == "key" || normalized == "synth") {
    return domain::StemRole::Keys;
  }
  if (normalized == "fx" || normalized == "effects" || normalized == "sfx") {
    return domain::StemRole::Fx;
  }
  if (normalized == "other" || normalized == "music" || normalized == "mix") {
    return domain::StemRole::Music;
  }
  return domain::StemRole::Unknown;
}

} // namespace

std::vector<domain::BatchItem> BatchQueueRunner::buildItemsFromFolder(const std::filesystem::path& inputFolder,
                                                                       const std::filesystem::path& outputFolder,
                                                                       const bool recursiveScan) const {
  std::vector<domain::BatchItem> items;
  std::error_code error;
  if (!std::filesystem::exists(inputFolder, error) || error) {
    return items;
  }

  struct GroupedStemFile {
    std::filesystem::path filePath;
    std::string roleToken;
  };
  struct GroupedSessionFiles {
    std::string displayName;
    std::vector<GroupedStemFile> stems;
  };

  std::unordered_map<std::string, GroupedSessionFiles> groupedFiles;
  auto appendFile = [&](const std::filesystem::path& filePath) {
    if (!hasAudioExtension(filePath)) {
      return;
    }

    auto parsed = parseStemName(filePath);
    std::string groupKey = parsed.groupKey;
    std::string groupDisplay = parsed.groupDisplay;

    std::error_code relativeError;
    const auto relativeParent = std::filesystem::relative(filePath.parent_path(), inputFolder, relativeError);
    if (!relativeError && !relativeParent.empty() && relativeParent != std::filesystem::path(".")) {
      const auto relativeParentText = pathToGenericUtf8(relativeParent);
      groupKey = toLower(relativeParentText) + "/" + groupKey;
      groupDisplay = relativeParentText + "/" + groupDisplay;
    }

    auto& grouped = groupedFiles[groupKey];
    if (grouped.displayName.empty()) {
      grouped.displayName = groupDisplay;
    }
    grouped.stems.push_back({filePath, parsed.roleToken});
  };

  if (recursiveScan) {
    for (const auto& entry : std::filesystem::recursive_directory_iterator(inputFolder)) {
      if (!entry.is_regular_file()) {
        continue;
      }
      appendFile(entry.path());
    }
  } else {
    for (const auto& entry : std::filesystem::directory_iterator(inputFolder)) {
      if (!entry.is_regular_file()) {
        continue;
      }
      appendFile(entry.path());
    }
  }

  std::unordered_set<std::string> reservedPaths;
  items.reserve(groupedFiles.size());
  for (auto& [groupName, grouped] : groupedFiles) {
    std::sort(grouped.stems.begin(), grouped.stems.end(), [](const auto& left, const auto& right) {
      return pathToUtf8(left.filePath.filename()) < pathToUtf8(right.filePath.filename());
    });

    domain::Session session;
    session.sessionName = grouped.displayName.empty() ? groupName : grouped.displayName;
    session.stems.reserve(grouped.stems.size());

    int stemIndex = 1;
    for (const auto& groupedStem : grouped.stems) {
      domain::Stem stem;
      stem.id = "stem_" + std::to_string(stemIndex++);
      stem.name = pathToUtf8(groupedStem.filePath.stem());
      stem.filePath = pathToUtf8(groupedStem.filePath);
      stem.role = roleFromSuffix(groupedStem.roleToken);
      stem.origin = domain::StemOrigin::Separated;
      session.stems.push_back(stem);
    }

    domain::BatchItem item;
    item.session = session;
    item.sourcePath = inputFolder;
    item.outputPath = buildUniqueOutputPath(outputFolder, session.sessionName, ".wav", reservedPaths);
    item.status = domain::BatchItemStatus::Pending;
    items.push_back(item);
  }

  std::sort(items.begin(), items.end(), [](const domain::BatchItem& a, const domain::BatchItem& b) {
    return a.session.sessionName < b.session.sessionName;
  });
  return items;
}

domain::BatchResult BatchQueueRunner::process(domain::BatchJob& job,
                                              const ProgressCallback& progressCallback,
                                              std::atomic_bool* cancelFlag) const {
  domain::BatchResult result;
  if (job.items.empty()) {
    return result;
  }

  const int defaultThreads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  const bool useAcceleration = job.settings.renderSettings.preferHardwareAcceleration;
  const int analysisThreads =
      useAcceleration ? std::max(1, job.settings.analysisThreads > 0 ? job.settings.analysisThreads : defaultThreads) : 1;
  const int renderThreads = useAcceleration
                                ? std::max(1,
                                           job.settings.renderParallelism > 0 ? job.settings.renderParallelism
                                                                               : std::max(1, defaultThreads / 2))
                                : 1;

  std::vector<double> perItemProgress(job.items.size(), 0.0);
  std::mutex progressStateMutex;
  double accumulatedProgress = 0.0;
  const auto reportProgress = [&](const size_t itemIndex, const double itemFraction, const std::string& stage) {
    if (!progressCallback || itemIndex >= perItemProgress.size()) {
      return;
    }

    const double bounded = std::clamp(itemFraction, 0.0, 1.0);
    double overallProgress = 0.0;
    {
      std::scoped_lock lock(progressStateMutex);
      const double previous = perItemProgress[itemIndex];
      const double next = std::max(previous, bounded);
      perItemProgress[itemIndex] = next;
      accumulatedProgress += (next - previous);
      overallProgress = std::clamp(accumulatedProgress / static_cast<double>(perItemProgress.size()), 0.0, 1.0);
    }
    progressCallback(itemIndex, overallProgress, stage);
  };

  if (job.settings.parallelAnalysis && analysisThreads > 1) {
    std::atomic<size_t> nextIndex{0};
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(analysisThreads));

    for (int worker = 0; worker < analysisThreads; ++worker) {
      workers.emplace_back([&]() {
        for (;;) {
          const size_t i = nextIndex.fetch_add(1);
          if (i >= job.items.size()) {
            break;
          }

          auto& item = job.items[i];
          if (cancelFlag != nullptr && cancelFlag->load()) {
            item.status = domain::BatchItemStatus::Cancelled;
            item.error = "Cancelled";
            reportProgress(i, 1.0, "Cancelled");
            continue;
          }

          if (item.status == domain::BatchItemStatus::Pending) {
            try {
              runAnalysisForItem(item);
              reportProgress(i, 0.12, "Analyzing stems");
            } catch (const std::exception& errorException) {
              item.status = domain::BatchItemStatus::Failed;
              item.error = errorException.what();
              reportProgress(i, 1.0, "Analysis failed");
            } catch (...) {
              item.status = domain::BatchItemStatus::Failed;
              item.error = "Unknown analysis failure.";
              reportProgress(i, 1.0, "Analysis failed");
            }
          }
        }
      });
    }

    for (auto& worker : workers) {
      worker.join();
    }
  } else {
    for (auto& item : job.items) {
      const auto i = static_cast<size_t>(&item - job.items.data());
      if (cancelFlag != nullptr && cancelFlag->load()) {
        item.status = domain::BatchItemStatus::Cancelled;
        item.error = "Cancelled";
        reportProgress(i, 1.0, "Cancelled");
        continue;
      }
      if (item.status == domain::BatchItemStatus::Pending) {
        try {
          runAnalysisForItem(item);
          reportProgress(i, 0.12, "Analyzing stems");
        } catch (const std::exception& errorException) {
          item.status = domain::BatchItemStatus::Failed;
          item.error = errorException.what();
          reportProgress(i, 1.0, "Analysis failed");
        } catch (...) {
          item.status = domain::BatchItemStatus::Failed;
          item.error = "Unknown analysis failure.";
          reportProgress(i, 1.0, "Analysis failed");
        }
      }
    }
  }

  std::unordered_set<std::string> reservedPaths;
  for (const auto& item : job.items) {
    if (!item.outputPath.empty()) {
      reservedPaths.insert(toPathKey(item.outputPath));
    }
  }

  for (auto& item : job.items) {
    auto settings = job.settings.renderSettings;
    const std::string resolvedFormat = util::WavWriter::resolveFormat(item.outputPath, settings.outputFormat);
    const std::string requiredExtension = extensionForFormat(resolvedFormat);

    if (!item.outputPath.empty()) {
      reservedPaths.erase(toPathKey(item.outputPath));
    }
    item.outputPath = buildUniqueOutputPath(job.settings.outputFolder, item.session.sessionName, requiredExtension, reservedPaths);
  }

  std::atomic<size_t> renderIndex{0};

  const auto renderWorker = [&]() {
    for (;;) {
      const size_t i = renderIndex.fetch_add(1);
      if (i >= job.items.size()) {
        break;
      }

      auto& item = job.items[i];
      if (cancelFlag != nullptr && cancelFlag->load()) {
        item.status = domain::BatchItemStatus::Cancelled;
        item.error = "Cancelled";
        reportProgress(i, 1.0, "Cancelled");
        continue;
      }

      if (item.status == domain::BatchItemStatus::Cancelled) {
        reportProgress(i, 1.0, "Cancelled");
        continue;
      }
      if (item.status == domain::BatchItemStatus::Failed) {
        reportProgress(i, 1.0, "Skipped failed item");
        continue;
      }

      item.status = domain::BatchItemStatus::Rendering;
      item.error.clear();

      auto settings = job.settings.renderSettings;
      if (settings.rendererName.empty()) {
        settings.rendererName = "BuiltIn";
      }
      if (settings.processingThreads <= 0) {
        settings.processingThreads = std::max(1, defaultThreads / std::max(1, renderThreads));
      }

      const std::string resolvedFormat = util::WavWriter::resolveFormat(item.outputPath, settings.outputFormat);
      settings.outputFormat = resolvedFormat;
      settings.outputPath = pathToUtf8(item.outputPath);

      try {
        const auto renderResult = renderers::renderWithPipeline(
            item.session,
            settings,
            [&](const double stageProgress, const std::string& stage) {
              reportProgress(i, stageProgress, stage);
            },
            cancelFlag);

        if (renderResult.cancelled) {
          item.status = domain::BatchItemStatus::Cancelled;
          item.error = "Cancelled";
          reportProgress(i, 1.0, "Cancelled");
        } else if (renderResult.success) {
          item.status = domain::BatchItemStatus::Completed;
          item.reportPath = renderResult.reportPath;
          reportProgress(i, 1.0, "Render complete");
        } else {
          item.status = domain::BatchItemStatus::Failed;
          item.error = renderResult.logs.empty() ? "Render failed" : renderResult.logs.back();
          reportProgress(i, 1.0, "Render failed");
        }
      } catch (const std::exception& error) {
        item.status = domain::BatchItemStatus::Failed;
        item.error = error.what();
        reportProgress(i, 1.0, "Render failed");
      }
    }
  };

  if (renderThreads > 1) {
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(renderThreads));
    for (int worker = 0; worker < renderThreads; ++worker) {
      workers.emplace_back(renderWorker);
    }
    for (auto& worker : workers) {
      worker.join();
    }
  } else {
    renderWorker();
  }

  for (const auto& item : job.items) {
    switch (item.status) {
      case domain::BatchItemStatus::Completed:
        ++result.completed;
        break;
      case domain::BatchItemStatus::Failed:
        ++result.failed;
        break;
      case domain::BatchItemStatus::Cancelled:
        ++result.cancelled;
        break;
      default:
        break;
    }
  }
  return result;
}

} // namespace automix::engine
