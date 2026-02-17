#include "engine/BatchQueueRunner.h"

#include <algorithm>
#include <cctype>
#include <exception>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "analysis/StemAnalyzer.h"
#include "automix/HeuristicAutoMixStrategy.h"
#include "renderers/RendererFactory.h"
#include "util/StringUtils.h"
#include "util/WavWriter.h"

namespace automix::engine {
namespace {

using ::automix::util::toLower;
using ::automix::util::extensionForFormat;

bool hasAudioExtension(const std::filesystem::path& path) {
  const std::string ext = toLower(path.extension().string());
  return ext == ".wav" || ext == ".aif" || ext == ".aiff" || ext == ".flac" || ext == ".mp3" || ext == ".ogg";
}

void runAnalysisForItem(domain::BatchItem& item) {
  item.status = domain::BatchItemStatus::Analyzing;
  analysis::StemAnalyzer analyzer;
  automix::HeuristicAutoMixStrategy autoMix;
  const auto analysisEntries = analyzer.analyzeSession(item.session);
  item.session.mixPlan = autoMix.buildPlan(item.session, analysisEntries, 1.0);
}

std::pair<std::string, std::string> splitGroupAndStem(const std::filesystem::path& filePath) {
  const std::string stemName = toLower(filePath.stem().string());
  static const std::vector<std::string> suffixes = {
      "_vocals", "-vocals", "_vocal", "-vocal", "_vox", "-vox",
      "_bass", "-bass", "_drums", "-drums", "_drum", "-drum",
      "_other", "-other", "_music", "-music"};

  for (const auto& suffix : suffixes) {
    if (stemName.size() <= suffix.size()) {
      continue;
    }
    if (stemName.ends_with(suffix)) {
      const std::string group = stemName.substr(0, stemName.size() - suffix.size());
      const std::string role = suffix.substr(1);
      return {group.empty() ? "song" : group, role};
    }
  }

  return {stemName, "mix"};
}

domain::StemRole roleFromSuffix(const std::string& suffix) {
  if (suffix == "vocals" || suffix == "vocal" || suffix == "vox") {
    return domain::StemRole::Vocals;
  }
  if (suffix == "bass") {
    return domain::StemRole::Bass;
  }
  if (suffix == "drums" || suffix == "drum") {
    return domain::StemRole::Drums;
  }
  if (suffix == "other" || suffix == "music") {
    return domain::StemRole::Music;
  }
  return domain::StemRole::Unknown;
}

} // namespace

std::vector<domain::BatchItem> BatchQueueRunner::buildItemsFromFolder(const std::filesystem::path& inputFolder,
                                                                       const std::filesystem::path& outputFolder) const {
  std::vector<domain::BatchItem> items;
  std::error_code error;
  if (!std::filesystem::exists(inputFolder, error) || error) {
    return items;
  }

  std::unordered_map<std::string, std::vector<std::filesystem::path>> groupedFiles;
  for (const auto& entry : std::filesystem::directory_iterator(inputFolder)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    if (!hasAudioExtension(entry.path())) {
      continue;
    }

    const auto [group, stem] = splitGroupAndStem(entry.path());
    (void)stem;
    groupedFiles[group].push_back(entry.path());
  }

  items.reserve(groupedFiles.size());
  for (const auto& [groupName, files] : groupedFiles) {
    domain::Session session;
    session.sessionName = groupName;
    session.stems.reserve(files.size());

    int stemIndex = 1;
    for (const auto& file : files) {
      const auto split = splitGroupAndStem(file);
      const auto& suffix = split.second;
      domain::Stem stem;
      stem.id = "stem_" + std::to_string(stemIndex++);
      stem.name = file.stem().string();
      stem.filePath = file.string();
      stem.role = roleFromSuffix(suffix);
      stem.origin = domain::StemOrigin::Separated;
      session.stems.push_back(stem);
    }

    domain::BatchItem item;
    item.session = session;
    item.sourcePath = inputFolder;
    item.outputPath = outputFolder / (groupName + "_master.wav");
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
            continue;
          }

          if (item.status == domain::BatchItemStatus::Pending) {
            try {
              runAnalysisForItem(item);
            } catch (const std::exception& errorException) {
              item.status = domain::BatchItemStatus::Failed;
              item.error = errorException.what();
            } catch (...) {
              item.status = domain::BatchItemStatus::Failed;
              item.error = "Unknown analysis failure.";
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
      if (cancelFlag != nullptr && cancelFlag->load()) {
        item.status = domain::BatchItemStatus::Cancelled;
        continue;
      }
      if (item.status == domain::BatchItemStatus::Pending) {
        try {
          runAnalysisForItem(item);
        } catch (const std::exception& errorException) {
          item.status = domain::BatchItemStatus::Failed;
          item.error = errorException.what();
        } catch (...) {
          item.status = domain::BatchItemStatus::Failed;
          item.error = "Unknown analysis failure.";
        }
      }
    }
  }

  std::atomic<int> completed{0};
  std::atomic<int> failed{0};
  std::atomic<int> cancelled{0};
  std::atomic<size_t> renderIndex{0};
  std::mutex itemMutex;

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
        ++cancelled;
        continue;
      }

      item.status = domain::BatchItemStatus::Rendering;

      auto settings = job.settings.renderSettings;
      if (settings.rendererName.empty()) {
        settings.rendererName = "BuiltIn";
      }
      if (settings.processingThreads <= 0) {
        settings.processingThreads = std::max(1, defaultThreads / std::max(1, renderThreads));
      }

      const std::string resolvedFormat = util::WavWriter::resolveFormat(item.outputPath, settings.outputFormat);
      const std::string requiredExtension = extensionForFormat(resolvedFormat);
      if (item.outputPath.empty()) {
        item.outputPath = job.settings.outputFolder / (item.session.sessionName + "_master" + requiredExtension);
      } else if (toLower(item.outputPath.extension().string()) != requiredExtension) {
        item.outputPath.replace_extension(requiredExtension);
      }

      settings.outputFormat = resolvedFormat;
      settings.outputPath = item.outputPath.string();

      try {
        auto renderer = renderers::createRenderer(settings.rendererName);
        const auto renderResult = renderer->render(
            item.session,
            settings,
            [&](const double stageProgress, const std::string& stage) {
              if (!progressCallback) {
                return;
              }
              const double itemWeight = 1.0 / static_cast<double>(job.items.size());
              const double progress = std::clamp(itemWeight * (static_cast<double>(i) + stageProgress), 0.0, 1.0);
              progressCallback(i, progress, stage);
            },
            cancelFlag);

        if (renderResult.cancelled) {
          item.status = domain::BatchItemStatus::Cancelled;
          item.error = "Cancelled";
          ++cancelled;
        } else if (renderResult.success) {
          item.status = domain::BatchItemStatus::Completed;
          item.reportPath = renderResult.reportPath;
          ++completed;
        } else {
          item.status = domain::BatchItemStatus::Failed;
          item.error = renderResult.logs.empty() ? "Render failed" : renderResult.logs.back();
          ++failed;
        }
      } catch (const std::exception& error) {
        std::scoped_lock lock(itemMutex);
        item.status = domain::BatchItemStatus::Failed;
        item.error = error.what();
        ++failed;
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

  result.completed = completed.load();
  result.failed = failed.load();
  result.cancelled = cancelled.load();
  return result;
}

} // namespace automix::engine
