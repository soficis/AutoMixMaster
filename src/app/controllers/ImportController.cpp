#include "app/controllers/ImportController.h"

#include <filesystem>

#include "ai/StemSeparator.h"

namespace automix::app {

ImportController::ImportController(juce::ThreadPool& threadPool, Callbacks callbacks)
    : threadPool_(threadPool), callbacks_(std::move(callbacks)) {}

void ImportController::importFiles(std::vector<juce::File> files,
                                   const bool useSeparation,
                                   const int preferredStemCount) {
  if (files.empty()) {
    return;
  }

  if (callbacks_.onStatus) {
    callbacks_.onStatus("Importing files...");
  }
  if (callbacks_.onTaskHistory) {
    callbacks_.onTaskHistory("Import started");
  }

  struct ImportJob final : juce::ThreadPoolJob {
    std::vector<juce::File> files;
    bool useSeparation;
    int preferredStemCount;
    Callbacks callbacks;

    ImportJob(std::vector<juce::File> f, bool sep, int stemCount, Callbacks cb)
        : juce::ThreadPoolJob("ImportJob"),
          files(std::move(f)),
          useSeparation(sep),
          preferredStemCount(stemCount),
          callbacks(std::move(cb)) {}

    JobStatus runJob() override {
      std::vector<domain::Stem> importedStems;
      std::vector<std::string> importLines;

      if (files.size() == 1 && useSeparation) {
        try {
          const auto mixPath = std::filesystem::path(files.front().getFullPathName().toStdString());
          const auto outputDir = mixPath.parent_path() / (mixPath.stem().string() + "_separated");

          ai::StemSeparator separator;
          ai::StemSeparator::SeparationOptions separationOptions;
          separationOptions.targetStemCount = preferredStemCount;
          const auto separationResult = separator.separate(mixPath, outputDir, separationOptions);
          if (separationResult.success) {
            importedStems = separationResult.stems;
            importLines.push_back("Separated import from: " + mixPath.string());
            importLines.push_back("Variant stems: " + std::to_string(separationResult.stemVariantCount));
            for (const auto& stem : separationResult.stems) {
              std::string line = "  stem -> " + stem.filePath + " role=" + domain::toString(stem.role);
              if (stem.separationConfidence.has_value()) {
                line += " confidence=" + std::to_string(stem.separationConfidence.value());
              }
              if (stem.separationArtifactRisk.has_value()) {
                line += " artifactRisk=" + std::to_string(stem.separationArtifactRisk.value());
              }
              importLines.push_back(line);
            }
            if (!separationResult.qaReportPath.empty()) {
              importLines.push_back("Separation QA report: " + separationResult.qaReportPath.string());
            }
            importLines.push_back("QA energyLeakage=" + std::to_string(separationResult.qaMetrics.energyLeakage) +
                                  " residualDistortion=" + std::to_string(separationResult.qaMetrics.residualDistortion) +
                                  " transientRetention=" + std::to_string(separationResult.qaMetrics.transientRetention));
            importLines.push_back(separationResult.logMessage);
          } else {
            importLines.push_back("Separation failed, importing original mix file as stem.");
            importLines.push_back(separationResult.logMessage);
          }
        } catch (const std::exception& error) {
          importLines.push_back("Separation error: " + std::string(error.what()));
        } catch (...) {
          importLines.push_back("Separation error: unknown failure");
        }
      }

      if (importedStems.empty()) {
        for (size_t i = 0; i < files.size(); ++i) {
          const auto& file = files[i];

          domain::Stem stem;
          stem.id = "stem_" + std::to_string(i + 1);
          stem.name = file.getFileNameWithoutExtension().toStdString();
          stem.filePath = file.getFullPathName().toStdString();
          stem.origin = useSeparation ? domain::StemOrigin::Separated : domain::StemOrigin::Recorded;
          stem.enabled = true;
          importedStems.push_back(stem);

          importLines.push_back(stem.name + " -> " + stem.filePath);
        }
      }

      auto capturedCallbacks = callbacks;
      ImportResult result{std::move(importedStems), std::move(importLines)};
      juce::MessageManager::callAsync([capturedCallbacks, result = std::move(result)]() mutable {
        if (capturedCallbacks.onImportComplete) {
          capturedCallbacks.onImportComplete(std::move(result));
        }
      });

      return jobHasFinished;
    }
  };

  threadPool_.addJob(new ImportJob(std::move(files), useSeparation, preferredStemCount, callbacks_), true);
}

} // namespace automix::app
