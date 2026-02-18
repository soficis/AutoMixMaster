#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "domain/RenderSettings.h"
#include "domain/Session.h"

namespace automix::domain {

enum class BatchItemStatus {
  Pending,
  Analyzing,
  Rendering,
  Completed,
  Failed,
  Cancelled,
};

struct BatchItem {
  Session session;
  std::filesystem::path sourcePath;
  std::filesystem::path outputPath;
  BatchItemStatus status = BatchItemStatus::Pending;
  std::string error;
  std::string reportPath;
};

struct BatchSettings {
  std::filesystem::path outputFolder;
  int analysisThreads = 1;
  int renderParallelism = 1;
  bool parallelAnalysis = true;
  RenderSettings renderSettings;
};

struct BatchJob {
  std::vector<BatchItem> items;
  BatchSettings settings;
};

struct BatchResult {
  int completed = 0;
  int failed = 0;
  int cancelled = 0;
};

std::string toString(BatchItemStatus status);

} // namespace automix::domain
