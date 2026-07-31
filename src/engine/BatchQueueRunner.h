#pragma once

#include <atomic>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <vector>

#include "domain/BatchTypes.h"

namespace automix::engine {

class BatchQueueRunner {
 public:
  using ProgressCallback = std::function<void(size_t itemIndex, double fraction, const std::string& stage)>;

  struct ProgressDetail {
    size_t itemIndex = 0;
    double overallFraction = 0.0;
    double itemFraction = 0.0;
    std::string stage;
    std::string itemName;
    domain::BatchItemStatus status = domain::BatchItemStatus::Pending;
    size_t completedCount = 0;
    size_t failedCount = 0;
    size_t totalCount = 0;
  };

  using DetailedProgressCallback = std::function<void(const ProgressDetail&)>;

  struct StreamConfig {
    size_t chunkSize = 8;
    bool enableStreaming = false;
  };

  std::vector<domain::BatchItem> buildItemsFromFolder(const std::filesystem::path& inputFolder,
                                                       const std::filesystem::path& outputFolder,
                                                       bool recursiveScan = false) const;

  domain::BatchResult process(domain::BatchJob& job,
                              const ProgressCallback& progressCallback,
                              std::atomic_bool* cancelFlag) const;

  domain::BatchResult processStreaming(domain::BatchJob& job,
                                       const DetailedProgressCallback& progressCallback,
                                       std::atomic_bool* cancelFlag,
                                       StreamConfig config = {}) const;
};

} // namespace automix::engine
