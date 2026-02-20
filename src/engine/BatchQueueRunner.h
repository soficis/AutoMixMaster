#pragma once

#include <atomic>
#include <filesystem>
#include <functional>
#include <vector>

#include "domain/BatchTypes.h"

namespace automix::engine {

class BatchQueueRunner {
 public:
  using ProgressCallback = std::function<void(size_t itemIndex, double fraction, const std::string& stage)>;

  std::vector<domain::BatchItem> buildItemsFromFolder(const std::filesystem::path& inputFolder,
                                                      const std::filesystem::path& outputFolder,
                                                      bool recursiveScan = false) const;

  domain::BatchResult process(domain::BatchJob& job,
                              const ProgressCallback& progressCallback,
                              std::atomic_bool* cancelFlag) const;
};

} // namespace automix::engine
