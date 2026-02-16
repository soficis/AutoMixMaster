#pragma once

#include <map>
#include <string>
#include <vector>

namespace automix::util {

std::map<std::string, std::string> applyMetadataPolicy(
    const std::map<std::string, std::string>& sourceMetadata,
    const std::string& policy,
    const std::map<std::string, std::string>& metadataTemplate = {},
    std::vector<std::string>* notes = nullptr);

} // namespace automix::util
