#include "util/MetadataPolicy.h"

#include <algorithm>
#include <cctype>
#include <set>

namespace automix::util {
namespace {

std::string toLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

std::string normalizeKey(const std::string& key) {
  std::string normalized;
  normalized.reserve(key.size());
  for (const auto c : key) {
    if (std::isalnum(static_cast<unsigned char>(c)) != 0) {
      normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
  }
  return normalized;
}

bool isCommonMetadataKey(const std::string& key) {
  static const std::set<std::string> keys = {
      "title", "track", "song", "tit2", "artist", "performer", "albumartist", "tpe1",
      "album", "talb", "year", "date", "tyer", "tdrc", "tracknumber", "trck",
      "genre", "tcon", "comment", "description", "comm",
  };
  return keys.find(key) != keys.end();
}

} // namespace

std::map<std::string, std::string> applyMetadataPolicy(
    const std::map<std::string, std::string>& sourceMetadata,
    const std::string& policy,
    const std::map<std::string, std::string>& metadataTemplate,
    std::vector<std::string>* notes) {
  const auto normalizedPolicy = toLower(policy.empty() ? "copy_all" : policy);

  if (normalizedPolicy == "strip") {
    if (notes != nullptr) {
      notes->push_back("Metadata policy=strip: removing all metadata tags.");
    }
    return {};
  }

  if (normalizedPolicy == "copy_common" || normalizedPolicy == "copy_common_only") {
    std::map<std::string, std::string> filtered;
    for (const auto& [key, value] : sourceMetadata) {
      if (value.empty()) {
        continue;
      }
      if (isCommonMetadataKey(normalizeKey(key))) {
        filtered[key] = value;
      }
    }
    if (notes != nullptr) {
      notes->push_back("Metadata policy=copy_common: preserving common distribution tags only.");
    }
    return filtered;
  }

  if (normalizedPolicy == "override_template") {
    std::map<std::string, std::string> merged;
    for (const auto& [key, value] : sourceMetadata) {
      if (!value.empty() && isCommonMetadataKey(normalizeKey(key))) {
        merged[key] = value;
      }
    }

    for (const auto& [key, value] : metadataTemplate) {
      if (!key.empty() && !value.empty()) {
        merged[key] = value;
      }
    }

    if (notes != nullptr) {
      notes->push_back("Metadata policy=override_template: merged common source tags with template overrides.");
    }
    return merged;
  }

  if (notes != nullptr) {
    notes->push_back("Metadata policy=copy_all: preserving all source metadata tags.");
  }
  return sourceMetadata;
}

} // namespace automix::util
