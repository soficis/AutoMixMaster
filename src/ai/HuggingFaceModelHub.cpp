#include "ai/HuggingFaceModelHub.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <set>
#include <sstream>
#include <unordered_set>

#include <juce_core/juce_core.h>
#include <nlohmann/json.hpp>

#include "util/StringUtils.h"

namespace automix::ai {
namespace {

using ::automix::util::toLower;
using ::automix::util::trim;

std::optional<std::string> readEnvironment(const char* key) {
#if defined(_WIN32)
  char* buffer = nullptr;
  size_t length = 0;
  if (_dupenv_s(&buffer, &length, key) != 0 || buffer == nullptr) {
    return std::nullopt;
  }

  std::string value(buffer, length > 0 ? length - 1 : 0);
  free(buffer);
  value = trim(value);
  if (value.empty()) {
    return std::nullopt;
  }
  return value;
#else
  const char* value = std::getenv(key);
  if (value == nullptr || *value == '\0') {
    return std::nullopt;
  }

  const auto trimmed = trim(value);
  if (trimmed.empty()) {
    return std::nullopt;
  }
  return trimmed;
#endif
}

std::string iso8601NowUtc() {
  return juce::Time::getCurrentTime().toISO8601(true).toStdString();
}

std::string sanitizeToken(std::string token) {
  token = trim(std::move(token));
  if (!token.empty() && token.rfind("Bearer ", 0) == 0) {
    token = token.substr(7);
  }
  return token;
}

std::string buildHeaders(const std::string& token) {
  std::string headers = "Accept: application/json\n";
  if (!token.empty()) {
    headers += "Authorization: Bearer " + token + "\n";
  }
  return headers;
}

std::optional<nlohmann::json> fetchJson(const std::string& url,
                                        const std::string& token,
                                        std::string* detail = nullptr) {
  int statusCode = 0;
  const auto options = juce::URL::InputStreamOptions(juce::URL::ParameterHandling::inAddress)
                           .withConnectionTimeoutMs(45000)
                           .withNumRedirectsToFollow(8)
                           .withStatusCode(&statusCode)
                           .withExtraHeaders(buildHeaders(token));
  const auto input = juce::URL(url).createInputStream(options);
  if (input == nullptr) {
    if (detail != nullptr) {
      *detail = "Failed to fetch JSON (no stream): " + url;
    }
    return std::nullopt;
  }

  if (statusCode >= 400) {
    if (detail != nullptr) {
      *detail = "Failed to fetch JSON (HTTP " + std::to_string(statusCode) + "): " + url;
    }
    return std::nullopt;
  }

  const auto text = input->readEntireStreamAsString().toStdString();
  if (text.empty()) {
    if (detail != nullptr) {
      *detail = "Received empty JSON body.";
    }
    return std::nullopt;
  }

  try {
    return nlohmann::json::parse(text);
  } catch (const std::exception& error) {
    if (detail != nullptr) {
      *detail = "Failed parsing JSON: " + std::string(error.what());
    }
    return std::nullopt;
  }
}

bool downloadToFile(const std::string& url,
                    const std::filesystem::path& outputPath,
                    const std::string& token,
                    std::string* detail = nullptr) {
  int statusCode = 0;
  const auto options = juce::URL::InputStreamOptions(juce::URL::ParameterHandling::inAddress)
                           .withConnectionTimeoutMs(60000)
                           .withNumRedirectsToFollow(8)
                           .withStatusCode(&statusCode)
                           .withExtraHeaders(buildHeaders(token));
  const auto input = juce::URL(url).createInputStream(options);
  if (input == nullptr) {
    if (detail != nullptr) {
      *detail = "Download failed (no stream): " + url;
    }
    return false;
  }

  if (statusCode >= 400) {
    if (detail != nullptr) {
      *detail = "Download failed (HTTP " + std::to_string(statusCode) + "): " + url;
    }
    return false;
  }

  std::error_code error;
  std::filesystem::create_directories(outputPath.parent_path(), error);
  if (error) {
    if (detail != nullptr) {
      *detail = "Failed creating output directory: " + outputPath.parent_path().string();
    }
    return false;
  }

  juce::File outFile(outputPath.string());
  std::unique_ptr<juce::FileOutputStream> out(outFile.createOutputStream());
  if (out == nullptr || !out->openedOk()) {
    if (detail != nullptr) {
      *detail = "Failed opening output file: " + outputPath.string();
    }
    return false;
  }

  out->writeFromInputStream(*input, -1);
  out->flush();

  if (!std::filesystem::is_regular_file(outputPath, error) || error) {
    if (detail != nullptr) {
      *detail = "Downloaded output missing: " + outputPath.string();
    }
    return false;
  }

  return true;
}

namespace sha256_impl {

static constexpr uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

void processBlock(uint32_t state[8], const uint8_t block[64]) {
  uint32_t w[64];
  for (int i = 0; i < 16; ++i) {
    w[i] = (static_cast<uint32_t>(block[i * 4]) << 24) |
           (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
           (static_cast<uint32_t>(block[i * 4 + 2]) << 8) |
           static_cast<uint32_t>(block[i * 4 + 3]);
  }
  for (int i = 16; i < 64; ++i) {
    const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
    const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }

  uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
  uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
  for (int i = 0; i < 64; ++i) {
    const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
    const uint32_t ch = (e & f) ^ (~e & g);
    const uint32_t temp1 = h + S1 + ch + K[i] + w[i];
    const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
    const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t temp2 = S0 + maj;
    h = g; g = f; f = e; e = d + temp1;
    d = c; c = b; b = a; a = temp1 + temp2;
  }

  state[0] += a; state[1] += b; state[2] += c; state[3] += d;
  state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

} // namespace sha256_impl

std::string computeFileSha256(const std::filesystem::path& filePath) {
  std::ifstream file(filePath, std::ios::binary);
  if (!file.is_open()) {
    return "";
  }

  uint32_t state[8] = {
      0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  };

  uint8_t block[64];
  uint64_t totalBytes = 0;
  while (file) {
    file.read(reinterpret_cast<char*>(block), 64);
    const auto bytesRead = static_cast<size_t>(file.gcount());
    totalBytes += bytesRead;
    if (bytesRead == 64) {
      sha256_impl::processBlock(state, block);
    } else {
      // Padding
      block[bytesRead] = 0x80;
      std::fill(block + bytesRead + 1, block + 64, static_cast<uint8_t>(0));
      if (bytesRead >= 56) {
        sha256_impl::processBlock(state, block);
        std::fill(block, block + 64, static_cast<uint8_t>(0));
      }
      const uint64_t totalBits = totalBytes * 8;
      for (int i = 0; i < 8; ++i) {
        block[63 - i] = static_cast<uint8_t>(totalBits >> (i * 8));
      }
      sha256_impl::processBlock(state, block);
    }
  }

  static const char digits[] = "0123456789abcdef";
  std::string hex;
  hex.reserve(64);
  for (int i = 0; i < 8; ++i) {
    for (int j = 3; j >= 0; --j) {
      const uint8_t byte = static_cast<uint8_t>(state[i] >> (j * 8));
      hex.push_back(digits[(byte >> 4) & 0x0f]);
      hex.push_back(digits[byte & 0x0f]);
    }
  }
  return hex;
}

std::string sanitizeRepoId(const std::string& repoId) {
  std::string out;
  out.reserve(repoId.size() + 8);
  for (const auto c : repoId) {
    if (std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '-' || c == '_') {
      out.push_back(c);
    } else if (c == '/') {
      out.append("__");
    } else {
      out.push_back('_');
    }
  }
  return out.empty() ? "model" : out;
}

std::string escapePathPreservingSlash(const std::string& path) {
  std::stringstream stream(path);
  std::string token;
  std::string encoded;
  bool first = true;
  while (std::getline(stream, token, '/')) {
    if (!first) {
      encoded.push_back('/');
    }
    first = false;
    encoded += juce::URL::addEscapeChars(token, false).toStdString();
  }
  return encoded;
}

std::string sourceUrlForRepo(const std::string& repoId) {
  return "https://huggingface.co/" + repoId;
}

std::string inferUseCase(const std::string& repoId,
                         const std::vector<std::string>& tags,
                         const std::string& fallbackQuery) {
  const auto repoLower = toLower(repoId);
  auto joined = repoLower;
  for (const auto& tag : tags) {
    joined += "|" + toLower(tag);
  }
  joined += "|" + toLower(fallbackQuery);

  if (joined.find("demucs") != std::string::npos ||
      joined.find("mdx") != std::string::npos ||
      joined.find("roformer") != std::string::npos ||
      joined.find("unmix") != std::string::npos ||
      joined.find("source-separation") != std::string::npos ||
      joined.find("separator") != std::string::npos) {
    return "stem-separation";
  }
  if (joined.find("clap") != std::string::npos) {
    return "style-retrieval-embedding";
  }
  if (joined.find("panns") != std::string::npos) {
    return "audio-tagging-embedding";
  }
  if (joined.find("basic-pitch") != std::string::npos || joined.find("midi") != std::string::npos) {
    return "pitch-midi-analysis";
  }
  if (joined.find("master") != std::string::npos || joined.find("loudness") != std::string::npos) {
    return "mastering-assistant";
  }
  return "general-audio-model";
}

std::string inferLicense(const nlohmann::json& json) {
  if (json.contains("cardData") && json.at("cardData").is_object()) {
    const auto cardLicense = json.at("cardData").value("license", "");
    if (!cardLicense.empty()) {
      return cardLicense;
    }
  }

  if (json.contains("tags") && json.at("tags").is_array()) {
    for (const auto& tagJson : json.at("tags")) {
      if (!tagJson.is_string()) {
        continue;
      }
      const auto tag = tagJson.get<std::string>();
      if (tag.rfind("license:", 0) == 0 && tag.size() > 8) {
        return tag.substr(8);
      }
    }
  }

  return "unknown";
}

std::string pickPrimaryFile(const std::vector<std::string>& files,
                            bool* hasOnnxOut = nullptr) {
  if (hasOnnxOut != nullptr) {
    *hasOnnxOut = false;
  }

  if (files.empty()) {
    return "";
  }

  const std::vector<std::string> preferred = {
      "model.onnx",
      "model_fp16.onnx",
      "model_int8.onnx",
      "pytorch_model.bin",
      "model.safetensors",
      "model.pt",
      "model.ckpt",
      "checkpoint.pt",
  };

  auto lowerFiles = files;
  std::vector<std::string> lower;
  lower.reserve(files.size());
  for (const auto& file : files) {
    lower.push_back(toLower(file));
  }

  for (const auto& preferredName : preferred) {
    const auto needle = toLower(preferredName);
    const auto it = std::find(lower.begin(), lower.end(), needle);
    if (it != lower.end()) {
      const auto index = static_cast<size_t>(std::distance(lower.begin(), it));
      if (hasOnnxOut != nullptr && needle.size() >= 5 && needle.ends_with(".onnx")) {
        *hasOnnxOut = true;
      }
      return files[index];
    }
  }

  for (size_t i = 0; i < files.size(); ++i) {
    const auto item = toLower(files[i]);
    if (item.ends_with(".onnx")) {
      if (hasOnnxOut != nullptr) {
        *hasOnnxOut = true;
      }
      return files[i];
    }
  }

  for (size_t i = 0; i < files.size(); ++i) {
    const auto item = toLower(files[i]);
    if (item.ends_with(".safetensors") || item.ends_with(".bin") || item.ends_with(".pt") || item.ends_with(".ckpt")) {
      return files[i];
    }
  }

  return "";
}

bool trustedOrganization(const std::string& repoId) {
  const auto slash = repoId.find('/');
  const auto org = toLower(slash == std::string::npos ? repoId : repoId.substr(0, slash));
  static const std::unordered_set<std::string> trusted = {
      "laion",
      "speechbrain",
      "spotify",
      "faroit",
      "intel",
      "espnet",
      "openai",
      "mozilla",
      "facebook",
      "meta",
      "microsoft",
  };
  return trusted.find(org) != trusted.end();
}

nlohmann::json loadJsonIfPresent(const std::filesystem::path& path) {
  try {
    std::ifstream in(path);
    if (!in.is_open()) {
      return nlohmann::json::array();
    }
    nlohmann::json parsed;
    in >> parsed;
    return parsed;
  } catch (...) {
    return nlohmann::json::array();
  }
}

void writeJson(const std::filesystem::path& path, const nlohmann::json& json) {
  std::error_code error;
  std::filesystem::create_directories(path.parent_path(), error);
  std::ofstream out(path);
  out << json.dump(2);
}

void updateInstallRegistry(const std::filesystem::path& root,
                           const HubModelInfo& model,
                           const HubInstallResult& result) {
  const auto registryPath = root / "install_registry.json";
  auto registry = loadJsonIfPresent(registryPath);
  if (!registry.is_array()) {
    registry = nlohmann::json::array();
  }

  bool updated = false;
  for (auto& item : registry) {
    if (!item.is_object() || item.value("repoId", "") != model.repoId) {
      continue;
    }
    item = {
        {"repoId", model.repoId},
        {"revision", result.revision},
        {"installedAtUtc", iso8601NowUtc()},
        {"installPath", result.installPath.string()},
        {"primaryFile", result.primaryFilePath.filename().string()},
        {"useCase", model.useCase},
        {"license", model.license},
        {"sourceUrl", model.sourceUrl},
        {"downloads", model.downloads},
        {"likes", model.likes},
    };
    updated = true;
    break;
  }

  if (!updated) {
    registry.push_back({
        {"repoId", model.repoId},
        {"revision", result.revision},
        {"installedAtUtc", iso8601NowUtc()},
        {"installPath", result.installPath.string()},
        {"primaryFile", result.primaryFilePath.filename().string()},
        {"useCase", model.useCase},
        {"license", model.license},
        {"sourceUrl", model.sourceUrl},
        {"downloads", model.downloads},
        {"likes", model.likes},
    });
  }

  std::sort(registry.begin(), registry.end(), [](const nlohmann::json& a, const nlohmann::json& b) {
    return a.value("repoId", "") < b.value("repoId", "");
  });
  writeJson(registryPath, registry);
}

void appendInstallLog(const std::filesystem::path& root,
                      const HubModelInfo& model,
                      const HubInstallResult& result) {
  const auto logPath = root / "install_log.jsonl";
  std::error_code error;
  std::filesystem::create_directories(logPath.parent_path(), error);

  nlohmann::json event = {
      {"timestampUtc", iso8601NowUtc()},
      {"repoId", model.repoId},
      {"revision", result.revision},
      {"success", result.success},
      {"message", result.message},
      {"installPath", result.installPath.string()},
      {"primaryFile", result.primaryFilePath.filename().string()},
      {"useCase", model.useCase},
      {"license", model.license},
  };

  std::ofstream out(logPath, std::ios::app);
  out << event.dump() << "\n";
}

} // namespace

std::vector<std::string> HuggingFaceModelHub::defaultRecommendedSearchTerms() {
  return {
      "demucs",
      "mdx23c",
      "bs-roformer",
      "mel-band-roformer",
      "open-unmix",
      "clap",
      "basic-pitch",
      "panns",
  };
}

std::string HuggingFaceModelHub::resolveToken(const std::string& explicitToken) const {
  auto token = sanitizeToken(explicitToken);
  if (!token.empty()) {
    return token;
  }

  for (const auto* key : {"AUTOMIX_HF_TOKEN", "HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN"}) {
    if (const auto value = readEnvironment(key); value.has_value()) {
      token = sanitizeToken(*value);
      if (!token.empty()) {
        return token;
      }
    }
  }

  return "";
}

std::optional<HubModelInfo> HuggingFaceModelHub::modelInfo(const std::string& repoId, const std::string& token) const {
  if (repoId.empty()) {
    return std::nullopt;
  }

  const auto effectiveToken = resolveToken(token);
  const auto url = "https://huggingface.co/api/models/" + juce::URL::addEscapeChars(repoId, false).toStdString();
  std::string detail;
  const auto payload = fetchJson(url, effectiveToken, &detail);
  if (!payload.has_value() || !payload->is_object()) {
    return std::nullopt;
  }

  HubModelInfo info;
  info.repoId = payload->value("id", repoId);
  info.displayName = info.repoId;
  info.revision = payload->value("sha", "main");
  info.downloads = payload->value("downloads", 0);
  info.likes = payload->value("likes", 0);
  info.privateRepo = payload->value("private", false);
  info.gated = payload->value("gated", false);
  info.disabled = payload->value("disabled", false);
  info.lastModified = payload->value("lastModified", "");
  info.license = inferLicense(*payload);
  info.sourceUrl = sourceUrlForRepo(info.repoId);

  if (payload->contains("tags") && payload->at("tags").is_array()) {
    for (const auto& tag : payload->at("tags")) {
      if (tag.is_string()) {
        info.tags.push_back(tag.get<std::string>());
      }
    }
  }

  if (payload->contains("siblings") && payload->at("siblings").is_array()) {
    for (const auto& sibling : payload->at("siblings")) {
      if (!sibling.is_object()) {
        continue;
      }
      const auto file = sibling.value("rfilename", "");
      if (!file.empty()) {
        info.files.push_back(file);
        if (sibling.contains("lfs") && sibling.at("lfs").is_object()) {
          const auto sha = sibling.at("lfs").value("sha256", "");
          if (!sha.empty()) {
            info.fileSha256[file] = sha;
          }
        }
      }
    }
  }

  info.primaryFile = pickPrimaryFile(info.files, &info.hasOnnx);
  info.useCase = inferUseCase(info.repoId, info.tags, "");

  return info;
}

std::vector<HubModelInfo> HuggingFaceModelHub::discoverRecommended(const HubModelQueryOptions& options) const {
  const auto effectiveToken = resolveToken(options.token);
  std::vector<HubModelInfo> discovered;
  std::set<std::string> seen;

  for (const auto& query : defaultRecommendedSearchTerms()) {
    const auto escaped = juce::URL::addEscapeChars(query, false).toStdString();
    const auto url = "https://huggingface.co/api/models?search=" + escaped +
                     "&sort=downloads&direction=-1&limit=" + std::to_string(std::max<size_t>(1, options.maxResultsPerQuery));

    const auto response = fetchJson(url, effectiveToken);
    if (!response.has_value() || !response->is_array()) {
      continue;
    }

    for (const auto& item : *response) {
      if (!item.is_object()) {
        continue;
      }

      const auto repoId = item.value("id", "");
      if (repoId.empty() || !seen.insert(repoId).second) {
        continue;
      }

      auto info = modelInfo(repoId, effectiveToken);
      if (!info.has_value()) {
        continue;
      }

      if (info->privateRepo || info->disabled) {
        continue;
      }
      if (!options.includeGated && info->gated) {
        continue;
      }
      if (info->primaryFile.empty()) {
        continue;
      }

      info->useCase = inferUseCase(info->repoId, info->tags, query);
      const bool trust = trustedOrganization(info->repoId);
      const bool hasOpenLicense = info->license != "unknown" && info->license != "other";
      info->recommended = trust || (hasOpenLicense && info->downloads >= 50);
      discovered.push_back(std::move(info.value()));
    }
  }

  std::sort(discovered.begin(), discovered.end(), [](const HubModelInfo& a, const HubModelInfo& b) {
    if (a.recommended != b.recommended) {
      return a.recommended > b.recommended;
    }
    if (a.downloads != b.downloads) {
      return a.downloads > b.downloads;
    }
    if (a.likes != b.likes) {
      return a.likes > b.likes;
    }
    return a.repoId < b.repoId;
  });

  if (discovered.size() > 40) {
    discovered.resize(40);
  }
  return discovered;
}

HubInstallResult HuggingFaceModelHub::installModel(const std::string& repoId, const HubInstallOptions& options) const {
  HubInstallResult result;
  result.repoId = repoId;

  const auto effectiveToken = resolveToken(options.token);
  const auto info = modelInfo(repoId, effectiveToken);
  if (!info.has_value()) {
    result.message = "Unable to fetch model info from Hugging Face.";
    return result;
  }

  if (info->privateRepo || info->disabled) {
    result.message = "Model is private or disabled.";
    return result;
  }
  if (info->gated && effectiveToken.empty()) {
    result.message = "Model is gated. Set HF token in AUTOMIX_HF_TOKEN/HF_TOKEN/HUGGINGFACE_TOKEN.";
    return result;
  }
  if (info->primaryFile.empty()) {
    result.message = "Model does not expose a downloadable primary model file.";
    return result;
  }

  const auto destinationRoot = options.destinationRoot.empty() ? std::filesystem::path("assets/modelhub") : options.destinationRoot;
  const auto installPath = destinationRoot / sanitizeRepoId(info->repoId);
  const auto primaryPath = installPath / std::filesystem::path(info->primaryFile).filename();
  result.installPath = installPath;
  result.primaryFilePath = primaryPath;
  result.revision = info->revision.empty() ? "main" : info->revision;

  std::error_code error;
  if (std::filesystem::is_regular_file(primaryPath, error) && !error && !options.overwrite) {
    result.success = true;
    result.message = "Model already installed.";
    result.downloadedFiles = {primaryPath.filename().string()};
    updateInstallRegistry(destinationRoot, info.value(), result);
    appendInstallLog(destinationRoot, info.value(), result);
    return result;
  }

  std::filesystem::create_directories(installPath, error);
  if (error) {
    result.message = "Failed to create install directory: " + installPath.string();
    return result;
  }

  const auto revision = result.revision;
  const auto filePathInRepo = escapePathPreservingSlash(info->primaryFile);
  const auto downloadUrl = "https://huggingface.co/" + info->repoId + "/resolve/" + revision + "/" + filePathInRepo;
  std::string detail;
  if (!downloadToFile(downloadUrl, primaryPath, effectiveToken, &detail)) {
    result.message = detail.empty() ? "Model download failed." : detail;
    appendInstallLog(destinationRoot, info.value(), result);
    return result;
  }

  const auto expectedShaIt = info->fileSha256.find(info->primaryFile);
  if (expectedShaIt != info->fileSha256.end() && !expectedShaIt->second.empty()) {
    const auto computedSha = computeFileSha256(primaryPath);
    if (computedSha != expectedShaIt->second) {
      std::filesystem::remove(primaryPath, error);
      result.message = "SHA-256 verification failed for " + primaryPath.filename().string() +
                       ". Expected: " + expectedShaIt->second +
                       ", computed: " + (computedSha.empty() ? "(unable to compute)" : computedSha) +
                       ". Corrupted download removed.";
      appendInstallLog(destinationRoot, info.value(), result);
      return result;
    }
  }

  result.downloadedFiles.push_back(primaryPath.filename().string());

  if (options.downloadReadme) {
    const auto readmeIt = std::find_if(info->files.begin(), info->files.end(), [](const std::string& file) {
      return toLower(file) == "readme.md";
    });
    if (readmeIt != info->files.end()) {
      const auto readmeUrl = "https://huggingface.co/" + info->repoId + "/resolve/" + revision + "/README.md";
      const auto readmePath = installPath / "README.md";
      if (downloadToFile(readmeUrl, readmePath, effectiveToken, nullptr)) {
        result.downloadedFiles.push_back("README.md");
      }
    }
  }

  nlohmann::json metadata = {
      {"schemaVersion", 1},
      {"repoId", info->repoId},
      {"revision", revision},
      {"name", info->displayName},
      {"useCase", info->useCase},
      {"license", info->license},
      {"sourceUrl", info->sourceUrl},
      {"downloads", info->downloads},
      {"likes", info->likes},
      {"installedAtUtc", iso8601NowUtc()},
      {"primaryFile", primaryPath.filename().string()},
      {"primaryFileSha256", computeFileSha256(primaryPath)},
      {"hasOnnx", info->hasOnnx},
      {"tokenUsed", !effectiveToken.empty()},
      {"availableFiles", info->files},
      {"tags", info->tags},
  };
  result.metadataPath = installPath / "modelhub.json";
  writeJson(result.metadataPath, metadata);

  result.success = true;
  result.message = "Model installed successfully.";
  updateInstallRegistry(destinationRoot, info.value(), result);
  appendInstallLog(destinationRoot, info.value(), result);
  return result;
}

} // namespace automix::ai
