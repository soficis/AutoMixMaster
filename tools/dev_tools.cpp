#include <algorithm>
#include <cctype>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "ai/IModelInference.h"
#include "ai/FeatureSchema.h"
#include "ai/ModelPackLoader.h"
#include "ai/OnnxModelInference.h"
#include "ai/RtNeuralInference.h"
#include "analysis/StemAnalyzer.h"
#include "domain/Stem.h"
#include "domain/StemOrigin.h"
#include "domain/StemRole.h"
#include "engine/AudioFileIO.h"
#include "engine/OfflineRenderPipeline.h"
#include "engine/SessionRepository.h"
#include "util/LameDownloader.h"
#include "util/WavWriter.h"
#include "renderers/ExternalLimiterRenderer.h"

namespace {

std::string sanitizeFileName(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char c) {
    if (std::isalnum(c) || c == '_' || c == '-') {
      return static_cast<char>(c);
    }
    return '_';
  });
  if (value.empty()) {
    return "segment";
  }
  return value;
}

std::optional<std::string> argValue(const std::vector<std::string>& args, const std::string& key) {
  for (size_t i = 0; i + 1 < args.size(); ++i) {
    if (args[i] == key) {
      return args[i + 1];
    }
  }
  return std::nullopt;
}

bool hasFlag(const std::vector<std::string>& args, const std::string& key) {
  return std::find(args.begin(), args.end(), key) != args.end();
}

automix::engine::AudioBuffer sliceBuffer(const automix::engine::AudioBuffer& input,
                                         const int maxSamples) {
  const int outputSamples = std::max(0, std::min(input.getNumSamples(), maxSamples));
  automix::engine::AudioBuffer output(input.getNumChannels(), outputSamples, input.getSampleRate());

  for (int ch = 0; ch < input.getNumChannels(); ++ch) {
    for (int i = 0; i < outputSamples; ++i) {
      output.setSample(ch, i, input.getSample(ch, i));
    }
  }

  return output;
}

std::vector<double> deterministicFeatures(const size_t count) {
  std::vector<double> features;
  features.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    features.push_back(static_cast<double>(i + 1) / static_cast<double>(count + 1));
  }
  return features;
}

std::string taskFromModelType(const std::string& type) {
  if (type == "role_classifier") {
    return "role_classifier";
  }
  if (type == "master_parameters") {
    return "master_parameters";
  }
  return "mix_parameters";
}

struct SupportedModelPack {
  std::string id;
  std::string type;
  std::string description;
};

struct SupportedLimiter {
  std::string id;
  std::string name;
  std::string description;
  std::string licenseId;
};

const std::vector<SupportedModelPack>& supportedModelPacks() {
  static const std::vector<SupportedModelPack> packs = {
      {"demo-role-v1", "role_classifier", "Deterministic demo role-classifier pack."},
      {"demo-mix-v1", "mix_parameters", "Deterministic demo mix-parameter pack."},
      {"demo-master-v1", "master_parameters", "Deterministic demo mastering pack."},
  };
  return packs;
}

const std::vector<SupportedLimiter>& supportedLimiters() {
  static const std::vector<SupportedLimiter> limiters = {
      {"phaselimiter", "PhaseLimiter", "Phase limiter external renderer package.", "See assets/phaselimiter/licenses"},
      {"external-template", "ExternalLimiterTemplate", "Template external limiter descriptor for custom tools.", "User-supplied"},
  };
  return limiters;
}

std::optional<std::filesystem::path> findRepoPath(const std::filesystem::path& relativePath) {
  std::error_code error;
  auto current = std::filesystem::absolute(std::filesystem::current_path(error), error);
  if (error) {
    return std::nullopt;
  }
  for (int depth = 0; depth < 6; ++depth) {
    const auto candidate = current / relativePath;
    if (std::filesystem::exists(candidate, error) && !error) {
      return candidate;
    }
    if (!current.has_parent_path()) {
      break;
    }
    current = current.parent_path();
  }
  return std::nullopt;
}

void copyDirectory(const std::filesystem::path& source, const std::filesystem::path& destination) {
  std::error_code error;
  std::filesystem::create_directories(destination, error);
  if (error) {
    throw std::runtime_error("Failed to create destination directory: " + destination.string());
  }
  std::filesystem::copy(source,
                        destination,
                        std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing,
                        error);
  if (error) {
    throw std::runtime_error("Failed to copy directory: " + source.string() + " -> " + destination.string());
  }
}

int commandListSupportedModels() {
  std::cout << "Supported model packs:\n";
  for (const auto& pack : supportedModelPacks()) {
    std::cout << "  - " << pack.id << " [" << pack.type << "] " << pack.description << "\n";
  }
  return 0;
}

int commandInstallSupportedModel(const std::vector<std::string>& args) {
  const auto idArg = argValue(args, "--id");
  if (!idArg.has_value()) {
    std::cerr << "install-supported-model requires --id <model_id>\n";
    return 2;
  }
  const std::filesystem::path destinationRoot = argValue(args, "--dest").value_or("assets/models");

  const auto it = std::find_if(supportedModelPacks().begin(), supportedModelPacks().end(),
                               [&](const SupportedModelPack& pack) { return pack.id == *idArg; });
  if (it == supportedModelPacks().end()) {
    std::cerr << "Unknown supported model id: " << *idArg << "\n";
    return 2;
  }

  const auto source = findRepoPath(std::filesystem::path("tools/catalog/modelpacks") / it->id);
  if (!source.has_value()) {
    std::cerr << "Catalog source not found for model: " << it->id << "\n";
    return 1;
  }

  const auto destination = destinationRoot / it->id;
  copyDirectory(source.value(), destination);
  std::cout << "Installed model pack '" << it->id << "' to " << destination.string() << "\n";
  return 0;
}

int commandListSupportedLimiters() {
  std::cout << "Supported limiters:\n";
  for (const auto& limiter : supportedLimiters()) {
    std::cout << "  - " << limiter.id << " [" << limiter.name << "] " << limiter.description << "\n";
  }
  return 0;
}

int commandInstallSupportedLimiter(const std::vector<std::string>& args) {
  const auto idArg = argValue(args, "--id");
  if (!idArg.has_value()) {
    std::cerr << "install-supported-limiter requires --id <limiter_id>\n";
    return 2;
  }
  const std::filesystem::path destinationRoot = argValue(args, "--dest").value_or("assets/limiters");
  std::filesystem::create_directories(destinationRoot);

  if (*idArg == "phaselimiter") {
    const auto source = findRepoPath("assets/phaselimiter");
    if (!source.has_value()) {
      std::cerr << "PhaseLimiter source package not found under assets/phaselimiter\n";
      return 1;
    }
    const auto destination = destinationRoot / "phaselimiter";
    std::filesystem::create_directories(destination);
    copyDirectory(source.value(), destination / "runtime");

    nlohmann::json descriptor = {
        {"id", "PhaseLimiterPack"},
        {"name", "PhaseLimiter (pack)"},
        {"version", "external"},
        {"licenseId", "See assets/phaselimiter/licenses"},
        {"binaryPath", "runtime/phase_limiter"},
        {"bundledByDefault", false},
    };
    std::ofstream out(destination / "renderer.json");
    out << descriptor.dump(2);
    std::cout << "Installed limiter '" << *idArg << "' to " << destination.string() << "\n";
    return 0;
  }

  if (*idArg == "external-template") {
    const auto destination = destinationRoot / "external-template";
    std::filesystem::create_directories(destination);
    nlohmann::json descriptor = {
        {"id", "ExternalTemplate"},
        {"name", "External Limiter Template"},
        {"version", "1.0"},
        {"licenseId", "User-supplied"},
        {"binaryPath", "your_limiter_binary_here"},
        {"bundledByDefault", false},
    };
    std::ofstream out(destination / "renderer.json");
    out << descriptor.dump(2);
    std::cout << "Installed limiter template to " << destination.string() << "\n";
    return 0;
  }

  std::cerr << "Unknown supported limiter id: " << *idArg << "\n";
  return 2;
}

int commandInstallLameFallback(const std::vector<std::string>& args) {
  const bool force = hasFlag(args, "--force");
  const bool jsonOutput = hasFlag(args, "--json");

  const auto result = automix::util::LameDownloader::ensureAvailable(force);
  const auto cachePath = automix::util::LameDownloader::cacheBinaryPath();

  if (jsonOutput) {
    nlohmann::json payload = {
        {"success", result.success},
        {"attempted", result.attempted},
        {"path", result.success ? result.executablePath.string() : cachePath.string()},
        {"detail", result.detail},
    };
    std::cout << payload.dump(2) << "\n";
    return result.success ? 0 : 1;
  }

  std::cout << "LAME fallback installation:\n";
  std::cout << "  Success: " << (result.success ? "yes" : "no") << "\n";
  std::cout << "  Attempted download: " << (result.attempted ? "yes" : "no") << "\n";
  std::cout << "  Binary path: " << (result.success ? result.executablePath.string() : cachePath.string()) << "\n";
  if (!result.detail.empty()) {
    std::cout << "  Detail: " << result.detail << "\n";
  }

  return result.success ? 0 : 1;
}

int commandExportFeatures(const std::vector<std::string>& args) {
  const auto sessionPathArg = argValue(args, "--session");
  const auto outPathArg = argValue(args, "--out");
  if (!sessionPathArg.has_value() || !outPathArg.has_value()) {
    std::cerr << "export-features requires --session <session.json> and --out <features.jsonl>\n";
    return 2;
  }

  automix::engine::SessionRepository repository;
  const auto session = repository.load(*sessionPathArg);

  std::unordered_map<std::string, automix::domain::Stem> stemsById;
  for (const auto& stem : session.stems) {
    stemsById.emplace(stem.id, stem);
  }

  automix::analysis::StemAnalyzer analyzer;
  const auto entries = analyzer.analyzeSession(session);

  std::ofstream out(*outPathArg);
  if (!out.is_open()) {
    std::cerr << "Failed to open output file: " << *outPathArg << "\n";
    return 1;
  }

  for (const auto& entry : entries) {
    nlohmann::json line;
    line["sessionName"] = session.sessionName;
    line["stemId"] = entry.stemId;
    line["stemName"] = entry.stemName;
    line["metrics"] = {
        {"peakDb", entry.metrics.peakDb},
        {"rmsDb", entry.metrics.rmsDb},
        {"crestDb", entry.metrics.crestDb},
        {"crestFactor", entry.metrics.crestFactor},
        {"lowEnergy", entry.metrics.lowEnergy},
        {"midEnergy", entry.metrics.midEnergy},
        {"highEnergy", entry.metrics.highEnergy},
        {"silenceRatio", entry.metrics.silenceRatio},
        {"stereoCorrelation", entry.metrics.stereoCorrelation},
        {"stereoWidth", entry.metrics.stereoWidth},
        {"dcOffset", entry.metrics.dcOffset},
        {"subEnergy", entry.metrics.subEnergy},
        {"bassEnergy", entry.metrics.bassEnergy},
        {"lowMidEnergy", entry.metrics.lowMidEnergy},
        {"highMidEnergy", entry.metrics.highMidEnergy},
        {"presenceEnergy", entry.metrics.presenceEnergy},
        {"airEnergy", entry.metrics.airEnergy},
        {"spectralCentroidHz", entry.metrics.spectralCentroidHz},
        {"spectralSpreadHz", entry.metrics.spectralSpreadHz},
        {"spectralFlatness", entry.metrics.spectralFlatness},
        {"spectralFlux", entry.metrics.spectralFlux},
        {"onsetStrength", entry.metrics.onsetStrength},
        {"mfccCoefficients", entry.metrics.mfccCoefficients},
        {"constantQBins", entry.metrics.constantQBins},
        {"channelBalanceDb", entry.metrics.channelBalanceDb},
        {"artifactRisk", entry.metrics.artifactRisk},
        {"artifactSwirlRisk", entry.metrics.artifactProfile.swirlRisk},
        {"artifactSmearRisk", entry.metrics.artifactProfile.smearRisk},
        {"artifactNoiseDominance", entry.metrics.artifactProfile.noiseDominance},
        {"artifactHarmonicity", entry.metrics.artifactProfile.harmonicity},
        {"artifactPhaseInstability", entry.metrics.artifactProfile.phaseInstability},
    };

    const auto it = stemsById.find(entry.stemId);
    if (it != stemsById.end()) {
      line["origin"] = automix::domain::toString(it->second.origin);
      line["role"] = automix::domain::toString(it->second.role);
    }

    out << line.dump() << "\n";
  }

  std::cout << "Exported " << entries.size() << " feature rows to " << *outPathArg << "\n";
  return 0;
}

int commandExportSegments(const std::vector<std::string>& args) {
  const auto sessionPathArg = argValue(args, "--session");
  const auto outDirArg = argValue(args, "--out-dir");
  if (!sessionPathArg.has_value() || !outDirArg.has_value()) {
    std::cerr << "export-segments requires --session <session.json> and --out-dir <directory>\n";
    return 2;
  }

  double segmentSeconds = 5.0;
  if (const auto secondsArg = argValue(args, "--segment-seconds"); secondsArg.has_value()) {
    segmentSeconds = std::max(0.5, std::stod(*secondsArg));
  }

  automix::engine::SessionRepository repository;
  const auto session = repository.load(*sessionPathArg);

  const std::filesystem::path outDir(*outDirArg);
  const std::filesystem::path stemDir = outDir / "stems";
  std::filesystem::create_directories(stemDir);

  automix::engine::AudioFileIO fileIO;
  automix::util::WavWriter writer;
  for (const auto& stem : session.stems) {
    if (!stem.enabled) {
      continue;
    }

    const auto buffer = fileIO.readAudioFile(stem.filePath);
    const int maxSamples = static_cast<int>(segmentSeconds * buffer.getSampleRate());
    const auto segment = sliceBuffer(buffer, maxSamples);
    const auto outPath = stemDir / (sanitizeFileName(stem.id + "_" + stem.name) + ".wav");
    writer.write(outPath, segment, 24);
  }

  automix::domain::RenderSettings settings = session.renderSettings;
  settings.outputSampleRate = settings.outputSampleRate > 0 ? settings.outputSampleRate : 44100;
  settings.blockSize = settings.blockSize > 0 ? settings.blockSize : 1024;
  settings.outputBitDepth = 24;
  settings.outputPath = (outDir / "mix_full.wav").string();

  automix::engine::OfflineRenderPipeline pipeline;
  const auto rawMix = pipeline.renderRawMix(session, settings, {}, nullptr);
  if (rawMix.cancelled) {
    std::cerr << "Raw mix render cancelled while exporting segments.\n";
    return 1;
  }

  const int maxMixSamples = static_cast<int>(segmentSeconds * rawMix.mixBuffer.getSampleRate());
  const auto mixSegment = sliceBuffer(rawMix.mixBuffer, maxMixSamples);
  writer.write(outDir / "mix_segment.wav", mixSegment, 24);

  nlohmann::json manifest = {
      {"sessionName", session.sessionName},
      {"segmentSeconds", segmentSeconds},
      {"stemCount", session.stems.size()},
      {"mixSegmentPath", (outDir / "mix_segment.wav").string()},
  };
  std::ofstream manifestOut(outDir / "manifest.json");
  manifestOut << manifest.dump(2);

  std::cout << "Exported stem and mix segments to " << outDir.string() << "\n";
  return 0;
}

int commandValidateModelPack(const std::vector<std::string>& args) {
  const auto packArg = argValue(args, "--pack");
  if (!packArg.has_value()) {
    std::cerr << "validate-modelpack requires --pack <directory>\n";
    return 2;
  }

  const std::filesystem::path packDir(*packArg);
  automix::ai::ModelPackLoader loader;
  const auto maybePack = loader.load(packDir);
  if (!maybePack.has_value()) {
    std::cerr << "Model pack validation failed: could not load model.json or model file.\n";
    return 1;
  }
  const auto& pack = maybePack.value();

  std::cout << "Model pack loaded: " << pack.id << " engine=" << pack.engine
            << " type=" << pack.type << " version=" << pack.version << "\n";

  std::unique_ptr<automix::ai::IModelInference> inference = std::make_unique<automix::ai::NullModelInference>();
  if (pack.engine == "onnxruntime") {
    inference = std::make_unique<automix::ai::OnnxModelInference>();
  } else if (pack.engine == "rtneural") {
    inference = std::make_unique<automix::ai::RtNeuralInference>();
    if (!inference->isAvailable()) {
      std::cout << "Warning: RTNeural backend not enabled in this build. Schema-only validation performed.\n";
    }
  }

  const auto modelPath = pack.rootPath / pack.modelFile;
  if (!inference->loadModel(modelPath)) {
    if (pack.engine == "unknown") {
      std::cout << "Warning: unknown model engine; skipped runtime inference validation.\n";
      return 0;
    }
    if (!inference->isAvailable()) {
      return 0;
    }
    std::cerr << "Model pack validation failed: backend refused to load model file.\n";
    return 1;
  }

  const size_t featureCount = pack.inputFeatureCount.value_or(automix::ai::FeatureSchemaV1::featureCount());
  const automix::ai::InferenceRequest request{
      .task = taskFromModelType(pack.type),
      .features = deterministicFeatures(featureCount),
  };
  const auto result = inference->run(request);
  if (!result.usedModel) {
    std::cerr << "Model pack validation failed: sample inference did not use model (" << result.logMessage << ")\n";
    return 1;
  }

  for (const auto& key : pack.expectedOutputKeys) {
    if (!result.outputs.contains(key)) {
      std::cerr << "Model pack validation failed: missing expected output key '" << key << "'\n";
      return 1;
    }
  }

  std::cout << "Model pack validation passed.\n";
  return 0;
}

int commandValidateExternalLimiter(const std::vector<std::string>& args) {
  const auto binaryArg = argValue(args, "--binary");
  if (!binaryArg.has_value()) {
    std::cerr << "validate-external-limiter requires --binary <path>\n";
    return 2;
  }

  const bool jsonOutput = hasFlag(args, "--json");
  const std::filesystem::path binaryPath(*binaryArg);
  const auto validation = automix::renderers::ExternalLimiterRenderer::validateBinary(binaryPath);

  if (jsonOutput) {
    nlohmann::json payload = {
        {"binary", binaryPath.string()},
        {"valid", validation.valid},
        {"version", validation.version},
        {"errorCode", validation.errorCode},
        {"diagnostics", validation.diagnostics},
        {"supportedFeatures", validation.supportedFeatures},
    };
    std::cout << payload.dump(2) << "\n";
    return validation.valid ? 0 : 1;
  }

  std::cout << "External limiter validation summary:\n";
  std::cout << "  Binary: " << binaryPath.string() << "\n";
  std::cout << "  Valid: " << (validation.valid ? "yes" : "no") << "\n";
  std::cout << "  Version: " << (validation.version.empty() ? "(none)" : validation.version) << "\n";
  std::cout << "  Error code: " << (validation.errorCode.empty() ? "(none)" : validation.errorCode) << "\n";
  std::cout << "  Diagnostics: " << validation.diagnostics << "\n";

  if (!validation.supportedFeatures.empty()) {
    std::cout << "  Supported features:\n";
    for (const auto& feature : validation.supportedFeatures) {
      std::cout << "    - " << feature << "\n";
    }
  } else {
    std::cout << "  Supported features: (none reported)\n";
  }

  return validation.valid ? 0 : 1;
}

void printUsage() {
  std::cout << "Usage:\n";
  std::cout << "  automix_dev_tools export-features --session <session.json> --out <features.jsonl>\n";
  std::cout << "  automix_dev_tools export-segments --session <session.json> --out-dir <dir> [--segment-seconds <sec>]\n";
  std::cout << "  automix_dev_tools validate-modelpack --pack <modelpack_dir>\n";
  std::cout << "  automix_dev_tools validate-external-limiter --binary <path> [--json]\n";
  std::cout << "  automix_dev_tools list-supported-models\n";
  std::cout << "  automix_dev_tools install-supported-model --id <model_id> [--dest <assets/models>]\n";
  std::cout << "  automix_dev_tools list-supported-limiters\n";
  std::cout << "  automix_dev_tools install-supported-limiter --id <limiter_id> [--dest <assets/limiters>]\n";
  std::cout << "  automix_dev_tools install-lame-fallback [--force] [--json]\n";
}

} // namespace

int main(int argc, char** argv) {
  try {
    std::vector<std::string> args;
    args.reserve(static_cast<size_t>(argc));
    for (int i = 1; i < argc; ++i) {
      args.emplace_back(argv[i]);
    }

    if (args.empty()) {
      printUsage();
      return 2;
    }

    const std::string command = args.front();
    if (command == "export-features") {
      return commandExportFeatures(args);
    }
    if (command == "export-segments") {
      return commandExportSegments(args);
    }
    if (command == "validate-modelpack") {
      return commandValidateModelPack(args);
    }
    if (command == "validate-external-limiter") {
      return commandValidateExternalLimiter(args);
    }
    if (command == "list-supported-models") {
      return commandListSupportedModels();
    }
    if (command == "install-supported-model") {
      return commandInstallSupportedModel(args);
    }
    if (command == "list-supported-limiters") {
      return commandListSupportedLimiters();
    }
    if (command == "install-supported-limiter") {
      return commandInstallSupportedLimiter(args);
    }
    if (command == "install-lame-fallback") {
      return commandInstallLameFallback(args);
    }

    printUsage();
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "Developer tool error: " << error.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "Developer tool error: unknown exception\n";
    return 1;
  }
}
