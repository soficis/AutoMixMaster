#include <algorithm>
#include <cctype>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "ai/IModelInference.h"
#include "ai/ModelPackLoader.h"
#ifdef ENABLE_ONNX
#include "ai/OnnxModelInference.h"
#endif
#include "ai/RtNeuralInference.h"
#include "analysis/StemAnalyzer.h"
#include "domain/Stem.h"
#include "domain/StemOrigin.h"
#include "domain/StemRole.h"
#include "engine/AudioFileIO.h"
#include "engine/OfflineRenderPipeline.h"
#include "engine/SessionRepository.h"
#include "util/WavWriter.h"

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
        {"lowEnergy", entry.metrics.lowEnergy},
        {"midEnergy", entry.metrics.midEnergy},
        {"highEnergy", entry.metrics.highEnergy},
        {"silenceRatio", entry.metrics.silenceRatio},
        {"stereoCorrelation", entry.metrics.stereoCorrelation},
        {"stereoWidth", entry.metrics.stereoWidth},
        {"artifactRisk", entry.metrics.artifactRisk},
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
#ifdef ENABLE_ONNX
    inference = std::make_unique<automix::ai::OnnxModelInference>();
#else
    std::cout << "Warning: ONNX backend not enabled in this build. Schema-only validation performed.\n";
#endif
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

  const size_t featureCount = pack.inputFeatureCount.value_or(5u);
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

void printUsage() {
  std::cout << "Usage:\n";
  std::cout << "  automix_dev_tools export-features --session <session.json> --out <features.jsonl>\n";
  std::cout << "  automix_dev_tools export-segments --session <session.json> --out-dir <dir> [--segment-seconds <sec>]\n";
  std::cout << "  automix_dev_tools validate-modelpack --pack <modelpack_dir>\n";
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
