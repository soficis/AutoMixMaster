#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include "ai/BsRoformerPack.h"
#include "ai/ITensorInference.h"
#include "ai/ModelCatalogValidator.h"
#include "ai/ModelLicensePolicy.h"
#include "ai/ModelPackLoader.h"
#include "ai/OnnxTensorInference.h"
#include "ai/SeparationRunner.h"
#include "ai/StemSeparator.h"
#include "ai/TensorTypes.h"
#include "analysis/SpectrogramFrontEnd.h"
#include "domain/RenderSettings.h"
#include "engine/AudioBuffer.h"
#include "engine/AudioFileIO.h"
#include "util/WavWriter.h"
#include "TorchStftGolden.h"

namespace ai = automix::ai;
namespace analysis = automix::analysis;
namespace engine = automix::engine;

namespace {

// BS-RoFormer's STFT (spec 5.4.1).
analysis::StftParams roformerStft() {
  analysis::StftParams params;
  params.nFft = 2048;
  params.hopLength = 441;
  params.winLength = 2048;
  params.center = true;
  params.normalized = false;
  params.zeroDc = true;
  return params;
}

engine::AudioBuffer makeTestSignal(const int channels, const int samples, const double sampleRate) {
  engine::AudioBuffer buffer(channels, samples, sampleRate);
  for (int ch = 0; ch < channels; ++ch) {
    for (int i = 0; i < samples; ++i) {
      const double t = static_cast<double>(i) / sampleRate;
      const double value = 0.4 * std::sin(2.0 * 3.14159265358979323846 * (220.0 + 110.0 * ch) * t) +
                           0.2 * std::sin(2.0 * 3.14159265358979323846 * 3130.0 * t + 0.3 * ch) +
                           0.05 * std::sin(0.37 * static_cast<double>(i * (ch + 3)));
      buffer.setSample(ch, i, static_cast<float>(value));
    }
  }
  return buffer;
}

float maxAbsDifference(const engine::AudioBuffer& a, const engine::AudioBuffer& b) {
  float worst = 0.0f;
  for (int ch = 0; ch < a.getNumChannels(); ++ch) {
    for (int i = 0; i < a.getNumSamples(); ++i) {
      worst = std::max(worst, std::abs(a.getSample(ch, i) - b.getSample(ch, i)));
    }
  }
  return worst;
}

// Records the bindings it is handed and answers with a scripted output. This is
// the stand-in for a real graph that lets the runner be proven with no ORT SDK.
class FakeTensorInference final : public ai::ITensorInference {
 public:
  using Script = std::function<ai::TensorInferenceResult(const std::vector<ai::TensorBinding>&)>;

  FakeTensorInference(std::vector<ai::TensorSpec> inputs, std::vector<ai::TensorSpec> outputs, Script script)
      : inputs_(std::move(inputs)), outputs_(std::move(outputs)), script_(std::move(script)) {}

  bool isAvailable() const override { return true; }
  bool loadModel(const std::filesystem::path&) override { return true; }
  std::vector<ai::TensorSpec> inputSpecs() const override { return inputs_; }
  std::vector<ai::TensorSpec> outputSpecs() const override { return outputs_; }
  ai::TensorInferenceResult run(const std::vector<ai::TensorBinding>& inputs) const override {
    ++calls;
    lastBindings = inputs;
    return script_(inputs);
  }

  mutable int calls = 0;
  mutable std::vector<ai::TensorBinding> lastBindings;

 private:
  std::vector<ai::TensorSpec> inputs_;
  std::vector<ai::TensorSpec> outputs_;
  Script script_;
};

const ai::TensorSpec kRoformerInput{"mix_spec", ai::TensorElementType::Float32, {1, 801, 4100}};
const ai::TensorSpec kRoformerOutput{"vocals_mask", ai::TensorElementType::Float32, {1, 1, 2050, 801, 2}};

// A mask of 1 + 0i: the masked spectrum is the input spectrum.
ai::TensorInferenceResult identityMask(const std::vector<ai::TensorBinding>&) {
  ai::TensorInferenceResult result;
  result.usedModel = true;
  ai::Tensor mask;
  mask.spec = kRoformerOutput;
  mask.data.assign(1 * 1 * 2050 * 801 * 2, 0.0f);
  for (std::size_t i = 0; i < mask.data.size(); i += 2) {
    mask.data[i] = 1.0f;
  }
  result.outputs.push_back(std::move(mask));
  return result;
}

ai::RunnerConfig roformerRunnerConfig() {
  ai::RunnerConfig config;
  config.stft = roformerStft();
  config.inputLayout = ai::InputLayout::FoldedStereo;
  config.outputMode = ai::OutputMode::Mask;
  config.targetStem = "vocals";
  config.stems = {{"vocals", ""}, {"instrumental", "vocals"}};
  return config;
}

// Spec section 6, installed form (names present).
nlohmann::json roformerContractJson() {
  return nlohmann::json::parse(R"({
    "engine": "bs_roformer",
    "sample_rate": 44100,
    "stereo": true,
    "chunk_samples": 352800,
    "overlap_samples": 88200,
    "stft": { "n_fft": 2048, "hop_length": 441, "win_length": 2048,
              "window": "hann", "periodic_window": true,
              "center": true, "pad_mode": "reflect", "normalized": false, "zero_dc": true },
    "input_layout": "folded_stereo",
    "inputs":  [ { "name": "input",  "shape": [1, 801, 4100],      "dtype": "float32" } ],
    "outputs": [ { "name": "output", "shape": [1, 1, 2050, 801, 2], "dtype": "float32" } ],
    "output_mode": "mask",
    "target_stem": "vocals",
    "stems": [ { "name": "vocals" }, { "name": "instrumental", "residual_of": "vocals" } ]
  })");
}

ai::TensorContract parseContract(const nlohmann::json& json) {
  std::string error;
  const auto contract = ai::parseTensorContract(json, error);
  REQUIRE(contract.has_value());
  REQUIRE(error.empty());
  return *contract;
}

const std::vector<ai::TensorSpec> kProbedInputs{{"input", ai::TensorElementType::Float32, {1, 801, 4100}}};
const std::vector<ai::TensorSpec> kProbedOutputs{{"output", ai::TensorElementType::Float32, {1, 1, 2050, 801, 2}}};

std::string contractError(const nlohmann::json& json,
                          const std::vector<ai::TensorSpec>& inputs = kProbedInputs,
                          const std::vector<ai::TensorSpec>& outputs = kProbedOutputs) {
  const auto contract = parseContract(json);
  std::string error;
  const bool ok = ai::checkTensorContract(contract, inputs, outputs, error);
  REQUIRE_FALSE(ok);
  REQUIRE_FALSE(error.empty());
  return error;
}

} // namespace

TEST_CASE("Tensor contract validation table", "[ai][tensor]") {
  const ai::TensorSpec dynamicBatch{"input", ai::TensorElementType::Float32, {-1, 801, 4100}};
  const ai::TensorSpec probed{"input", ai::TensorElementType::Float32, {1, 801, 4100}};
  const ai::TensorSpec probedBatch4{"input", ai::TensorElementType::Float32, {4, 801, 4100}};
  REQUIRE(ai::shapesMatch(dynamicBatch, probed));
  REQUIRE(ai::shapesMatch(dynamicBatch, probedBatch4));

  const ai::TensorSpec wrongRank{"input", ai::TensorElementType::Float32, {1, 2, 1025, 801}};
  REQUIRE_FALSE(ai::shapesMatch(dynamicBatch, wrongRank));
  REQUIRE_FALSE(ai::shapesMatch(probed, wrongRank));

  const ai::TensorSpec wrongStaticDim{"input", ai::TensorElementType::Float32, {1, 796, 4100}};
  REQUIRE_FALSE(ai::shapesMatch(probed, wrongStaticDim));
  REQUIRE_FALSE(ai::shapesMatch(dynamicBatch, wrongStaticDim));

  // An expected dynamic axis matches anything, but a dynamic probed axis does not
  // satisfy a static expectation: the graph promised less than was declared.
  const ai::TensorSpec probedDynamic{"input", ai::TensorElementType::Float32, {1, -1, 4100}};
  REQUIRE_FALSE(ai::shapesMatch(probed, probedDynamic));

  REQUIRE(ai::elementCount(probed) == std::optional<std::size_t>(3284100));
  REQUIRE(ai::elementCount(dynamicBatch) == std::optional<std::size_t>(3284100));
  const ai::TensorSpec twoDynamic{"input", ai::TensorElementType::Float32, {-1, -1, 4100}};
  REQUIRE_FALSE(ai::elementCount(twoDynamic).has_value());
  const ai::TensorSpec output{"output", ai::TensorElementType::Float32, {1, 1, 2050, 801, 2}};
  REQUIRE(ai::elementCount(output) == std::optional<std::size_t>(3284100));
  REQUIRE(ai::elementCount(ai::TensorSpec{"scalar", ai::TensorElementType::Float32, {}}) ==
          std::optional<std::size_t>(1));

  REQUIRE(ai::describeShape({1, 801, 4100}) == "[1, 801, 4100]");
  REQUIRE(ai::describeShape({-1, 2}) == "[-1, 2]");
  REQUIRE(ai::describeShape({}) == "[]");
  REQUIRE(ai::describeDtype(ai::TensorElementType::Float32) == "float32");
}

TEST_CASE("Null tensor inference reports unavailable", "[ai][tensor]") {
  ai::NullTensorInference null;
  REQUIRE(!null.isAvailable());
  REQUIRE(!null.loadModel("any.onnx"));
  REQUIRE(!null.lastLog().empty());
  REQUIRE(null.inputSpecs().empty());
  REQUIRE(null.outputSpecs().empty());
  const auto result = null.run({});
  REQUIRE(result.usedModel == false);
  REQUIRE(result.outputs.empty());
  REQUIRE(result.logMessage.find("NullTensorInference") != std::string::npos);
}

TEST_CASE("STFT round-trip identity with exact length", "[ai][tensor][stft]") {
  const auto params = roformerStft();
  const auto input = makeTestSignal(2, 352800, 44100.0);
  const auto spectrum = analysis::analyze(input, params);
  REQUIRE(spectrum.channels == 2);
  REQUIRE(spectrum.freqBins == 1025);
  REQUIRE(spectrum.frames == 801);

  const auto synth = analysis::synthesize(spectrum, params);
  // Equality, not tolerance: any drift here accumulates at every chunk boundary.
  REQUIRE(synth.getNumSamples() == 441 * (801 - 1));
  REQUIRE(synth.getNumSamples() == input.getNumSamples());
  REQUIRE(synth.getNumChannels() == 2);
  REQUIRE(maxAbsDifference(input, synth) < 1.0e-5f);
}

TEST_CASE("Analytic DC check", "[ai][tensor][stft]") {
  // Premise: reflect-padding a constant yields a constant, so every frame of a
  // constant signal is constant and bin 0 equals sum(window) exactly. Switching
  // the pad mode (e.g. to zero padding) invalidates this expected value and must
  // be a deliberate, visible change.
  const auto params = roformerStft();
  engine::AudioBuffer ones(2, 44100, 44100.0);
  for (int ch = 0; ch < 2; ++ch) {
    for (int i = 0; i < ones.getNumSamples(); ++i) {
      ones.setSample(ch, i, 1.0f);
    }
  }
  const auto spectrum = analysis::analyze(ones, params);
  // Periodic Hann of length 2048 sums to exactly 1024.
  constexpr double windowSum = 1024.0;
  double worstDc = 0.0;
  double worstDcImag = 0.0;
  for (int ch = 0; ch < spectrum.channels; ++ch) {
    for (int frame = 0; frame < spectrum.frames; ++frame) {
      const auto index = spectrum.index(ch, 0, frame);
      worstDc = std::max(worstDc, std::abs(static_cast<double>(spectrum.real[index]) - windowSum));
      worstDcImag = std::max(worstDcImag, std::abs(static_cast<double>(spectrum.imag[index])));
    }
  }
  REQUIRE(worstDc < 1.0e-3);
  REQUIRE(worstDcImag < 1.0e-3);
}

TEST_CASE("Complex multiply rejects a mask whose geometry differs", "[ai][tensor][stft]") {
  const auto params = roformerStft();
  const auto spectrum = analysis::analyze(makeTestSignal(2, 8820, 44100.0), params);
  auto shortMask = spectrum;
  shortMask.frames -= 1;
  bool threw = false;
  try {
    static_cast<void>(analysis::complexMultiply(spectrum, shortMask, true));
  } catch (const std::invalid_argument& error) {
    threw = true;
    REQUIRE(std::string(error.what()).find("does not match") != std::string::npos);
  }
  REQUIRE(threw);

  // zeroDc clears bin 0 of the product and nothing else.
  auto unitMask = spectrum;
  std::fill(unitMask.real.begin(), unitMask.real.end(), 1.0f);
  std::fill(unitMask.imag.begin(), unitMask.imag.end(), 0.0f);
  const auto product = analysis::complexMultiply(spectrum, unitMask, true);
  REQUIRE(product.real[product.index(1, 0, 3)] == 0.0f);
  REQUIRE(product.real[product.index(1, 7, 3)] == spectrum.real[spectrum.index(1, 7, 3)]);
  REQUIRE(product.imag[product.index(1, 7, 3)] == spectrum.imag[spectrum.index(1, 7, 3)]);
}

TEST_CASE("Runner binds by declared name", "[ai][tensor][runner]") {
  FakeTensorInference fake({kRoformerInput}, {kRoformerOutput}, identityMask);
  const auto mix = makeTestSignal(2, 352800, 44100.0);
  const auto result = ai::SeparationRunner::separate(mix, fake, roformerRunnerConfig());
  REQUIRE(result.usedModel);
  REQUIRE(fake.calls == 1);
  REQUIRE(fake.lastBindings.size() == 1);
  const auto& binding = fake.lastBindings.front();
  REQUIRE(binding.expected.name == "mix_spec");
  REQUIRE(binding.expected.elementType == ai::TensorElementType::Float32);
  REQUIRE(binding.expected.dims == std::vector<int64_t>{1, 801, 4100});
  REQUIRE(binding.data.size() == std::size_t{801} * 4100);

  // Folded-stereo placement: element (t, (f * channels + ch) * 2 + reim).
  const auto spectrum = analysis::analyze(mix, roformerStft());
  for (const int frame : {0, 17, 800}) {
    for (const int bin : {0, 5, 1024}) {
      for (const int ch : {0, 1}) {
        const auto base = static_cast<std::size_t>(frame) * 4100 + (static_cast<std::size_t>(bin) * 2 + static_cast<std::size_t>(ch)) * 2;
        REQUIRE(binding.data[base] == spectrum.real[spectrum.index(ch, bin, frame)]);
        REQUIRE(binding.data[base + 1] == spectrum.imag[spectrum.index(ch, bin, frame)]);
      }
    }
  }

  REQUIRE(result.stemNames == std::vector<std::string>{"vocals", "instrumental"});
  REQUIRE(result.stemAudio.size() == 2);
  REQUIRE(result.stemAudio[0].getNumSamples() == mix.getNumSamples());
}

TEST_CASE("Wrong-size output falls back honestly", "[ai][tensor][runner]") {
  const auto mix = makeTestSignal(2, 352800, 44100.0);

  FakeTensorInference wrongFrames({kRoformerInput}, {kRoformerOutput}, [](const auto&) {
    ai::TensorInferenceResult result;
    result.usedModel = true;
    ai::Tensor mask;
    mask.spec = {"vocals_mask", ai::TensorElementType::Float32, {1, 1, 2050, 796, 2}};
    mask.data.assign(2050 * 796 * 2, 1.0f);
    result.outputs.push_back(std::move(mask));
    return result;
  });
  const auto wrongShape = ai::SeparationRunner::separate(mix, wrongFrames, roformerRunnerConfig());
  REQUIRE_FALSE(wrongShape.usedModel);
  REQUIRE(wrongShape.stemAudio.empty());
  REQUIRE(wrongShape.logMessage.find("vocals_mask") != std::string::npos);
  REQUIRE(wrongShape.logMessage.find("[1, 1, 2050, 796, 2]") != std::string::npos);
  REQUIRE(wrongShape.logMessage.find("[1, 1, 2050, 801, 2]") != std::string::npos);

  FakeTensorInference wrongName({kRoformerInput}, {kRoformerOutput}, [](const auto& bindings) {
    auto result = identityMask(bindings);
    result.outputs.front().spec.name = "out_spec_real";
    return result;
  });
  const auto misnamed = ai::SeparationRunner::separate(mix, wrongName, roformerRunnerConfig());
  REQUIRE_FALSE(misnamed.usedModel);
  REQUIRE(misnamed.logMessage.find("vocals_mask") != std::string::npos);

  // All-or-nothing: a failure in a later chunk discards the chunks that worked.
  const auto longMix = makeTestSignal(2, 352800 * 2, 44100.0);
  int progressCalls = 0;
  auto config = roformerRunnerConfig();
  config.progressCallback = [&](int, int) { ++progressCalls; };
  int call = 0;
  FakeTensorInference failsOnSecondChunk({kRoformerInput}, {kRoformerOutput}, [&](const auto& bindings) {
    ++call;
    if (call == 2) {
      ai::TensorInferenceResult failed;
      failed.logMessage = "scripted failure on chunk 2";
      return failed;
    }
    return identityMask(bindings);
  });
  const auto partial = ai::SeparationRunner::separate(longMix, failsOnSecondChunk, config);
  REQUIRE_FALSE(partial.usedModel);
  REQUIRE(partial.stemAudio.empty());
  REQUIRE(partial.logMessage.find("chunk 2/3") != std::string::npos);
  REQUIRE(progressCalls == 1);
}

TEST_CASE("Window-squared normalization under chunking", "[ai][tensor][runner]") {
  // Three chunks (stride 264600): the constant must survive every chunk
  // boundary and crossfade. zero_dc is off here because it would, correctly,
  // remove a constant.
  constexpr int samples = 352800 + 2 * 264600 - 1000;
  engine::AudioBuffer constant(2, samples, 44100.0);
  for (int ch = 0; ch < 2; ++ch) {
    for (int i = 0; i < samples; ++i) {
      constant.setSample(ch, i, 0.5f);
    }
  }
  auto config = roformerRunnerConfig();
  config.stft.zeroDc = false;
  std::vector<std::pair<int, int>> progress;
  config.progressCallback = [&](int done, int total) { progress.emplace_back(done, total); };
  FakeTensorInference fake({kRoformerInput}, {kRoformerOutput}, identityMask);

  REQUIRE(ai::SeparationRunner::chunkCount(samples, config) == 3);
  const auto result = ai::SeparationRunner::separate(constant, fake, config);
  REQUIRE(result.usedModel);
  REQUIRE(fake.calls == 3);
  REQUIRE(progress == std::vector<std::pair<int, int>>{{1, 3}, {2, 3}, {3, 3}});
  REQUIRE(result.stemAudio[0].getNumSamples() == samples);
  REQUIRE(maxAbsDifference(result.stemAudio[0], constant) < 1.0e-5f);

  engine::AudioBuffer silence(2, samples, 44100.0);
  REQUIRE(maxAbsDifference(result.stemAudio[1], silence) < 1.0e-5f);

  // A non-constant signal also reconstructs through the same path, and the
  // residual stem is exactly mix - target.
  const auto music = makeTestSignal(2, samples, 44100.0);
  const auto separated = ai::SeparationRunner::separate(music, fake, config);
  REQUIRE(separated.usedModel);
  REQUIRE(maxAbsDifference(separated.stemAudio[0], music) < 1.0e-4f);
  float worstSum = 0.0f;
  for (int ch = 0; ch < 2; ++ch) {
    for (int i = 0; i < samples; ++i) {
      const float sum = separated.stemAudio[0].getSample(ch, i) + separated.stemAudio[1].getSample(ch, i);
      worstSum = std::max(worstSum, std::abs(sum - music.getSample(ch, i)));
    }
  }
  REQUIRE(worstSum < 1.0e-5f);
}

TEST_CASE("Runner refuses input the contract does not describe", "[ai][tensor][runner]") {
  FakeTensorInference fake({kRoformerInput}, {kRoformerOutput}, identityMask);

  const auto at48k = makeTestSignal(2, 48000, 48000.0);
  const auto wrongRate = ai::SeparationRunner::separate(at48k, fake, roformerRunnerConfig());
  REQUIRE_FALSE(wrongRate.usedModel);
  REQUIRE(wrongRate.logMessage.find("44100") != std::string::npos);
  REQUIRE(wrongRate.logMessage.find("48000") != std::string::npos);

  const auto mono = makeTestSignal(1, 44100, 44100.0);
  const auto monoResult = ai::SeparationRunner::separate(mono, fake, roformerRunnerConfig());
  REQUIRE_FALSE(monoResult.usedModel);
  REQUIRE(monoResult.logMessage.find("1 channel") != std::string::npos);
  REQUIRE(fake.calls == 0);

  // A track shorter than one chunk is one zero-padded partial chunk; with no
  // neighbour the crossfade is the identity.
  const auto shortMix = makeTestSignal(2, 100000, 44100.0);
  const auto shortResult = ai::SeparationRunner::separate(shortMix, fake, roformerRunnerConfig());
  REQUIRE(shortResult.usedModel);
  REQUIRE(fake.calls == 1);
  REQUIRE(shortResult.stemAudio[0].getNumSamples() == 100000);
  auto noDc = roformerRunnerConfig();
  noDc.stft.zeroDc = false;
  const auto shortIdentity = ai::SeparationRunner::separate(shortMix, fake, noDc);
  REQUIRE(maxAbsDifference(shortIdentity.stemAudio[0], shortMix) < 1.0e-4f);
}

TEST_CASE("Golden torch.stft fixture", "[ai][tensor][stft]") {
  // Catches a consistent-but-wrong convention (window periodicity, pad mode,
  // sign of the exponent, bin order) that the round-trip test cannot, because
  // forward and inverse would share the misreading. See TorchStftGolden.h for
  // the generating expression.
  analysis::StftParams params;
  params.nFft = 256;
  params.hopLength = 64;
  params.winLength = 256;
  params.center = true;
  params.normalized = false;
  params.zeroDc = false;

  engine::AudioBuffer ramp(1, 4096, 44100.0);
  for (int i = 0; i < 4096; ++i) {
    ramp.setSample(0, i, static_cast<float>(i) / 4096.0f);
  }
  const auto spectrum = analysis::analyze(ramp, params);
  REQUIRE(spectrum.freqBins == torch_stft_golden::kFreqBins);
  REQUIRE(spectrum.frames == torch_stft_golden::kFrames);

  const auto toFloat = [](const std::uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  };
  double worst = 0.0;
  for (int bin = 0; bin < spectrum.freqBins; ++bin) {
    for (int frame = 0; frame < spectrum.frames; ++frame) {
      const auto golden = static_cast<std::size_t>(bin * spectrum.frames + frame);
      const auto index = spectrum.index(0, bin, frame);
      worst = std::max(worst, std::abs(static_cast<double>(spectrum.real[index]) - toFloat(torch_stft_golden::kReal[golden])));
      worst = std::max(worst, std::abs(static_cast<double>(spectrum.imag[index]) - toFloat(torch_stft_golden::kImag[golden])));
    }
  }
  // Peak magnitude here is ~128 (bin 0, sum of a 256-tap Hann); 1e-5 absolute
  // is float32 rounding at that scale, far below any convention error.
  REQUIRE(worst < 1.0e-5);
}

TEST_CASE("Tensor contract load-time failures", "[ai][tensor][contract]") {
  std::string error;
  REQUIRE(ai::checkTensorContract(parseContract(roformerContractJson()), kProbedInputs, kProbedOutputs, error));
  // A graph with a dynamic batch axis accepts the declared static shape.
  REQUIRE(ai::checkTensorContract(parseContract(roformerContractJson()),
                                  {{"input", ai::TensorElementType::Float32, {-1, 801, 4100}}}, kProbedOutputs, error));

  auto wrongFft = roformerContractJson();
  wrongFft["stft"]["n_fft"] = 1024;
  wrongFft["stft"]["win_length"] = 1024;
  auto message = contractError(wrongFft);
  REQUIRE(message.find("input 'input'") != std::string::npos);
  REQUIRE(message.find("[1, 801, 4100]") != std::string::npos);
  REQUIRE(message.find("[1, 801, 2052]") != std::string::npos);

  message = contractError(roformerContractJson(), {{"input", ai::TensorElementType::Float32, {1, 796, 4100}}});
  REQUIRE(message.find("'input'") != std::string::npos);
  REQUIRE(message.find("[1, 796, 4100]") != std::string::npos);
  REQUIRE(message.find("[1, 801, 4100]") != std::string::npos);

  message = contractError(roformerContractJson(), kProbedInputs,
                          {{"output", ai::TensorElementType::Float32, {1, 2, 2050, 801, 2}}});
  REQUIRE(message.find("'output'") != std::string::npos);

  message = contractError(roformerContractJson(), {{"mix", ai::TensorElementType::Float32, {1, 801, 4100}}});
  REQUIRE(message.find("'input'") != std::string::npos);

  message = contractError(roformerContractJson(), {{"input", ai::TensorElementType::Float32, {1, 801, 4100}},
                                                   {"extra", ai::TensorElementType::Float32, {1}}});
  REQUIRE(message.find("'extra'") != std::string::npos);

  auto badLayout = roformerContractJson();
  badLayout["input_layout"] = "interleaved";
  REQUIRE(contractError(badLayout).find("'interleaved'") != std::string::npos);

  auto badMode = roformerContractJson();
  badMode["output_mode"] = "banana";
  REQUIRE(contractError(badMode).find("'banana'") != std::string::npos);

  auto badPad = roformerContractJson();
  badPad["stft"]["pad_mode"] = "constant";
  REQUIRE(contractError(badPad).find("'constant'") != std::string::npos);

  auto badDtype = roformerContractJson();
  badDtype["outputs"][0]["dtype"] = "float16";
  message = contractError(badDtype);
  REQUIRE(message.find("'output'") != std::string::npos);
  REQUIRE(message.find("float16") != std::string::npos);

  auto badTarget = roformerContractJson();
  badTarget["target_stem"] = "drums";
  REQUIRE(contractError(badTarget).find("'drums'") != std::string::npos);

  auto missingField = roformerContractJson();
  missingField["stft"].erase("zero_dc");
  REQUIRE_FALSE(ai::parseTensorContract(missingField, error).has_value());
  REQUIRE(error.find("zero_dc") != std::string::npos);
}

TEST_CASE("Tensor contract manifest round-trips in installed and catalog form", "[ai][tensor][contract]") {
  // Catalog form: names omitted, filled by the install-time probe.
  auto catalog = roformerContractJson();
  catalog["inputs"][0].erase("name");
  catalog["outputs"][0].erase("name");
  const auto catalogContract = parseContract(catalog);
  REQUIRE(catalogContract.inputs.front().name.empty());
  std::string error;
  REQUIRE(ai::checkTensorContract(catalogContract, {{"whatever_the_export_called_it", ai::TensorElementType::Float32,
                                                     {1, 801, 4100}}},
                                  kProbedOutputs, error));
  REQUIRE(ai::tensorContractToJson(catalogContract) == catalog);

  const auto installed = parseContract(roformerContractJson());
  REQUIRE(ai::tensorContractToJson(installed) == roformerContractJson());

  const auto config = ai::runnerConfigFromContract(installed, error);
  REQUIRE(config.has_value());
  REQUIRE(config->chunkSamples == 352800);
  REQUIRE(config->overlapSamples == 88200);
  REQUIRE(config->channels == 2);
  REQUIRE(config->stft.nFft == 2048);
  REQUIRE(config->stft.hopLength == 441);
  REQUIRE(config->stft.zeroDc);
  REQUIRE(config->inputLayout == ai::InputLayout::FoldedStereo);
  REQUIRE(config->outputMode == ai::OutputMode::Mask);
  REQUIRE(config->stems.size() == 2);
  REQUIRE(config->stems[1].residualOf == "vocals");

  // Writer -> loader, end to end: the tensor pack loads, and a null contract
  // leaves the manifest without the block (scalar packs unchanged).
  const auto root = std::filesystem::temp_directory_path() / "automix_tensor_contract_manifest";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  {
    std::ofstream model(root / "bs_roformer.onnx", std::ios::binary);
    model << "not a real graph";
  }
  ai::HubModelInfo info;
  info.repoId = "xycld/BS-RoFormer-ONNX";
  info.modelId = "xycld/BS-RoFormer-ONNX";
  info.license = "mit";
  info.sourceUrl = "https://huggingface.co/xycld/BS-RoFormer-ONNX";
  ai::HubInstallResult install;
  install.primaryFilePath = root / "bs_roformer.onnx";
  ai::ModelCompatibilityResult compatibility;
  compatibility.compatible = true;
  compatibility.taskScope = "separation";
  compatibility.packType = "separation_model";
  compatibility.engine = "bs_roformer";
  compatibility.expectedOutputKeys = {"output"};

  REQUIRE(ai::writeTurnkeyModelPackManifest(root, info, install, compatibility, &installed, &error));
  ai::ModelPackLoader loader;
  const auto pack = loader.load(root);
  REQUIRE(pack.has_value());
  REQUIRE(pack->tensorContract.has_value());
  REQUIRE(ai::tensorContractToJson(*pack->tensorContract) == roformerContractJson());

  REQUIRE(ai::writeTurnkeyModelPackManifest(root, info, install, compatibility, nullptr, &error));
  {
    // Scoped: Windows cannot remove_all a directory holding an open file.
    std::ifstream manifestIn(root / "model.json");
    REQUIRE_FALSE(nlohmann::json::parse(manifestIn).contains("tensor_contract"));
  }
  const auto scalarPack = loader.load(root);
  REQUIRE(scalarPack.has_value());
  REQUIRE_FALSE(scalarPack->tensorContract.has_value());
  std::filesystem::remove_all(root);
}

TEST_CASE("Onnx tensor inference without a model stays unavailable", "[ai][tensor][onnx]") {
  ai::OnnxTensorInference inference;
  REQUIRE_FALSE(inference.isAvailable());
  REQUIRE_FALSE(inference.loadModel(std::filesystem::temp_directory_path() / "automix_no_such_model.onnx"));
  REQUIRE(inference.backendDiagnostics().find("automix_no_such_model.onnx") != std::string::npos);
  REQUIRE(inference.inputSpecs().empty());
  const auto result = inference.run({});
  REQUIRE_FALSE(result.usedModel);
  REQUIRE(result.outputs.empty());
}

#ifdef AUTOMIX_HAS_NATIVE_ORT

namespace {

// Synthetic graphs generated by tests/fixtures/tensor/make_fixtures.py.
std::filesystem::path tensorFixture(const char* name) {
  return std::filesystem::path(AUTOMIX_SOURCE_DIR) / "tests" / "fixtures" / "tensor" / name;
}

} // namespace

TEST_CASE("Native tensor inference probes the real graph", "[ai][tensor][onnx][native]") {
  // Weights are download-only and never committed. Point this at an installed
  // BS-RoFormer export to run the probe against the real graph.
  const char* modelPath = std::getenv("AUTOMIX_BS_ROFORMER_ONNX");
  if (modelPath == nullptr || !std::filesystem::is_regular_file(modelPath)) {
    SKIP("AUTOMIX_BS_ROFORMER_ONNX does not name a BS-RoFormer .onnx file");
  }

  auto catalog = roformerContractJson();
  catalog["inputs"][0].erase("name");
  catalog["outputs"][0].erase("name");
  ai::OnnxTensorInference inference;
  inference.setTensorContract(parseContract(catalog));
  const bool loaded = inference.loadModel(modelPath);
  INFO(inference.backendDiagnostics());
  REQUIRE(loaded);
  REQUIRE(inference.usingNativeSession());
  REQUIRE(inference.inputSpecs().size() == 1);
  REQUIRE(inference.outputSpecs().size() == 1);
  REQUIRE(ai::shapesMatch(inference.inputSpecs().front(), kRoformerInput));
  REQUIRE(ai::shapesMatch(inference.outputSpecs().front(), kRoformerOutput));
}

TEST_CASE("Native tensor inference separates through a synthetic mask graph", "[ai][tensor][onnx][native]") {
  ai::OnnxTensorInference inference;
  inference.setTensorContract(parseContract(roformerContractJson()));
  const bool loaded = inference.loadModel(tensorFixture("identity_mask.onnx"));
  INFO(inference.backendDiagnostics());
  REQUIRE(loaded);
  REQUIRE(inference.usingNativeSession());
  REQUIRE(inference.inputSpecs().size() == 1);
  REQUIRE(inference.inputSpecs().front().name == "input");
  REQUIRE(inference.inputSpecs().front().dims == std::vector<int64_t>{1, 801, 4100});
  REQUIRE(inference.outputSpecs().front().name == "output");
  REQUIRE(inference.outputSpecs().front().dims == std::vector<int64_t>{1, 1, 2050, 801, 2});

  // Two chunks, so the native session runs more than once and the crossfade is exercised.
  const int samples = 352800 + 120000;
  const auto mix = makeTestSignal(2, samples, 44100.0);
  // zero_dc is off so a 1 + 0i mask is an exact identity; with it on, each
  // frame's DC bin is (correctly) removed and the vocals differ from the mix.
  auto config = roformerRunnerConfig();
  config.stft.zeroDc = false;
  REQUIRE(ai::SeparationRunner::chunkCount(samples, config) == 2);
  const auto result = ai::SeparationRunner::separate(mix, inference, config);
  INFO(result.logMessage);
  REQUIRE(result.usedModel);
  REQUIRE(result.stemNames == std::vector<std::string>{"vocals", "instrumental"});
  REQUIRE(result.stemAudio[0].getNumSamples() == samples);
  REQUIRE(maxAbsDifference(result.stemAudio[0], mix) < 1.0e-4f);
  const engine::AudioBuffer silence(2, samples, 44100.0);
  REQUIRE(maxAbsDifference(result.stemAudio[1], silence) < 1.0e-4f);

  // With the contract's zero_dc on, the residual still sums back to the mix exactly.
  const auto zeroDc = ai::SeparationRunner::separate(mix, inference, roformerRunnerConfig());
  REQUIRE(zeroDc.usedModel);
  float worstSum = 0.0f;
  for (int ch = 0; ch < 2; ++ch) {
    for (int i = 0; i < samples; ++i) {
      const float sum = zeroDc.stemAudio[0].getSample(ch, i) + zeroDc.stemAudio[1].getSample(ch, i);
      worstSum = std::max(worstSum, std::abs(sum - mix.getSample(ch, i)));
    }
  }
  REQUIRE(worstSum < 1.0e-5f);

  // Bindings are matched by name; an unknown name is refused, never bound by position.
  ai::TensorBinding wrongName{{"mix", ai::TensorElementType::Float32, {1, 801, 4100}},
                              std::vector<float>(801 * 4100, 0.0f)};
  const auto refused = inference.run({wrongName});
  REQUIRE_FALSE(refused.usedModel);
  REQUIRE(refused.logMessage.find("'input'") != std::string::npos);

  ai::TensorBinding wrongShape{{"input", ai::TensorElementType::Float32, {1, 800, 4100}},
                               std::vector<float>(800 * 4100, 0.0f)};
  const auto misshapen = inference.run({wrongShape});
  REQUIRE_FALSE(misshapen.usedModel);
  REQUIRE(misshapen.logMessage.find("[1, 800, 4100]") != std::string::npos);
}

TEST_CASE("Native tensor inference refuses a contract the graph does not match", "[ai][tensor][onnx][native]") {
  auto wrongGeometry = roformerContractJson();
  wrongGeometry["outputs"][0]["shape"] = {1, 2, 2050, 801, 2};
  wrongGeometry["output_mode"] = "direct";
  wrongGeometry["stems"] = {{{"name", "vocals"}}, {{"name", "instrumental"}}};
  ai::OnnxTensorInference inference;
  inference.setTensorContract(parseContract(wrongGeometry));
  REQUIRE_FALSE(inference.loadModel(tensorFixture("identity_mask.onnx")));
  REQUIRE_FALSE(inference.isAvailable());
  REQUIRE(inference.backendDiagnostics().find("'output'") != std::string::npos);
}

TEST_CASE("Native tensor inference rejects non-float32 graph I/O", "[ai][tensor][onnx][native]") {
  ai::OnnxTensorInference inference;
  REQUIRE_FALSE(inference.loadModel(tensorFixture("int64_io.onnx")));
  REQUIRE_FALSE(inference.isAvailable());
  const auto diagnostics = inference.backendDiagnostics();
  REQUIRE(diagnostics.find("'tokens'") != std::string::npos);
  REQUIRE(diagnostics.find("int64") != std::string::npos);
}

TEST_CASE("Native tensor inference names a missing external-data sidecar", "[ai][tensor][onnx][native]") {
  const auto root = std::filesystem::temp_directory_path() / "automix_tensor_external_data";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  std::filesystem::copy_file(tensorFixture("external_data.onnx"), root / "external_data.onnx");
  std::filesystem::copy_file(tensorFixture("external_data.onnx.data"), root / "external_data.onnx.data");

  ai::OnnxTensorInference present;
  const bool loaded = present.loadModel(root / "external_data.onnx");
  INFO(present.backendDiagnostics());
  REQUIRE(loaded);
  const auto ran = present.run({{{"x", ai::TensorElementType::Float32, {1, 4}}, {0.5f, 0.5f, 0.5f, 0.5f}}});
  REQUIRE(ran.usedModel);
  REQUIRE(ran.outputs.size() == 1);
  REQUIRE(ran.outputs.front().spec.name == "y");
  REQUIRE(ran.outputs.front().data == std::vector<float>{1.5f, 2.5f, 3.5f, 4.5f});

  std::filesystem::remove(root / "external_data.onnx.data");
  ai::OnnxTensorInference missing;
  REQUIRE_FALSE(missing.loadModel(root / "external_data.onnx"));
  REQUIRE_FALSE(missing.isAvailable());
  REQUIRE(missing.backendDiagnostics().find("external_data.onnx.data") != std::string::npos);
  std::filesystem::remove_all(root);
}

#endif

// Spec test 9. Tensor separation is opt-in per session; a default-on flag would
// route every separated import through a new backend and move golden files.
TEST_CASE("Tensor separation defaults off", "[tensor][settings]") {
  const automix::domain::RenderSettings settings;
  REQUIRE_FALSE(settings.tensorSeparationEnabled);
}

namespace {

// Real sibling list of xycld/BS-RoFormer-ONNX (HF API, 2026-09-30), in the
// alphabetical order the API returns.
const std::vector<std::string> kBsRoformerSiblings = {
    ".gitattributes",
    "README.md",
    "bs_roformer_ep317_sdr12.9755.onnx",
    "bs_roformer_ep317_sdr12.9755.onnx.data",
    "bs_roformer_ep317_sdr12.9755_quantized_uint8.onnx",
};

nlohmann::json roformerCatalogJson() {
  auto catalog = roformerContractJson();
  catalog["inputs"][0].erase("name");
  catalog["outputs"][0].erase("name");
  return catalog;
}

// An installed tensor pack as the hub writes it: model.json carrying the
// catalog contract, plus the model file (copied, or a stand-in that no backend
// can open).
std::filesystem::path writeTensorPack(const std::filesystem::path& root, const std::filesystem::path* modelSource) {
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  const auto modelPath = root / "bs_roformer.onnx";
  if (modelSource != nullptr) {
    std::filesystem::copy_file(*modelSource, modelPath);
  } else {
    std::ofstream model(modelPath, std::ios::binary);
    model << "not a real graph";
  }
  ai::HubModelInfo info;
  info.repoId = ai::kBsRoformerRepoId;
  info.modelId = ai::kBsRoformerRepoId;
  info.license = "mit";
  ai::HubInstallResult install;
  install.primaryFilePath = modelPath;
  ai::ModelCompatibilityResult compatibility;
  compatibility.compatible = true;
  compatibility.taskScope = "separation";
  compatibility.packType = "separation_model";
  compatibility.engine = "bs_roformer";
  const auto contract = ai::bsRoformerCatalogContract();
  std::string error;
  REQUIRE(ai::writeTurnkeyModelPackManifest(root, info, install, compatibility, &contract, &error));
  return root;
}

std::vector<char> readBytes(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

} // namespace

TEST_CASE("BS-RoFormer catalog entry installs the quantized build as a separation pack", "[ai][tensor][catalog]") {
  const auto curated = ai::curatedModelIds();
  REQUIRE(std::find(curated.begin(), curated.end(), std::string(ai::kBsRoformerRepoId)) != curated.end());

  // The generic preference takes the first .onnx: the fp32 graph, which is
  // useless without its ~640 MB sidecar. The pin is what prevents that.
  bool hasOnnx = false;
  REQUIRE(ai::primaryFileForRepo("someone/else", kBsRoformerSiblings, &hasOnnx) ==
          "bs_roformer_ep317_sdr12.9755.onnx");
  REQUIRE(ai::primaryFileForRepo(ai::kBsRoformerRepoId, kBsRoformerSiblings, &hasOnnx) ==
          ai::kBsRoformerQuantizedFile);
  REQUIRE(hasOnnx);
  const std::vector<std::string> withoutQuantized(kBsRoformerSiblings.begin(), kBsRoformerSiblings.end() - 1);
  REQUIRE(ai::primaryFileForRepo(ai::kBsRoformerRepoId, withoutQuantized).empty());

  // Classified from the real card data, the way HuggingFaceModelHub::modelInfo does.
  ai::HubModelInfo info;
  info.repoId = ai::kBsRoformerRepoId;
  info.tags = {"onnxruntime", "onnx", "music", "music-source-separation", "audio", "bs-roformer", "roformer",
               "vocals", "instrumental", "quantized", "audio-to-audio", "license:mit"};
  info.files = kBsRoformerSiblings;
  info.primaryFile = ai::primaryFileForRepo(info.repoId, info.files, &info.hasOnnx);
  info.useCase = ai::HuggingFaceModelHub::inferUseCase(info.repoId, info.tags, "");
  const auto compatibility = ai::validateCatalogModel(info);
  INFO(compatibility.reason);
  REQUIRE(compatibility.compatible);
  REQUIRE(compatibility.taskScope == "separation");

  // MIT: licence-driven consent does not prompt; attribution is in NOTICE.
  REQUIRE_FALSE(ai::ModelLicensePolicy::requiresUserConsent("mit"));
}

TEST_CASE("BS-RoFormer catalog contract is the spec block with names omitted", "[ai][tensor][catalog]") {
  const auto contract = ai::bsRoformerCatalogContract();
  REQUIRE(ai::tensorContractToJson(contract) == roformerCatalogJson());
  std::string error;
  REQUIRE(ai::checkTensorContract(contract, kProbedInputs, kProbedOutputs, error));
}

TEST_CASE("Installed tensor contract takes the graph's names only when a graph can be probed", "[ai][tensor][catalog]") {
  const auto catalog = ai::bsRoformerCatalogContract();
  std::string error;
#ifdef AUTOMIX_HAS_NATIVE_ORT
  const auto fixture = std::filesystem::path(AUTOMIX_SOURCE_DIR) / "tests" / "fixtures" / "tensor" / "identity_mask.onnx";
  const auto installed = ai::resolveInstalledTensorContract(catalog, fixture, error);
  INFO(error);
  REQUIRE(installed.has_value());
  REQUIRE(ai::tensorContractToJson(*installed) == roformerContractJson());

  // A graph that does not satisfy the contract fails the install, named.
  const auto int64Graph = fixture.parent_path() / "int64_io.onnx";
  REQUIRE_FALSE(ai::resolveInstalledTensorContract(catalog, int64Graph, error).has_value());
  REQUIRE_FALSE(error.empty());
#else
  const auto installed =
      ai::resolveInstalledTensorContract(catalog, std::filesystem::temp_directory_path() / "unprobed.onnx", error);
  REQUIRE(installed.has_value());
  REQUIRE(ai::tensorContractToJson(*installed) == roformerCatalogJson());
#endif
}

TEST_CASE("Tensor flag off leaves separation byte-identical; on falls back honestly", "[ai][tensor][separator]") {
  const auto tempRoot = std::filesystem::temp_directory_path() / "automix_tensor_flag_separation";
  std::filesystem::remove_all(tempRoot);
  std::filesystem::create_directories(tempRoot);
  const auto mixPath = tempRoot / "mix.wav";
  automix::util::WavWriter writer;
  writer.write(mixPath, makeTestSignal(2, 44100, 44100.0), 24);

  const auto packRoot = writeTensorPack(tempRoot / "pack", nullptr);
  const auto emptyRoot = tempRoot / "no_pack";
  std::filesystem::create_directories(emptyRoot);

  ai::StemSeparator withPack(packRoot);
  ai::StemSeparator withoutPack(emptyRoot);
  REQUIRE(withPack.isTensorModelAvailable());
  REQUIRE_FALSE(withoutPack.isTensorModelAvailable());
  // The legacy path never adopts a tensor pack's model file.
  REQUIRE_FALSE(withPack.isModelAvailable());

  const auto baseline = withoutPack.separate(mixPath, tempRoot / "out_baseline");
  const auto flagOff = withPack.separate(mixPath, tempRoot / "out_off");
  REQUIRE(baseline.success);
  REQUIRE(flagOff.success);
  REQUIRE(flagOff.logMessage == baseline.logMessage);
  REQUIRE(flagOff.usedModel == baseline.usedModel);
  REQUIRE(flagOff.generatedFiles.size() == baseline.generatedFiles.size());
  for (std::size_t i = 0; i < baseline.generatedFiles.size(); ++i) {
    REQUIRE(flagOff.generatedFiles[i].filename() == baseline.generatedFiles[i].filename());
    REQUIRE(readBytes(flagOff.generatedFiles[i]) == readBytes(baseline.generatedFiles[i]));
  }

  ai::StemSeparator::SeparationOptions tensorOn;
  tensorOn.useTensorModel = true;
  const auto fellBack = withPack.separate(mixPath, tempRoot / "out_on", tensorOn);
  INFO(fellBack.logMessage);
  REQUIRE(fellBack.success);
  REQUIRE_FALSE(fellBack.usedModel);
  REQUIRE(fellBack.logMessage.rfind("Tensor separation unavailable (tensor model did not load", 0) == 0);
  REQUIRE(fellBack.stems.size() == baseline.stems.size());
  std::filesystem::remove_all(tempRoot);
}

#ifdef AUTOMIX_HAS_NATIVE_ORT

TEST_CASE("Tensor separation runs end to end through StemSeparator", "[ai][tensor][separator][native]") {
  const auto tempRoot = std::filesystem::temp_directory_path() / "automix_tensor_separator_e2e";
  std::filesystem::remove_all(tempRoot);
  std::filesystem::create_directories(tempRoot);
  const auto mixPath = tempRoot / "mix.wav";
  automix::util::WavWriter writer;
  writer.write(mixPath, makeTestSignal(2, 100000, 44100.0), 24);

  const auto fixture = std::filesystem::path(AUTOMIX_SOURCE_DIR) / "tests" / "fixtures" / "tensor" / "identity_mask.onnx";
  const auto packRoot = writeTensorPack(tempRoot / "pack", &fixture);

  ai::StemSeparator separator(packRoot);
  ai::StemSeparator::SeparationOptions options;
  options.useTensorModel = true;
  int lastDone = -1;
  int lastTotal = -1;
  options.tensorProgress = [&](const int done, const int total) {
    lastDone = done;
    lastTotal = total;
  };
  const auto result = separator.separate(mixPath, tempRoot / "out", options);
  INFO(result.logMessage);
  REQUIRE(result.success);
  REQUIRE(result.usedModel);
  REQUIRE(lastTotal == 1);
  REQUIRE(lastDone == lastTotal);
  REQUIRE(result.logMessage.find("'instrumental' is the residual (mix - vocals), not a second separation") !=
          std::string::npos);

  REQUIRE(result.stems.size() == 2);
  REQUIRE(result.stems[0].role == automix::domain::StemRole::Vocals);
  REQUIRE(result.stems[1].role == automix::domain::StemRole::Music);
  for (const auto& stem : result.stems) {
    REQUIRE(stem.origin == automix::domain::StemOrigin::Separated);
    REQUIRE_FALSE(stem.separationConfidence.has_value());
    REQUIRE_FALSE(stem.separationArtifactRisk.has_value());
    REQUIRE(std::filesystem::is_regular_file(stem.filePath));
  }

  // zero_dc is on in the contract, so vocals are not the mix, but the residual
  // sums back to it; the only error left is each stem's 24-bit quantization.
  automix::engine::AudioFileIO io;
  const auto mix = io.readAudioFile(mixPath);
  const auto vocals = io.readAudioFile(result.stems[0].filePath);
  const auto instrumental = io.readAudioFile(result.stems[1].filePath);
  REQUIRE(vocals.getNumSamples() == mix.getNumSamples());
  float worst = 0.0f;
  for (int ch = 0; ch < 2; ++ch) {
    for (int i = 0; i < mix.getNumSamples(); ++i) {
      worst = std::max(worst, std::abs(vocals.getSample(ch, i) + instrumental.getSample(ch, i) - mix.getSample(ch, i)));
    }
  }
  REQUIRE(worst < 1.0e-5f);
  std::filesystem::remove_all(tempRoot);
}

#endif