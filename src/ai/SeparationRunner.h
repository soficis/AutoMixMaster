#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "ai/ITensorInference.h"
#include "analysis/SpectrogramFrontEnd.h"
#include "engine/AudioBuffer.h"

namespace automix::ai {

// How stereo spectra are laid out in the graph's input and output tensors.
//
//   FoldedStereo  (ZFTurbo / upstream BS-RoFormer, 'b s f t c -> b (f s) t c'):
//     input  [1, T, F*C*2]      element (t, (f*C + ch)*2 + reim)
//     output [1, N, F*C, T, 2]  element (n, f*C + ch, t, reim)
//   SplitChannels (channels on their own axis):
//     input  [1, C, F, T, 2]
//     output [1, N, C, F, T, 2]
//
// N is the stem axis: 1 in Mask mode, one entry per graph-produced stem in
// Direct mode. The trailing axis of length 2 is always (real, imag).
enum class InputLayout { FoldedStereo, SplitChannels };

// Mask: the graph returns one complex mask for the target stem; the caller
// multiplies it against the input spectrum. Direct: the graph returns one
// spectrogram per stem.
enum class OutputMode { Direct, Mask };

std::optional<InputLayout> inputLayoutFromString(const std::string& value);
std::optional<OutputMode> outputModeFromString(const std::string& value);

std::vector<int64_t> tensorInputDims(InputLayout layout, int channels, int freqBins, int frames);
std::vector<int64_t> tensorOutputDims(InputLayout layout, int stemAxis, int channels, int freqBins, int frames);

struct SeparationStem {
  std::string name;
  // Empty for a stem the graph produces. Otherwise the stem is computed
  // host-side as (input mix - residualOf), e.g. instrumental = mix - vocals.
  std::string residualOf;
};

struct RunnerConfig {
  int chunkSamples = 352800;
  int overlapSamples = 88200;  // chunkSamples / 4
  double sampleRate = 44100.0;
  int channels = 2;
  analysis::StftParams stft{};  // stft.zeroDc is applied in Mask mode
  InputLayout inputLayout = InputLayout::FoldedStereo;
  OutputMode outputMode = OutputMode::Mask;
  std::string targetStem = "vocals";
  std::vector<SeparationStem> stems{{"vocals", ""}, {"instrumental", "vocals"}};
  // JUCE-free on purpose: the runner stays testable with no message loop; the
  // controller that adapts it to the UI owns the SafePointer marshalling.
  std::function<void(int done, int total)> progressCallback;
  // Polled before every chunk and, where the backend supports it, during
  // inference (from a watcher thread, so it must be thread-safe). Returning
  // true discards all work and yields Result::cancelled.
  std::function<bool()> cancelRequested;
};

// Empty when the configuration is runnable; otherwise the reason it is not.
std::string validateRunnerConfig(const RunnerConfig& config);

class SeparationRunner final {
 public:
  struct Result {
    bool usedModel = false;
    // The caller asked to stop. Not a failure: callers must not fall back to
    // another separator, because the user wants no separation at all.
    bool cancelled = false;
    std::vector<engine::AudioBuffer> stemAudio;
    std::vector<std::string> stemNames;
    std::string logMessage;
  };

  // Chunks the track, runs every chunk through `inference`, and crossfades the
  // chunks back together. All-or-nothing: any chunk failure discards every stem
  // and returns usedModel == false with the reason.
  static Result separate(const engine::AudioBuffer& mix, const ITensorInference& inference, const RunnerConfig& config);

  // Number of chunks separate() will run for a track of `samples` samples.
  static int chunkCount(int samples, const RunnerConfig& config);
};

} // namespace automix::ai
