#pragma once

#include <cstddef>
#include <vector>

#include "engine/AudioBuffer.h"

namespace automix::analysis {

// torch.stft / torch.istft conventions, named explicitly because each one
// silently changes the spectrum if misread. The window is always a periodic
// Hann (torch.hann_window default); padding under `center` is always reflect.
struct StftParams {
  int nFft = 2048;
  int hopLength = 441;
  int winLength = 2048;
  // One flag, two opposite operations: analyze() reflect-pads nFft/2 samples at
  // each end, synthesize() drops nFft/2 samples at each end.
  bool center = true;
  // torch.stft(normalized=True) scales the forward transform by 1/sqrt(nFft).
  bool normalized = false;
  // Applied by complexMultiply(): zero bin 0 of the masked spectrum before
  // synthesis. Carried here because it is a property of how a mask is applied.
  bool zeroDc = true;
};

// Complex spectrum stored as two float planes, laid out [channel][bin][frame]
// (torch's (C, F, T) order). Keeps what StemAnalyzer discards: phase and
// channel identity.
struct Spectrogram {
  int channels = 0;
  int freqBins = 0;
  int frames = 0;
  double sampleRate = 44100.0;
  std::vector<float> real;
  std::vector<float> imag;

  [[nodiscard]] std::size_t index(int channel, int bin, int frame) const {
    return (static_cast<std::size_t>(channel) * static_cast<std::size_t>(freqBins) + static_cast<std::size_t>(bin)) *
               static_cast<std::size_t>(frames) +
           static_cast<std::size_t>(frame);
  }
};

// Number of frames analyze() produces for `samples` input samples.
int stftFrameCount(int samples, const StftParams& params);

// Exact synthesize() output length: hop * (frames - 1) under `center`.
int istftLength(int frames, const StftParams& params);

// Throws std::invalid_argument for parameters outside the implemented set and
// for inputs too short to reflect-pad (torch raises in the same case).
Spectrogram analyze(const engine::AudioBuffer& audio, const StftParams& params);
engine::AudioBuffer synthesize(const Spectrogram& spectrogram, const StftParams& params);

// Mask-mode application. Throws std::invalid_argument naming both shapes when
// the mask's geometry does not match the signal exactly: a plausible but wrong
// mask shape would otherwise yield numerically valid garbage.
Spectrogram complexMultiply(const Spectrogram& signal, const Spectrogram& mask, bool zeroDc);

} // namespace automix::analysis
