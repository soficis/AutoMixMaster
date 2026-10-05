#include "analysis/SpectrogramFrontEnd.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include <juce_dsp/juce_dsp.h>

namespace automix::analysis {
namespace {

bool isPowerOfTwo(const int value) {
  return value > 0 && (value & (value - 1)) == 0;
}

int fftOrderFor(const int nFft) {
  int order = 0;
  while ((1 << order) < nFft) {
    ++order;
  }
  return order;
}

void validate(const StftParams& params) {
  if (!isPowerOfTwo(params.nFft)) {
    throw std::invalid_argument("STFT n_fft must be a power of two, got " + std::to_string(params.nFft));
  }
  if (params.hopLength <= 0) {
    throw std::invalid_argument("STFT hop_length must be positive, got " + std::to_string(params.hopLength));
  }
  if (params.winLength <= 0 || params.winLength > params.nFft) {
    throw std::invalid_argument("STFT win_length must be in (0, n_fft], got " + std::to_string(params.winLength));
  }
}

// analyze() side of `center`: samples of reflect padding added at each end.
int centrePadSamples(const StftParams& params) {
  return params.center ? params.nFft / 2 : 0;
}

// synthesize() side of `center`: samples dropped from each end of the
// overlap-added signal. Numerically equal to the pad, deliberately named apart.
int centreTrimSamples(const StftParams& params) {
  return params.center ? params.nFft / 2 : 0;
}

// Periodic Hann of winLength (torch.hann_window(periodic=True)), zero-padded
// centred into nFft as torch.stft does when win_length < n_fft. JUCE's hann is
// the symmetric form, so the periodic window of length N is the first N taps of
// the symmetric window of length N + 1.
std::vector<float> makeAnalysisWindow(const StftParams& params) {
  std::vector<float> symmetric(static_cast<std::size_t>(params.winLength) + 1, 0.0f);
  juce::dsp::WindowingFunction<float>::fillWindowingTables(
      symmetric.data(), symmetric.size(), juce::dsp::WindowingFunction<float>::hann, false);

  std::vector<float> window(static_cast<std::size_t>(params.nFft), 0.0f);
  const auto offset = static_cast<std::size_t>((params.nFft - params.winLength) / 2);
  for (std::size_t i = 0; i < static_cast<std::size_t>(params.winLength); ++i) {
    window[offset + i] = symmetric[i];
  }
  return window;
}

} // namespace

int stftFrameCount(const int samples, const StftParams& params) {
  if (params.center) {
    return 1 + samples / params.hopLength;
  }
  if (samples < params.nFft) {
    return 0;
  }
  return 1 + (samples - params.nFft) / params.hopLength;
}

int istftLength(const int frames, const StftParams& params) {
  const int overlapAdded = params.nFft + params.hopLength * (frames - 1);
  return overlapAdded - 2 * centreTrimSamples(params);
}

Spectrogram analyze(const engine::AudioBuffer& audio, const StftParams& params) {
  validate(params);
  const int channels = audio.getNumChannels();
  const int samples = audio.getNumSamples();
  const int pad = centrePadSamples(params);
  if (params.center && samples <= pad) {
    throw std::invalid_argument("STFT reflect padding of " + std::to_string(pad) +
                                " samples needs more input than the " + std::to_string(samples) + " samples given");
  }
  const int frames = stftFrameCount(samples, params);
  if (channels <= 0 || frames <= 0) {
    throw std::invalid_argument("STFT input has no complete frame (" + std::to_string(channels) + " channels, " +
                                std::to_string(samples) + " samples)");
  }

  Spectrogram spectrogram;
  spectrogram.channels = channels;
  spectrogram.freqBins = params.nFft / 2 + 1;
  spectrogram.frames = frames;
  spectrogram.sampleRate = audio.getSampleRate();
  const std::size_t planeSize =
      static_cast<std::size_t>(channels) * static_cast<std::size_t>(spectrogram.freqBins) * static_cast<std::size_t>(frames);
  spectrogram.real.assign(planeSize, 0.0f);
  spectrogram.imag.assign(planeSize, 0.0f);

  const auto window = makeAnalysisWindow(params);
  juce::dsp::FFT fft(fftOrderFor(params.nFft));
  const float scale = params.normalized ? 1.0f / std::sqrt(static_cast<float>(params.nFft)) : 1.0f;

  const int paddedLength = samples + 2 * pad;
  std::vector<float> padded(static_cast<std::size_t>(paddedLength), 0.0f);
  std::vector<float> fftData(static_cast<std::size_t>(params.nFft) * 2, 0.0f);

  for (int ch = 0; ch < channels; ++ch) {
    const float* source = audio.getReadPointer(ch);
    for (int i = 0; i < samples; ++i) {
      padded[static_cast<std::size_t>(pad + i)] = source[i];
    }
    // torch pad_mode='reflect': mirror about the edge sample, edge excluded.
    for (int j = 0; j < pad; ++j) {
      padded[static_cast<std::size_t>(pad - 1 - j)] = source[j + 1];
      padded[static_cast<std::size_t>(pad + samples + j)] = source[samples - 2 - j];
    }

    for (int frame = 0; frame < frames; ++frame) {
      const std::size_t start = static_cast<std::size_t>(frame) * static_cast<std::size_t>(params.hopLength);
      std::fill(fftData.begin(), fftData.end(), 0.0f);
      for (std::size_t n = 0; n < static_cast<std::size_t>(params.nFft); ++n) {
        fftData[n] = padded[start + n] * window[n];
      }
      fft.performRealOnlyForwardTransform(fftData.data(), true);
      for (int bin = 0; bin < spectrogram.freqBins; ++bin) {
        const auto index = spectrogram.index(ch, bin, frame);
        spectrogram.real[index] = fftData[static_cast<std::size_t>(2 * bin)] * scale;
        spectrogram.imag[index] = fftData[static_cast<std::size_t>(2 * bin + 1)] * scale;
      }
    }
  }
  return spectrogram;
}

engine::AudioBuffer synthesize(const Spectrogram& spectrogram, const StftParams& params) {
  validate(params);
  if (spectrogram.freqBins != params.nFft / 2 + 1) {
    throw std::invalid_argument("iSTFT expects " + std::to_string(params.nFft / 2 + 1) + " frequency bins for n_fft " +
                                std::to_string(params.nFft) + ", got " + std::to_string(spectrogram.freqBins));
  }
  const std::size_t planeSize = static_cast<std::size_t>(spectrogram.channels) *
                                static_cast<std::size_t>(spectrogram.freqBins) *
                                static_cast<std::size_t>(spectrogram.frames);
  if (spectrogram.channels <= 0 || spectrogram.frames <= 0 || spectrogram.real.size() != planeSize ||
      spectrogram.imag.size() != planeSize) {
    throw std::invalid_argument("iSTFT spectrogram planes do not match their declared geometry");
  }

  const int frames = spectrogram.frames;
  const int nyquist = params.nFft / 2;
  const int overlapAddedLength = params.nFft + params.hopLength * (frames - 1);
  const int trim = centreTrimSamples(params);
  const int outputLength = istftLength(frames, params);
  if (outputLength <= 0) {
    throw std::invalid_argument("iSTFT of " + std::to_string(frames) + " frames yields no samples");
  }

  const auto window = makeAnalysisWindow(params);
  juce::dsp::FFT fft(fftOrderFor(params.nFft));
  const float scale = params.normalized ? std::sqrt(static_cast<float>(params.nFft)) : 1.0f;

  // Window-squared envelope: frames overlap heavily (hop < win), so the
  // overlap-added signal is divided by sum(w^2) at each sample, as torch.istft
  // does. Identical for every channel.
  std::vector<double> envelope(static_cast<std::size_t>(overlapAddedLength), 0.0);
  for (int frame = 0; frame < frames; ++frame) {
    const std::size_t start = static_cast<std::size_t>(frame) * static_cast<std::size_t>(params.hopLength);
    for (std::size_t n = 0; n < static_cast<std::size_t>(params.nFft); ++n) {
      envelope[start + n] += static_cast<double>(window[n]) * static_cast<double>(window[n]);
    }
  }

  engine::AudioBuffer output(spectrogram.channels, outputLength, spectrogram.sampleRate);
  std::vector<double> accumulator(static_cast<std::size_t>(overlapAddedLength), 0.0);
  std::vector<float> fftData(static_cast<std::size_t>(params.nFft) * 2, 0.0f);

  for (int ch = 0; ch < spectrogram.channels; ++ch) {
    std::fill(accumulator.begin(), accumulator.end(), 0.0);
    for (int frame = 0; frame < frames; ++frame) {
      std::fill(fftData.begin(), fftData.end(), 0.0f);
      for (int bin = 0; bin <= nyquist; ++bin) {
        const auto index = spectrogram.index(ch, bin, frame);
        fftData[static_cast<std::size_t>(2 * bin)] = spectrogram.real[index] * scale;
        // irfft semantics: the imaginary parts of the DC and Nyquist bins of a
        // real signal's spectrum are discarded, not folded into the output.
        const bool realOnlyBin = bin == 0 || bin == nyquist;
        fftData[static_cast<std::size_t>(2 * bin + 1)] = realOnlyBin ? 0.0f : spectrogram.imag[index] * scale;
      }
      fft.performRealOnlyInverseTransform(fftData.data());
      const std::size_t start = static_cast<std::size_t>(frame) * static_cast<std::size_t>(params.hopLength);
      for (std::size_t n = 0; n < static_cast<std::size_t>(params.nFft); ++n) {
        accumulator[start + n] += static_cast<double>(fftData[n]) * static_cast<double>(window[n]);
      }
    }

    float* destination = output.getWritePointer(ch);
    for (int i = 0; i < outputLength; ++i) {
      const auto source = static_cast<std::size_t>(i + trim);
      const double norm = envelope[source];
      destination[i] = norm > 1.0e-11 ? static_cast<float>(accumulator[source] / norm) : 0.0f;
    }
  }
  return output;
}

Spectrogram complexMultiply(const Spectrogram& signal, const Spectrogram& mask, const bool zeroDc) {
  const auto describe = [](const Spectrogram& s) {
    return "[" + std::to_string(s.channels) + " ch, " + std::to_string(s.freqBins) + " bins, " +
           std::to_string(s.frames) + " frames, 2 planes]";
  };
  const auto planeSize = [](const Spectrogram& s) {
    return static_cast<std::size_t>(s.channels) * static_cast<std::size_t>(s.freqBins) *
           static_cast<std::size_t>(s.frames);
  };
  if (signal.channels != mask.channels || signal.freqBins != mask.freqBins || signal.frames != mask.frames) {
    throw std::invalid_argument("mask shape " + describe(mask) + " does not match signal shape " + describe(signal));
  }
  if (signal.real.size() != planeSize(signal) || signal.imag.size() != planeSize(signal) ||
      mask.real.size() != planeSize(mask) || mask.imag.size() != planeSize(mask)) {
    throw std::invalid_argument("real/imag planes do not match the declared shape " + describe(signal));
  }

  Spectrogram product = signal;
  for (std::size_t i = 0; i < product.real.size(); ++i) {
    const float a = signal.real[i];
    const float b = signal.imag[i];
    const float c = mask.real[i];
    const float d = mask.imag[i];
    product.real[i] = a * c - b * d;
    product.imag[i] = a * d + b * c;
  }
  if (zeroDc) {
    for (int ch = 0; ch < product.channels; ++ch) {
      for (int frame = 0; frame < product.frames; ++frame) {
        const auto index = product.index(ch, 0, frame);
        product.real[index] = 0.0f;
        product.imag[index] = 0.0f;
      }
    }
  }
  return product;
}

} // namespace automix::analysis
