#pragma once

#include <cstdint>
#include <string>

#include "ai/ModelPackLoader.h"

namespace automix::ai {

// Open-Unmix UMX-HQ vocals (sigsep/open-unmix-pytorch, MIT), ONNX export by
// MixDirective. A 36 MB bidirectional-LSTM that runs on any CPU: the fast
// alternative to BS-RoFormer on machines that cannot run that model. On an 8 GiB
// Mac, BS-RoFormer needed over 5 minutes per chunk on the CPU and CoreML/ANE
// were killed by the OS. Its graph takes the STFT magnitude only (no phase), so
// the mix phase is kept and the model's output is applied as a ratio mask.
inline constexpr const char* kUmxVocalsRepoId = "MixDirective/open-unmix-umxhq-vocals-onnx";
inline constexpr const char* kUmxVocalsFile = "model.onnx";
// Pinned so a later push to the repo cannot change what installs; bump both deliberately.
inline constexpr const char* kUmxVocalsRevision = "e06097d3193a4c3168b5ddd2a479a680b1176c16";
inline constexpr const char* kUmxVocalsSha256 = "f27742bb52b24cb039614dc76d0764e279df3eb2a9d6daa82c6daee3369ec747";
inline constexpr const char* kUmxVocalsIntendedUse =
    "Vocal separation (Open-Unmix UMX-HQ, 36 MB, CPU friendly). Lower quality than BS-RoFormer but "
    "needs little memory. The graph produces vocals only; the instrumental stem is the residual "
    "mix - vocals, not a second separation.";

// Catalog form of the pack's tensor contract. ~10 s chunks (431 frames at hop
// 1024) with ~2 s of overlap, which also gives the LSTM context at chunk edges.
// The chunk is a whole number of hops (430 x 1024): the inverse STFT returns
// (frames - 1) x hop samples, so any other length would come back short.
inline TensorContract umxVocalsCatalogContract() {
  TensorContract contract;
  contract.engine = "open_unmix";
  contract.sampleRate = 44100;
  contract.stereo = true;
  contract.chunkSamples = 440320;
  contract.overlapSamples = 87040;
  contract.stft.nFft = 4096;
  contract.stft.hopLength = 1024;
  contract.stft.winLength = 4096;
  contract.stft.window = "hann";
  contract.stft.periodicWindow = true;
  contract.stft.center = true;
  contract.stft.padMode = "reflect";
  contract.stft.normalized = false;
  contract.stft.zeroDc = false;
  contract.inputLayout = "magnitude_channels";
  contract.inputs = {{"", {1, 2, 2049, 431}, "float32"}};
  contract.outputs = {{"", {1, 2, 2049, 431}, "float32"}};
  contract.outputMode = "ratio_mask";
  contract.targetStem = "vocals";
  contract.stems = {{"vocals", ""}, {"instrumental", "vocals"}};
  return contract;
}

} // namespace automix::ai
