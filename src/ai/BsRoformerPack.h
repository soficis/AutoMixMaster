#pragma once

#include <cstdint>

#include "ai/ModelPackLoader.h"

namespace automix::ai {

// BS-RoFormer vocal separation (arXiv 2309.02612), ONNX export by xycld. MIT,
// download-only; attribution lives in NOTICE.
inline constexpr const char* kBsRoformerRepoId = "xycld/BS-RoFormer-ONNX";
// Spec D4: the catalog installs the single-file uint8-quantized build. The repo's
// alphabetically-first .onnx is the fp32 graph, which is unusable without its
// ~640 MB external-data sidecar, so the primary file is pinned by name.
inline constexpr const char* kBsRoformerQuantizedFile = "bs_roformer_ep317_sdr12.9755_quantized_uint8.onnx";
// The fp32 graph with its weights in "<name>.data". Installed instead of the
// quantized build where a GPU session opens: the quantized build's integer ops
// have no CUDA kernels (ORT inserts ~870 Memcpy nodes), so on CUDA it took
// 303-504 s for a 196 s track, while fp32 on an RTX 5060 Ti took 73-91 s
// (quantized on CPU: 687 s).
// On CPU fp32 is ~45% slower than quantized, so CPU-only machines keep that.
// The installer inlines the sidecar (inlineExternalData) because ONNX Runtime
// cannot load this export with its weights external.
inline constexpr const char* kBsRoformerFp32File = "bs_roformer_ep317_sdr12.9755.onnx";
// Device memory the fp32 build needs while separating: measured peak 9442 MiB
// above baseline (RTX 5060 Ti, CUDA 13, per-run arena shrinkage), rounded up to
// 10 GiB. Written into the pack manifest as gpu_memory_mb.
inline constexpr std::uint64_t kBsRoformerFp32GpuMemoryMb = 10240;
inline constexpr const char* kBsRoformerIntendedUse =
    "Vocal separation (BS-RoFormer). The graph produces vocals only; the instrumental stem is the "
    "residual mix - vocals, not a second separation. Quantized build: quality versus fp32 is not "
    "yet measured.";
inline constexpr const char* kBsRoformerFp32IntendedUse =
    "Vocal separation (BS-RoFormer). The graph produces vocals only; the instrumental stem is the "
    "residual mix - vocals, not a second separation. fp32 build, installed for GPU (CUDA) inference.";

// Catalog form of the pack's tensor contract (spec section 6). Tensor names are
// omitted because the repo does not publish them; the install-time probe fills
// them in, and until then checkTensorContract() matches positionally.
inline TensorContract bsRoformerCatalogContract() {
  TensorContract contract;
  contract.engine = "bs_roformer";
  contract.sampleRate = 44100;
  contract.stereo = true;
  contract.chunkSamples = 352800;
  contract.overlapSamples = 88200;
  contract.stft.nFft = 2048;
  contract.stft.hopLength = 441;
  contract.stft.winLength = 2048;
  contract.stft.window = "hann";
  contract.stft.periodicWindow = true;
  contract.stft.center = true;
  contract.stft.padMode = "reflect";
  contract.stft.normalized = false;
  contract.stft.zeroDc = true;
  contract.inputLayout = "folded_stereo";
  contract.inputs = {{"", {1, 801, 4100}, "float32"}};
  contract.outputs = {{"", {1, 1, 2050, 801, 2}, "float32"}};
  contract.outputMode = "mask";
  contract.targetStem = "vocals";
  contract.stems = {{"vocals", ""}, {"instrumental", "vocals"}};
  return contract;
}

} // namespace automix::ai
