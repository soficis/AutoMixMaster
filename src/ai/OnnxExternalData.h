#pragma once

#include <filesystem>
#include <string>

namespace automix::ai {

struct ExternalDataInlineResult {
  bool success = false;
  int tensorsInlined = 0;
  std::string error;
};

// Rewrites an ONNX model whose graph initializers live in external-data files
// into one self-contained file, with each such initializer stored inline as
// raw_data. Everything else in the model is copied byte for byte.
//
// Why: ONNX Runtime's shape inference cannot read external initializers, so an
// export that keeps even a Split's size list external (BS-RoFormer fp32) fails
// to load at all. Inlining is only possible below protobuf's 2 GB message
// limit; larger models are refused.
//
// External locations must be relative paths inside the model's directory; a
// model naming an absolute path or one that escapes the directory is refused
// rather than having arbitrary files copied into it. The output is written to
// `outputPath` via a temporary file, so a failure never leaves a partial model.
ExternalDataInlineResult inlineExternalData(const std::filesystem::path& modelPath,
                                            const std::filesystem::path& outputPath);

} // namespace automix::ai
