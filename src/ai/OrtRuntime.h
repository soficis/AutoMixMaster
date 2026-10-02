#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#ifndef AUTOMIX_HAS_NATIVE_ORT
#define AUTOMIX_HAS_NATIVE_ORT 0
#endif

#ifndef AUTOMIX_HAS_EP_PLUGIN
#define AUTOMIX_HAS_EP_PLUGIN 0
#endif

#if AUTOMIX_HAS_NATIVE_ORT
#include <onnxruntime_cxx_api.h>
#endif

namespace automix::ai {

#if AUTOMIX_HAS_NATIVE_ORT

/// Process-wide ONNX Runtime environment and device discovery manager.
/// Owns the single Ort::Env instance, registers provider plugins (such as WebGPU),
/// and manages available device discovery.
class OrtRuntime {
 public:
  static OrtRuntime& instance();

  void warmUpAsync();

  Ort::Env& env();
  std::vector<std::string> availableProviders();
#if AUTOMIX_HAS_EP_PLUGIN
  const std::vector<Ort::ConstEpDevice>& webGpuDevices();
#endif
  std::string diagnostics();

 private:
  OrtRuntime();
  ~OrtRuntime() = default;

  OrtRuntime(const OrtRuntime&) = delete;
  OrtRuntime& operator=(const OrtRuntime&) = delete;

  void ensureInitialized();

  std::unique_ptr<Ort::Env> env_;
  std::vector<std::string> availableProviders_;
#if AUTOMIX_HAS_EP_PLUGIN
  std::vector<Ort::ConstEpDevice> webGpuDevices_;
#endif
  std::string diagnostics_;
  std::string initError_;
  std::once_flag initOnce_;
};

#else

class OrtRuntime {
 public:
  static OrtRuntime& instance() {
    static OrtRuntime s_instance;
    return s_instance;
  }

  void warmUpAsync() {}

  std::vector<std::string> availableProviders() { return {"cpu"}; }
  std::string diagnostics() { return "ONNX Runtime native SDK not enabled."; }

 private:
  OrtRuntime() = default;
  ~OrtRuntime() = default;
};

#endif

} // namespace automix::ai
