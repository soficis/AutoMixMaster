#include "ai/OrtRuntime.h"

#if AUTOMIX_HAS_NATIVE_ORT

#include <algorithm>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "ai/GpuProvider.h"

namespace automix::ai {
namespace {

std::filesystem::path resolveOrtCoreLibraryPath() {
#if defined(_WIN32)
  HMODULE hModule = nullptr;
  if (GetModuleHandleExW(
          GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
          reinterpret_cast<LPCWSTR>(&OrtGetApiBase),
          &hModule) && hModule != nullptr) {
    std::vector<wchar_t> path(32768, L'\0');
    const DWORD len = GetModuleFileNameW(hModule, path.data(), static_cast<DWORD>(path.size()));
    if (len > 0 && len < path.size()) {
      return std::filesystem::path(path.data());
    }
  }
#else
  Dl_info info;
  if (dladdr(reinterpret_cast<const void*>(&OrtGetApiBase), &info) && info.dli_fname != nullptr) {
    return std::filesystem::path(info.dli_fname);
  }
#endif
  return {};
}

std::filesystem::path resolveWebGpuPluginPath() {
  const auto corePath = resolveOrtCoreLibraryPath();
  if (corePath.empty()) {
    return {};
  }
#if defined(_WIN32)
  const auto candidate = corePath.parent_path() / "onnxruntime_providers_webgpu.dll";
  if (std::filesystem::exists(candidate)) {
    return candidate;
  }
#elif defined(__linux__)
  const auto candidate = corePath.parent_path() / "libonnxruntime_providers_webgpu.so";
  if (std::filesystem::exists(candidate)) {
    return candidate;
  }
#endif
  return {};
}

} // namespace

OrtRuntime& OrtRuntime::instance() {
  // Intentionally leaked to prevent Ort::Env being destroyed during static teardown
  // while other statics might still hold Ort::Session instances.
  static OrtRuntime* s_instance = new OrtRuntime();
  return *s_instance;
}

OrtRuntime::OrtRuntime() = default;

void OrtRuntime::warmUpAsync() {
  std::thread([this]() {
    ensureInitialized();
  }).detach();
}

void OrtRuntime::ensureInitialized() {
  std::call_once(initOnce_, [this]() {
    try {
      env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "AutoMixMasterGlobal");
    } catch (const std::exception& e) {
      initError_ = "Ort::Env initialization failed: " + std::string(e.what());
      diagnostics_ = initError_;
      availableProviders_ = {gpu::kProviderCpu};
      return;
    } catch (...) {
      initError_ = "Ort::Env initialization failed with unknown exception";
      diagnostics_ = initError_;
      availableProviders_ = {gpu::kProviderCpu};
      return;
    }

    std::vector<std::string> rawProviders;
    try {
      rawProviders = Ort::GetAvailableProviders();
    } catch (...) {
    }

    for (const auto& p : rawProviders) {
      availableProviders_.push_back(gpu::canonicalProviderName(p));
    }

    // Attempt WebGPU plugin registration on Windows / Linux
    const auto pluginPath = resolveWebGpuPluginPath();
    const auto ortVersion = gpu::parseOrtVersion(Ort::GetVersionString());
    const bool compiledIn = (AUTOMIX_HAS_EP_PLUGIN != 0);

    const auto u8Plugin = pluginPath.u8string();
    const std::string pluginPathStr(u8Plugin.begin(), u8Plugin.end());
    const auto decision = gpu::decidePluginEpAttempt(compiledIn, ortVersion, "webgpu", pluginPathStr);

    diagnostics_ = "ORT version: " + std::string(Ort::GetVersionString());

#if AUTOMIX_HAS_EP_PLUGIN
    if (decision.attempt) {
      try {
        env_->RegisterExecutionProviderLibrary("webgpu_ep", pluginPath.c_str());
        diagnostics_ += "; WebGPU plugin registered from " + pluginPathStr;

        auto devices = env_->GetEpDevices();
        for (const auto& dev : devices) {
          const std::string name = dev.EpName() ? dev.EpName() : "";
          if (name.find("WebGpu") != std::string::npos || name.find("webgpu") != std::string::npos) {
            webGpuDevices_.push_back(dev);
          }
        }

        if (!webGpuDevices_.empty()) {
          // TODO(measure): hybrid GPUs - confirm the WebGPU EP's first device is the discrete adapter (plan Phase F)
          const auto& chosen = webGpuDevices_[0];
          const char* vendorStr = chosen.Device().Vendor() ? chosen.Device().Vendor() : "unknown";
          const auto devId = chosen.Device().DeviceId();
          const auto devType = static_cast<int>(chosen.Device().Type());
          diagnostics_ += "; " + std::to_string(webGpuDevices_.size()) + " WebGPU device(s) found (chosen: vendor=" +
                          vendorStr + ", deviceId=" + std::to_string(devId) + ", type=" + std::to_string(devType) + ")";
          if (std::find(availableProviders_.begin(), availableProviders_.end(), gpu::kProviderWebGpu) ==
              availableProviders_.end()) {
            availableProviders_.push_back(gpu::kProviderWebGpu);
          }
        } else {
          diagnostics_ += "; no WebGPU device discovered (adapter unsupported or headless)";
        }
      } catch (const std::exception& e) {
        diagnostics_ += "; WebGPU registration failed (" + std::string(e.what()) + ")";
      }
    } else {
      diagnostics_ += "; WebGPU plugin registration skipped: " + decision.reason;
    }
#else
    diagnostics_ += "; WebGPU plugin: not compiled in (ONNX Runtime headers lack the plugin-EP API)";
#endif

    std::sort(availableProviders_.begin(), availableProviders_.end());
    availableProviders_.erase(
        std::unique(availableProviders_.begin(), availableProviders_.end()),
        availableProviders_.end());
    std::stable_sort(availableProviders_.begin(), availableProviders_.end(),
                     [](const std::string& a, const std::string& b) {
                       return gpu::providerPriority(a) < gpu::providerPriority(b);
                     });
  });
}

Ort::Env& OrtRuntime::env() {
  ensureInitialized();
  if (!env_) {
    throw std::runtime_error(initError_.empty() ? "Ort::Env not initialized" : initError_);
  }
  return *env_;
}

std::vector<std::string> OrtRuntime::availableProviders() {
  ensureInitialized();
  return availableProviders_;
}

#if AUTOMIX_HAS_EP_PLUGIN
const std::vector<Ort::ConstEpDevice>& OrtRuntime::webGpuDevices() {
  ensureInitialized();
  return webGpuDevices_;
}
#endif

std::string OrtRuntime::diagnostics() {
  ensureInitialized();
  return diagnostics_;
}

} // namespace automix::ai

#endif // AUTOMIX_HAS_NATIVE_ORT

