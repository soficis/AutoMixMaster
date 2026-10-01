#include "ai/GpuRuntimePack.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <memory>
#include <mutex>
#include <system_error>

#include <juce_core/juce_core.h>
#include <nlohmann/json.hpp>

#include "util/Sha256.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi.h>
#endif

namespace automix::ai::GpuRuntimePack {
namespace {

constexpr const char* kMarkerFile = "runtime.json";
constexpr const char* kPendingRemovalFile = "remove-pending";
constexpr std::uint32_t kNvidiaVendorId = 0x10DE;

std::filesystem::path binDirectory(const std::filesystem::path& root) { return root / "bin"; }

bool endsWithDll(std::string name) {
  std::transform(name.begin(), name.end(), name.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return name.size() > 4 && name.compare(name.size() - 4, 4, ".dll") == 0;
}

std::optional<std::vector<std::string>> markerLibraries(const std::filesystem::path& root) {
  try {
    std::ifstream in(root / kMarkerFile);
    if (!in.is_open()) {
      return std::nullopt;
    }
    const auto marker = nlohmann::json::parse(in);
    if (marker.value("version", "") != version() || !marker.contains("libraries")) {
      return std::nullopt;
    }
    return marker.at("libraries").get<std::vector<std::string>>();
  } catch (...) {
    return std::nullopt;
  }
}

// cudart first: everything else depends on it. The rest load with their own
// directory searched for dependencies, so their order does not matter.
int loadRank(const std::string& library) { return library.rfind("cudart", 0) == 0 ? 0 : 1; }

} // namespace

std::string version() { return "cuda13.4-cudnn9.27-1"; }

const std::vector<Archive>& archives() {
#if defined(_WIN32)
  // NVIDIA redistributables (CUDA EULA / cuDNN SLA), as published by NVIDIA on
  // PyPI. Pinned: a changed file fails its hash instead of being installed.
  static const std::vector<Archive> list = {
      {"nvidia_cuda_runtime-13.4.92-py3-none-win_amd64.whl",
       "https://files.pythonhosted.org/packages/86/00/d5436004268f049214193659ebc36550b5ef3925c3d13b4cc980e13be6f5/"
       "nvidia_cuda_runtime-13.4.92-py3-none-win_amd64.whl",
       "08dca5e4aba480c2fd5b55075c0fa71b84ef9dcf0521f2d58baa14a803a7311c", 2778543},
      {"nvidia_cublas-13.8.0.4-py3-none-win_amd64.whl",
       "https://files.pythonhosted.org/packages/a3/df/f1246959833e2c437db8be3e5b477f66b87f8817821ed40de6c7561c9a36/"
       "nvidia_cublas-13.8.0.4-py3-none-win_amd64.whl",
       "8c5494423bb8a46822cb6b0cb95d7fa4be2d7b96a31155dff083839ec8297910", 423266897},
      {"nvidia_cufft-12.4.0.43-py3-none-win_amd64.whl",
       "https://files.pythonhosted.org/packages/7b/cc/be7fe31058127336a66c88414a2ecc6beecf680baf85994a43eaf292b202/"
       "nvidia_cufft-12.4.0.43-py3-none-win_amd64.whl",
       "4ff7075f2d0b5f69291f70938d37a86ec632cbe5747184c74ba1f50f17accacc", 160953139},
      {"nvidia_cudnn_cu13-9.27.0.42-py3-none-win_amd64.whl",
       "https://files.pythonhosted.org/packages/87/6a/e55ff0ac26a5c6e2b21f41c9d04ad096b4ed6da593fba7e25845c61b0532/"
       "nvidia_cudnn_cu13-9.27.0.42-py3-none-win_amd64.whl",
       "7d96f634adafd55c72231eb0500ca77ab109ec8ebff7b33000b76e081bc4558e", 436469905},
  };
#else
  static const std::vector<Archive> list;  // Windows only for now
#endif
  return list;
}

std::uint64_t downloadBytes() {
  std::uint64_t total = 0;
  for (const auto& archive : archives()) {
    total += archive.bytes;
  }
  return total;
}

std::filesystem::path defaultRoot() {
#if defined(_WIN32)
  const auto base = juce::File::getSpecialLocation(juce::File::windowsLocalAppData);
#else
  const auto base = juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory);
#endif
  return std::filesystem::path(base.getFullPathName().toWideCharPointer()) / "AutoMixMaster" / "gpu-runtime" /
         version();
}

bool isInstalled(const std::filesystem::path& root) {
  const auto libraries = markerLibraries(root);
  if (!libraries.has_value() || libraries->empty()) {
    return false;
  }
  std::error_code error;
  for (const auto& library : *libraries) {
    if (!std::filesystem::is_regular_file(binDirectory(root) / library, error) || error) {
      return false;
    }
  }
  return true;
}

Fetcher httpFetcher() {
  return [](const std::string& url, const std::filesystem::path& destination,
            const std::function<bool(std::uint64_t)>& progress) -> std::string {
    int statusCode = 0;
    const auto options = juce::URL::InputStreamOptions(juce::URL::ParameterHandling::inAddress)
                             .withConnectionTimeoutMs(60000)
                             .withNumRedirectsToFollow(8)
                             .withStatusCode(&statusCode);
    const auto input = juce::URL(url).createInputStream(options);
    if (input == nullptr) {
      return "could not connect to " + url;
    }
    if (statusCode >= 400) {
      return "HTTP " + std::to_string(statusCode) + " from " + url;
    }
    std::ofstream out(destination, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
      return "cannot write " + destination.string();
    }
    std::vector<char> buffer(1 << 20);
    std::uint64_t written = 0;
    for (;;) {
      const auto read = input->read(buffer.data(), static_cast<int>(buffer.size()));
      if (read <= 0) {
        break;
      }
      out.write(buffer.data(), read);
      if (!out) {
        return "write failed for " + destination.string();
      }
      written += static_cast<std::uint64_t>(read);
      if (progress && !progress(written)) {
        return "cancelled";
      }
    }
    return {};
  };
}

std::string extractLibraries(const std::filesystem::path& wheel,
                             const std::filesystem::path& binDir,
                             std::vector<std::string>& extracted) {
  juce::ZipFile zip(juce::File(juce::String(wheel.wstring().c_str())));
  if (zip.getNumEntries() == 0) {
    return "'" + wheel.filename().string() + "' is not a readable archive";
  }
  std::error_code error;
  std::filesystem::create_directories(binDir, error);
  for (int i = 0; i < zip.getNumEntries(); ++i) {
    const auto* entry = zip.getEntry(i);
    const auto name = entry->filename.replaceCharacter('\\', '/').toStdString();
    const std::filesystem::path entryPath(name);
    for (const auto& part : entryPath) {
      if (part == "..") {
        return "archive entry '" + name + "' escapes the archive";
      }
    }
    if (entryPath.is_absolute() || entryPath.has_root_name() || !endsWithDll(name)) {
      continue;
    }
    const auto fileName = entryPath.filename();
    const auto& unused = unusedLibraries();
    if (std::find(unused.begin(), unused.end(), fileName.string()) != unused.end()) {
      continue;
    }
    std::unique_ptr<juce::InputStream> in(zip.createStreamForEntry(i));
    if (in == nullptr) {
      return "cannot read '" + name + "' from " + wheel.filename().string();
    }
    std::ofstream out(binDir / fileName, std::ios::binary | std::ios::trunc);
    std::vector<char> buffer(1 << 20);
    for (;;) {
      const auto read = in->read(buffer.data(), static_cast<int>(buffer.size()));
      if (read <= 0) {
        break;
      }
      out.write(buffer.data(), read);
    }
    if (!out) {
      return "cannot write " + (binDir / fileName).string();
    }
    extracted.push_back(fileName.string());
  }
  return {};
}

InstallResult install(const std::filesystem::path& root,
                      const std::function<bool(std::uint64_t, std::uint64_t)>& progress,
                      const Fetcher& fetch) {
  InstallResult result;
  if (archives().empty()) {
    result.message = "The GPU runtime pack is not available on this platform.";
    return result;
  }
  std::error_code error;
  const auto downloads = root / "downloads";
  std::filesystem::create_directories(downloads, error);
  std::filesystem::remove(root / kMarkerFile, error);  // invalid until this install completes

  const auto total = downloadBytes();
  std::uint64_t done = 0;
  std::vector<std::string> libraries;
  for (const auto& archive : archives()) {
    const auto wheel = downloads / archive.name;
    const bool alreadyHere = std::filesystem::is_regular_file(wheel, error) && util::fileSha256(wheel) == archive.sha256;
    if (!alreadyHere) {
      const auto base = done;
      const auto failure = fetch(archive.url, wheel, [&](std::uint64_t bytes) {
        return !progress || progress(base + bytes, total);
      });
      if (failure == "cancelled") {
        result.cancelled = true;
        result.message = "GPU runtime download cancelled.";
        return result;
      }
      if (!failure.empty()) {
        result.message = "Downloading " + archive.name + " failed: " + failure;
        return result;
      }
      if (const auto actual = util::fileSha256(wheel); actual != archive.sha256) {
        std::filesystem::remove(wheel, error);
        result.message = "SHA-256 mismatch for " + archive.name + " (expected " + archive.sha256 + ", got " +
                         (actual.empty() ? "unreadable" : actual) + "); the download was discarded.";
        return result;
      }
    }
    done += archive.bytes;
    if (progress && !progress(done, total)) {
      result.cancelled = true;
      result.message = "GPU runtime download cancelled.";
      return result;
    }
    if (const auto failure = extractLibraries(wheel, binDirectory(root), libraries); !failure.empty()) {
      result.message = "Unpacking " + archive.name + " failed: " + failure;
      return result;
    }
    std::filesystem::remove(wheel, error);
  }
  std::filesystem::remove(downloads, error);

  std::sort(libraries.begin(), libraries.end());
  libraries.erase(std::unique(libraries.begin(), libraries.end()), libraries.end());
  const nlohmann::json marker = {{"version", version()}, {"libraries", libraries}};
  {
    std::ofstream out(root / kMarkerFile, std::ios::trunc);
    out << marker.dump(2);
    if (!out) {
      result.message = "Cannot write the GPU runtime marker in " + root.string();
      return result;
    }
  }
  result.success = true;
  result.message = "GPU runtime installed (" + std::to_string(libraries.size()) + " libraries).";
  return result;
}

const std::vector<std::string>& unusedLibraries() {
  static const std::vector<std::string> list = {"cufftw64_12.dll", "nvblas64_13.dll"};
  return list;
}

namespace {

// Only ever delete something that is recognisably ours: the versioned folder
// holding a bin/ directory, a marker or a pending-removal note.
bool looksLikePack(const std::filesystem::path& root) {
  std::error_code error;
  return root.filename() == version() &&
         (std::filesystem::is_directory(root / "bin", error) || std::filesystem::exists(root / kMarkerFile, error) ||
          std::filesystem::exists(root / kPendingRemovalFile, error));
}

} // namespace

RemoveResult uninstall(const std::filesystem::path& root) {
  RemoveResult result;
  std::error_code error;
  if (!std::filesystem::exists(root, error)) {
    result.removedNow = true;
    result.message = "The GPU runtime is not installed.";
    return result;
  }
  if (!looksLikePack(root)) {
    result.message = "Refusing to delete '" + root.string() + "': it does not look like a GPU runtime pack.";
    return result;
  }
  std::filesystem::remove(root / kMarkerFile, error);  // first: nothing may preload a half-deleted pack
  std::filesystem::remove_all(root, error);
  if (!error && !std::filesystem::exists(root, error)) {
    result.removedNow = true;
    result.message = "GPU runtime removed.";
    return result;
  }
  // Libraries loaded by this process are locked on Windows.
  std::ofstream(root / kPendingRemovalFile) << "remove at next start\n";
  result.message = "GPU runtime disabled; its remaining files are removed the next time AutoMixMaster starts.";
  return result;
}

void completePendingRemoval(const std::filesystem::path& root) {
  std::error_code error;
  if (std::filesystem::exists(root / kPendingRemovalFile, error) && looksLikePack(root)) {
    std::filesystem::remove_all(root, error);
  }
}
bool preload(const std::filesystem::path& root) {
#if defined(_WIN32)
  static std::mutex mutex;
  static bool loaded = false;
  const std::scoped_lock lock(mutex);
  if (loaded) {
    return true;
  }
  if (!isInstalled(root)) {
    return false;
  }
  auto libraries = *markerLibraries(root);
  std::stable_sort(libraries.begin(), libraries.end(),
                   [](const std::string& a, const std::string& b) { return loadRank(a) < loadRank(b); });
  for (const auto& library : libraries) {
    // With an absolute path, LOAD_WITH_ALTERED_SEARCH_PATH resolves the
    // library's own dependencies from its directory first. Modules stay
    // loaded for the life of the process, which is what the provider needs.
    const auto path = binDirectory(root) / library;
    if (LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH) == nullptr &&
        library.rfind("cudart", 0) == 0) {
      return false;
    }
  }
  loaded = true;
  return true;
#else
  (void)root;
  return false;
#endif
}

std::pair<int, int> nvidiaDriverVersion(const std::uint64_t userModeDriverVersion) {
  const auto c = static_cast<int>((userModeDriverVersion >> 16) & 0xFFFF);
  const auto d = static_cast<int>(userModeDriverVersion & 0xFFFF);
  const int combined = (c % 10) * 10000 + d;  // the last five digits
  return {combined / 100, combined % 100};
}

std::optional<NvidiaAdapter> largestNvidiaAdapter() {
#if defined(_WIN32)
  IDXGIFactory1* factory = nullptr;
  if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory))) || factory == nullptr) {
    return std::nullopt;
  }
  std::optional<NvidiaAdapter> largest;
  IDXGIAdapter1* adapter = nullptr;
  for (UINT index = 0; factory->EnumAdapters1(index, &adapter) != DXGI_ERROR_NOT_FOUND; ++index) {
    DXGI_ADAPTER_DESC1 desc{};
    if (SUCCEEDED(adapter->GetDesc1(&desc)) && desc.VendorId == kNvidiaVendorId &&
        (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0) {
      NvidiaAdapter found;
      found.dedicatedBytes = static_cast<std::uint64_t>(desc.DedicatedVideoMemory);
      LARGE_INTEGER userModeVersion{};
      if (SUCCEEDED(adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &userModeVersion))) {
        const auto [major, minor] = nvidiaDriverVersion(static_cast<std::uint64_t>(userModeVersion.QuadPart));
        found.driverMajor = major;
        found.driverMinor = minor;
      }
      if (!largest.has_value() || found.dedicatedBytes > largest->dedicatedBytes) {
        largest = found;
      }
    }
    adapter->Release();
  }
  factory->Release();
  return largest;
#else
  return std::nullopt;
#endif
}

bool shouldOffer(const std::uint64_t minimumDedicatedBytes, const std::filesystem::path& root) {
  if (archives().empty() || isInstalled(root)) {
    return false;
  }
  const auto adapter = largestNvidiaAdapter();
  // An unreported driver version is not proof of an old one, but 1 GB is too
  // much to download on a guess: only offer what is known to work.
  return adapter.has_value() && adapter->dedicatedBytes >= minimumDedicatedBytes &&
         adapter->driverMajor.value_or(0) >= kMinimumDriverMajor;
}
} // namespace automix::ai::GpuRuntimePack
