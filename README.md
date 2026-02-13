# AutoMixMaster

AutoMixMaster is a JUCE/CMake desktop app for offline stem mixing and mastering with deterministic plans:

`Analysis → MixPlan → MasterPlan → Renderer → Audio + JSON report`

It supports lossy/lossless export, batch acceleration, model-pack auto discovery/install, limiter-pack discovery/install, and cancellation for long renders. Builds and runs on **Windows**, **Linux** (including WSL), and **macOS**.

---

## How AutoMixMaster Works

AutoMixMaster takes multi-track stems (individual instrument/vocal recordings) and produces a mixed and mastered audio file through a deterministic pipeline. Understanding each stage and its settings helps you control what the output sounds like.

### Pipeline Overview

```
Import Stems → Analysis → Auto Mix → Auto Master → Render → Export
```

1. **Import**: Load stems (`.wav`, `.aiff`, `.flac`, `.mp3`, `.ogg`). Optionally load an original mix reference for comparison.
2. **Analysis** (`StemAnalyzer`): Each stem is analyzed for loudness (LUFS), spectral energy (low/mid/high), stereo correlation, true peak level, and dynamic range. This data drives all automated decisions.
3. **Auto Mix** (`MixPlan`): Generates per-stem mixing decisions based on analysis. Settings that affect output:
   - **Gain (dB)**: Volume level per stem. Louder stems get attenuated; quiet stems get boosted to achieve balance.
   - **Pan**: Stereo placement (-1.0 left to +1.0 right). Keeps center instruments centered, spreads others.
   - **High-pass filter (Hz)**: Removes low rumble from non-bass stems. Higher values cut more bass.
   - **Mud cut (dB)**: Reduces 200-500 Hz buildup that makes mixes sound muddy.
   - **Compressor**: Tames dynamic range per stem. Threshold, ratio, and release shape the compression character. Lower threshold or higher ratio = more compressed/consistent sound.
   - **Expander**: Reduces noise floor in quiet passages. Gentle settings clean up bleed between stems.
   - **Dry/Wet**: Blends processed and unprocessed signals. 1.0 = fully processed.
   - **Bus headroom (dB)**: Reserves headroom before mastering. More headroom = safer limiting later.
4. **Auto Master** (`MasterPlan`): Shapes the mixed bus for final delivery. Key settings:
   - **Target LUFS**: Loudness target (e.g., -14 LUFS for streaming, -23 LUFS for broadcast). Lower values = quieter but more dynamic.
   - **True peak limit (dBTP)**: Maximum inter-sample peak level. -1.0 dBTP is the standard for streaming platforms.
   - **Pre-gain (dB)**: Adjusts level going into the mastering chain. Use to push into compression/limiting harder.
   - **Limiter ceiling (dB)**: Sets the absolute maximum output level. Lower = more headroom, less loudness.
   - **Limiter attack/release/lookahead**: Shape how the limiter catches peaks. Shorter attack = catches faster but can distort transients. Longer lookahead = smoother limiting.
   - **De-esser**: Reduces sibilance (harsh "s" sounds). Strength controls how aggressively it works.
   - **De-harsh EQ**: Dynamic reduction of fatiguing frequencies (2-8 kHz range).
   - **Stereo width**: Values > 1.0 widen the image, < 1.0 narrows toward mono.
   - **Low mono (Hz)**: Sums frequencies below this cutoff to mono. Tightens bass, improves mono compatibility.
   - **Soft clipper**: Adds gentle saturation before limiting. Drive controls intensity — subtle warmth at low settings, distortion at high.
   - **Dither bit depth**: Applied when reducing bit depth (e.g., 16-bit for CD). Reduces quantization artifacts.
   - **Preset**: `DefaultStreaming` (-14 LUFS), `Broadcast` (-23 LUFS), `UdioOptimized`, or `Custom`.
5. **Render**: Applies the mix plan and master plan through the selected renderer:
   - **BuiltIn**: All DSP runs internally — most portable, no external dependencies.
   - **PhaseLimiter**: Routes through an external PhaseLimiter binary for specialized transparent limiting. Falls back to BuiltIn if binary not found.
   - **External Limiter**: Uses any user-supplied limiter binary that honors the request JSON contract. Falls back to BuiltIn on failure/timeout.
6. **Export**: Writes the final audio file plus a JSON report with measured loudness, peak levels, spectral balance, and all applied settings.

### How Settings Affect Output Quality

| Setting Area | Conservative / Safe | Aggressive / Loud |
|---|---|---|
| Target LUFS | -16 to -14 (dynamic, natural) | -10 to -8 (crushed, loud) |
| Limiter ceiling | -2.0 dB (safe headroom) | -0.5 dB (maximum loudness) |
| Compressor ratio | 2:1 (gentle glue) | 8:1+ (heavy squash) |
| Pre-gain | 0 dB (clean) | +3-6 dB (drives limiter harder) |
| Soft clipper drive | 1.0 (off) | 1.3+ (audible saturation) |
| Stereo width | 1.0 (natural) | 1.3+ (wide but risks mono issues) |
| De-esser strength | 0.2 (subtle) | 0.6+ (aggressive, may dull vocals) |

### Residual Blend

When an original mix reference is loaded, the **Residual Blend %** slider controls how much of the difference between the original mix and the re-created mix gets blended back in. This preserves reverb tails, room ambience, and other elements not captured by the stems alone. 0% = stems only, 100% = full residual mixed in.

### AI Model Override

If AI model packs are installed, they can override the heuristic decisions for stem role classification, mix planning, and mastering. The AI models are trained on analyzed features and produce alternative gain/EQ/dynamics settings. The heuristic engine remains the fallback when no models are loaded.

---

## Quickstart (Windows — Visual Studio)

Requires **Visual Studio 2019+** (with C++ Desktop workload) and **CMake 3.24+**.

All dependencies (JUCE, nlohmann/json, libebur128, Catch2) are downloaded automatically via CMake `FetchContent` — no manual installs needed.

```powershell
# Configure (generates VS solution)
cmake -S . -B build_win_vs2022 -G "Visual Studio 17 2022" -A x64 -DBUILD_TESTING=ON -DBUILD_TOOLS=ON

# Build Release
cmake --build build_win_vs2022 --config Release --parallel

# Run tests
ctest --test-dir build_win_vs2022 --build-config Release --output-on-failure -j4

# Run the app
.\build_win_vs2022\AutoMixMasterApp_artefacts\Release\AutoMixMaster.exe
```

For **NMake** (command-line only, no IDE):

```powershell
# Run from a VS Developer Command Prompt
cmake -S . -B build_win_nmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build_win_nmake --parallel
ctest --test-dir build_win_nmake --output-on-failure -j4
```

For **Ninja** (faster builds, requires Ninja on PATH):

```powershell
cmake -S . -B build_win_ninja -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DBUILD_TOOLS=ON
cmake --build build_win_ninja --parallel
```

## Quickstart (WSL / Linux)

Verified on **February 13, 2026** (Ubuntu WSL2). Install Linux dependencies first:

```bash
# JUCE Linux dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install -y build-essential cmake pkg-config \
  libasound2-dev libjack-jackd2-dev ladspa-sdk \
  libfreetype-dev libfontconfig1-dev \
  libx11-dev libxcomposite-dev libxcursor-dev libxext-dev \
  libxinerama-dev libxrandr-dev libxrender-dev \
  libglu1-mesa-dev mesa-common-dev
```

Then build:

```bash
cd /path/to/AutoMixMaster
cmake -S . -B build_release -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DBUILD_TOOLS=ON
cmake --build build_release --parallel $(nproc)
TMPDIR=/tmp ctest --test-dir build_release --output-on-failure -j4
./build_release/AutoMixMasterApp_artefacts/Release/AutoMixMaster
```

From **PowerShell on Windows** (WSL proxy):

```powershell
wsl -e bash -lc "cd /mnt/v/AutoMixMaster && cmake -S . -B build_wsl_release -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DBUILD_TOOLS=ON"
wsl -e bash -lc "cd /mnt/v/AutoMixMaster && cmake --build build_wsl_release --parallel"
wsl -e bash -lc "cd /mnt/v/AutoMixMaster && TMPDIR=/tmp ctest --test-dir build_wsl_release --output-on-failure -j4"
```

`TMPDIR=/tmp` avoids WSL temp-directory permission issues in some Windows-hosted environments.

## Quickstart (macOS) — Untested

> **Note**: These macOS instructions are provided for reference but have **not been personally tested** by the maintainer. They are based on JUCE's documented CMake requirements and standard macOS development tooling. Community feedback and corrections are welcome.

### Prerequisites

| Requirement | Minimum | Install |
|---|---|---|
| macOS | 10.15 (Catalina) | — |
| Xcode | 14+ | App Store or [developer.apple.com](https://developer.apple.com/xcode/) |
| Xcode Command Line Tools | Matching Xcode | `xcode-select --install` |
| CMake | 3.24+ | `brew install cmake` or [cmake.org](https://cmake.org/download/) |

### Build with Xcode Generator (Recommended)

```bash
# Install prerequisites
xcode-select --install
brew install cmake

cd /path/to/AutoMixMaster

# Configure — Xcode generator, Universal binary (Apple Silicon + Intel)
cmake -S . -B build_macos -G Xcode \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=10.15 \
  -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" \
  -DBUILD_TESTING=ON \
  -DBUILD_TOOLS=ON

# Build Release
cmake --build build_macos --config Release

# Run tests
ctest --test-dir build_macos --build-config Release --output-on-failure -j$(sysctl -n hw.ncpu)

# Run the app
open build_macos/AutoMixMasterApp_artefacts/Release/AutoMixMaster.app
```

### Build with Unix Makefiles (Alternative)

```bash
cmake -S . -B build_macos -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=10.15 \
  -DBUILD_TESTING=ON \
  -DBUILD_TOOLS=ON
cmake --build build_macos -j$(sysctl -n hw.ncpu)
```

### macOS-Specific Notes

- **Frameworks**: JUCE's CMake automatically links CoreAudio, CoreMIDI, AudioToolbox, Accelerate, and other required Apple frameworks — no manual configuration needed.
- **Universal binaries**: Use `-DCMAKE_OSX_ARCHITECTURES="arm64;x86_64"` for Apple Silicon + Intel support. Omit for native-only builds.
- **Code signing**: Required for distribution. Add `-DCMAKE_XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY="Developer ID Application"` for signed builds.
- **App bundle**: JUCE's `juce_add_gui_app` creates a proper `.app` bundle with `Info.plist` automatically.
- **No additional system dependencies**: Unlike Linux, macOS does not need separate packages for audio/graphics — these ship as system frameworks.
- **Common issue**: If you see "No matching SDK found", run `xcode-select -s /Applications/Xcode.app/Contents/Developer`.

---

## Current App Features

- Import stems: `.wav`, `.aiff/.aif`, `.flac`, `.mp3`, `.ogg`
- Optional original mix reference for residual blend and soft target mastering
- Auto Mix plan generation (heuristic + optional AI override)
- Auto Master plan generation (heuristic + optional AI override)
- Renderer selection:
  - `BuiltIn`
  - `PhaseLimiter` (auto-discovered external binary, fallback-safe)
  - Asset/user external limiters (`renderer.json` descriptors)
- Export formats:
  - Lossless: `WAV`, `FLAC`, `AIFF`
  - Lossy: `OGG`, `MP3` (bitrate/quality controls)
- Batch folder processing:
  - Parallel analysis and parallel rendering
  - Lossy or lossless export in batch
  - Per-item output/report summary
- Cancel button for long export/batch tasks
- Session save/load
- A/B preview source switching for original vs rendered buffers

---

## GUI Workflow

1. Import stems.
2. Optional: load original mix.
3. Optional: choose AI packs (Role/Mix/Master).
4. Run **Auto Mix**.
5. Run **Auto Master**.
6. Choose renderer and export format/bitrate.
7. Click **Export** (or **Batch Folder**).
8. Use **Cancel** if needed.

Output files:

- `<name>.<ext>`
- `<name>.<ext>.report.json`

---

## Lossy Export and Codec Notes

Formats are selected from `RenderSettings.outputFormat` and resolved against file extension when `auto` is used.

- `WAV`, `AIFF`, `FLAC` are written directly.
- `OGG`, `MP3` are written when those JUCE codec paths are available in the build.
- `MP3` path supports JUCE MP3 writer and optional LAME fallback (`LAME_BIN`/`PATH`) when enabled.
- External/PhaseLimiter renderers render via WAV temp files and then encode final output into requested lossy/lossless format.

Primary references:

- JUCE `AudioFormatManager::registerBasicFormats` docs: <https://docs.juce.com/master/classAudioFormatManager.html>
- JUCE `MP3AudioFormat` docs: <https://docs.juce.com/master/classjuce_1_1MP3AudioFormat.html>
- JUCE `LAMEEncoderAudioFormat` docs: <https://docs.juce.com/master/classjuce_1_1LAMEEncoderAudioFormat.html>

---

## Batch Performance and Hardware Acceleration

Batch and offline rendering are now hardware-aware and multi-threaded:

- Parallel stem analysis with bounded worker threads.
- Parallel batch rendering (`renderParallelism` workers).
- Parallel stem import/pre-processing in offline render pipeline.
- Automatic defaults from `std::thread::hardware_concurrency()`.
- `RenderSettings.processingThreads` to override thread count.
- `RenderSettings.preferHardwareAcceleration` to force single-thread deterministic fallback when disabled.

This project currently accelerates via multi-core CPU parallelism (no GPU offload path in this repo).

---

## MasterPlan Application Across Renderers

All supported limiter paths now use `Session.masterPlan`:

- `BuiltInRenderer`: applies full master plan directly.
- `PhaseLimiterRenderer`:
  - uses plan limiter ceiling for PhaseLimiter invocation
  - runs compliance post-check bounded by master plan before final write
- `ExternalLimiterRenderer`:
  - includes master-plan limiter and gain parameters in request JSON
  - runs compliance post-check bounded by master plan before final write

If external processing fails/times out/missing, renderers fallback safely to `BuiltIn`.

---

## AI Model Packs

### Runtime scan roots

`ModelManager` scans model packs from:

- configured root(s) (default `ModelPacks`)
- `assets/models`
- `assets/modelpacks`
- `assets/ModelPacks`
- `Assets/ModelPacks`
- optional env var: `AUTOMIX_MODELPACK_PATHS`

Relative roots are resolved across current-directory ancestors, so packs are found from app, tool, and test working directories.

### Required metadata and schema gating

`model.json` is validated with deterministic load/run checks:

- required metadata: `license`, `source`, `feature_schema_version`
- model file must exist
- optional checksum must match
- feature schema compatibility required
- runtime inference rejects feature-count mismatches

### Supported engines

Both supported engines are functional in this codebase:

- `onnxruntime` (deterministic local backend with schema/task gating)
- `rtneural` (deterministic local backend)

### Included demo packs

- `assets/models/demo-role-v1`
- `assets/models/demo-mix-v1`
- `assets/models/demo-master-v1`

---

## Limiter Pack Discovery and Installation

`RendererRegistry` auto-discovers external limiter descriptors by scanning:

- `assets/limiters/**/renderer.json`
- `assets/renderers/**/renderer.json`
- `Assets/Limiters/**/renderer.json`

Relative roots are resolved across ancestor directories for runtime flexibility.

Included descriptor examples:

- `assets/limiters/phaselimiter/renderer.json`
- `assets/limiters/external-template/renderer.json`

---

## Dependencies

All build dependencies are fetched automatically via CMake `FetchContent` — no manual installation required on any platform:

| Dependency | Version | License | Purpose |
|---|---|---|---|
| **JUCE** | 8.0.8 | GPL v3 / Commercial | Audio framework, GUI, DSP, codec support |
| **nlohmann/json** | 3.11.3 | MIT | JSON serialization for sessions, reports, model metadata |
| **libebur128** | 1.2.6 | MIT | EBU R128 / ITU-R BS.1770 standards loudness metering |
| **Catch2** | 3.7.1 | BSL 1.0 | Unit and regression test framework (test builds only) |

Platform-specific requirements:

- **Windows**: Visual Studio 2019+ with C++ Desktop workload (or standalone MSVC toolchain).
- **Linux/WSL**: System packages for X11, ALSA, and FreeType (see Quickstart above).
- **macOS**: Xcode 14+ with Command Line Tools. No additional packages needed.

---

## Developer Tools (`automix_dev_tools`)

Build with `-DBUILD_TOOLS=ON`.

```bash
automix_dev_tools export-features --session <session.json> --out <features.jsonl>
automix_dev_tools export-segments --session <session.json> --out-dir <dir> [--segment-seconds <sec>]
automix_dev_tools validate-modelpack --pack <modelpack_dir>
automix_dev_tools list-supported-models
automix_dev_tools install-supported-model --id <model_id> [--dest <assets/models>]
automix_dev_tools list-supported-limiters
automix_dev_tools install-supported-limiter --id <limiter_id> [--dest <assets/limiters>]
```

Supported model installers:

- `demo-role-v1`
- `demo-mix-v1`
- `demo-master-v1`

Supported limiter installers:

- `phaselimiter`
- `external-template`

---

## Build Options

Common CMake toggles:

- `BUILD_TESTING=ON|OFF`
- `BUILD_TOOLS=ON|OFF`
- `DISTRIBUTION_MODE=OSS|PROPRIETARY`
- `ENABLE_PHASELIMITER=ON|OFF`
- `ENABLE_ONNX=ON|OFF`
- `ENABLE_RTNEURAL=ON|OFF`
- `ENABLE_LIBEBUR128=ON|OFF`
- `ENABLE_EXTERNAL_TOOL_SUPPORT=ON|OFF`
- `ENABLE_GPL_BUNDLED_LIMITERS=ON|OFF`

---

## Testing

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON -DBUILD_TOOLS=ON
cmake --build build --parallel
TMPDIR=/tmp ctest --test-dir build --output-on-failure -j4
```

Regression CLI:

```bash
./build/automix_regression_cli --baseline ./tests/regression/baselines.json
```

---

## Cross-Platform Compatibility

The codebase is verified cross-platform with the following design:

- All platform-specific code is guarded by `#if defined(_WIN32)` / `#elif defined(__APPLE__)` / `#else` (Linux).
- Path separator handling uses `std::filesystem` throughout — no hardcoded slashes or backslashes.
- Environment variable reading uses `_dupenv_s` on Windows and `std::getenv` elsewhere.
- `PATH` parsing uses `;` delimiter on Windows and `:` on Linux/macOS.
- External process execution uses JUCE `ChildProcess`, which abstracts platform differences.
- Thread concurrency defaults via `std::thread::hardware_concurrency()` (cross-platform).
- PhaseLimiter binary discovery scans platform-appropriate subdirectories (`windows/`, `mac/`, `linux/`) and executable names (`.exe` on Windows).
- JUCE's CMake integration automatically links the correct system frameworks per platform (CoreAudio on macOS, ALSA/JACK on Linux, WASAPI on Windows).

---

## Known Limitations

- GPU compute acceleration is not implemented; acceleration is CPU-thread based.
- MP3/OGG export depends on codec support compiled into JUCE for your build environment.
- External limiter integrations require the external binary to honor the request JSON contract.
- The current local AI backends are deterministic inference adapters, not native high-throughput production runtimes.
- Real-time transport/DAW-style playback editing is still limited; this app is primarily an offline render workflow.

---

## License

The AutoMixMaster project is licensed under the **GNU General Public License v3 (GPLv3)**.

In plain English (not legal advice):
- If you distribute the app (or modified versions), you need to provide the corresponding source under GPLv3.
- You can use it privately without distribution obligations.

### Distribution Modes

The project supports two distribution modes controlled by the `DISTRIBUTION_MODE` CMake option:

- **OSS mode** (`DISTRIBUTION_MODE=OSS`): Uses GPL v3 licensed JUCE framework. All components must be GPL-compatible. This is the default mode for open-source distribution.
- **Proprietary mode** (`DISTRIBUTION_MODE=PROPRIETARY`): Uses commercially licensed JUCE framework. Allows proprietary distribution while maintaining copyleft-bundling gates for GPL components.

### Third-party Components

This project includes/uses third-party software with its own licensing:

- **JUCE** (fetched at build time): Dual-licensed GPL v3 / Commercial. Used under GPL v3 in OSS mode, commercial license in proprietary mode.
- **nlohmann/json** (fetched at build time): MIT License.
- **Catch2** (fetched at build time for tests): Boost Software License 1.0.
- **libebur128** (fetched at build time): MIT License.
- **PhaseLimiter binaries and resources** under `assets/phaselimiter/`: See that folder for its license files (typically GPL or compatible).

All third-party dependencies are compatible with the chosen distribution mode.
