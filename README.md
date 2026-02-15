# AutoMixMaster

AutoMixMaster is a JUCE/CMake desktop app for deterministic stem mixing/mastering:

`Analysis -> MixPlan -> MasterPlan -> Renderer -> Audio + JSON report`

This branch implements the full roadmap scope from `docs/roadmap.md`, including GPU-provider aware AI runtime controls, codec availability probing, external limiter contract validation, DAW-style transport preview, multiband mastering controls, platform loudness presets, stem-separation import flow, and expanded spectral feature extraction.

## Implemented Roadmap Highlights

1. GPU-accelerated ML runtime controls
- `RenderSettings.gpuExecutionProvider` supports `auto|cpu|directml|coreml|cuda`.
- GUI selector added under `ML Provider`.
- ONNX adapter supports provider preference selection with CPU fallback and diagnostics.

2. Enhanced codec portability and visibility
- `WavWriter::getAvailableFormats()` probes writer availability at runtime.
- GUI export dropdown reflects available codecs and provides diagnostics for unavailable formats.
- MP3 export now supports two fallback layers:
  - JUCE MP3/LAME encoder integration when enabled at build-time.
  - External `lame` CLI encoding fallback when JUCE MP3 writer creation is unavailable.
  - Cross-platform on-demand LAME downloader fallback when no local `lame` binary is present.
- Runtime availability marks MP3 as available if either writer path is valid.

3. External limiter contract validation
- `ExternalLimiterRenderer::validateBinary()` performs `--validate` handshake, timeout handling, schema checks, and capability parsing.
- `RendererRegistry` now validates external renderers during discovery and marks invalid binaries unavailable with diagnostics.
- Formal schema added: `docs/external_limiter_contract.schema.json`.
- New tooling command: `automix_dev_tools validate-external-limiter --binary <path>`.

4. AI runtime throughput improvements
- ONNX adapter includes graph optimization policy (`ORT_ENABLE_ALL` semantics), warmup, preallocated scratch buffers, batch run support, and quantized variant preference (`*_int8.onnx`, `*_fp16.onnx`).
- ONNX backend diagnostics now expose runtime counters and timing telemetry:
  - `calls`, `batches`, `provider_fallbacks`, `avg_inference_ms`, `warmup_ms`.
- Model metadata now supports specialization fields consumed by ONNX runtime setup:
  - `preferredPrecision`, `providerAffinity`, `defaultIntraOpThreads`, `defaultInterOpThreads`, `enableProfiling`.
- Build toggles for RTNeural acceleration options are exposed: `RTNEURAL_XSIMD`, `RTNEURAL_USE_AVX`.

5. Real-time transport preview
- `TransportController` with play/pause/stop/seek/progress state broadcasting.
- Waveform preview component with playhead cursor.
- GUI transport slider plus stem solo/mute preview routing.

6. Multiband dynamics in mastering
- Added `dsp::MultibandProcessor`.
- `MasterPlan` supports `enableMultibandCompressor` and `multibandSettings`.
- Heuristic mastering chain inserts multiband stage before limiter when enabled.

7. Platform loudness presets
- Added `MasterPreset` values: `Spotify`, `AppleMusic`, `YouTube`, `AmazonMusic`, `Tidal`, `BroadcastEbuR128`.
- Data-driven preset overrides loaded from `assets/mastering/platform_presets.json`.
- GUI now has a platform preset selector.

8. Stem separation integration path
- Optional `StemSeparator` integrated into import flow for single mixed-file imports when `AI-separated stems` is enabled.
- Implements model-backed overlap-add separation when a separator model is available.
- Includes deterministic overlap-add fallback and residual-safe reconstruction when model output is unavailable.
- Per-stem `separationConfidence` and `separationArtifactRisk` are now attached to imported stems and serialized in sessions.

9. Advanced spectral analysis and ML features
- Analysis now includes multi-resolution STFT summaries, spectral flux, onset strength, crest factor, MFCCs, and constant-Q bins.
- Feature schema expanded in both runtime and training export:
  - `src/ai/FeatureSchema.cpp`
  - `tools/training/feature_schema_v1.json`

10. UI responsiveness and task scheduling
- `Auto Mix` and `Auto Master` now execute on background worker threads and report progress back to the JUCE message thread.
- UI remains responsive during intensive operations; cancel remains available.
- Preview rebuilding from Auto Mix now runs asynchronously to avoid post-task UI stalls.

## Build

### Configure + build (Linux/WSL/macOS)

```bash
cmake -S . -B build-codex -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DBUILD_TOOLS=ON
cmake --build build-codex --parallel
```

### Configure + build (Windows, VS 2026)

Windows Visual Studio builds are pinned to the VS 2026 generator (`Visual Studio 18 2026`).

```powershell
cmake -S . -B build_win_vs2026_release -G "Visual Studio 18 2026" -A x64 -DBUILD_TESTING=OFF -DBUILD_TOOLS=OFF -DBUILD_SHARED_LIBS=OFF
cmake --build build_win_vs2026_release --config Release --parallel
```

Built executable:
- `build_win_vs2026_release/AutoMixMasterApp_artefacts/Release/AutoMixMaster.exe`

## Run

### App

```bash
./build-codex/AutoMixMasterApp_artefacts/AutoMixMaster
```

### Tests

```bash
TMPDIR=/tmp ctest --test-dir build-codex --output-on-failure -j4
```

As of **February 14, 2026**, this branch passes all 50 tests in `automix_tests`.

## GUI Workflow

1. Import stems (or one mixed track with `AI-separated stems` enabled).
2. Optional: load original mix for A/B and residual blend.
3. Choose AI packs and ML provider (`auto/cpu/directml/coreml/cuda`).
4. Run `Auto Mix`.
5. Run `Auto Master` and choose master/platform presets.
6. Optional: set stem solo/mute and audition with transport controls.
7. Choose renderer + export format.
8. Export single song or batch folder.

Outputs:
- `<name>.<ext>`
- `<name>.<ext>.report.json`

## Renderers

- `BuiltIn`: always available in-process renderer.
- `PhaseLimiter`: external binary discovery with fallback-safe behavior.
- External limiters from descriptor `renderer.json` files.

External descriptor scan roots:
- `assets/limiters/**/renderer.json`
- `assets/renderers/**/renderer.json`
- `Assets/Limiters/**/renderer.json`

External renderers are now validated at discovery time via the `--validate` contract.

## Codec Availability

Export codec availability is probed at runtime and surfaced in the GUI.

Supported format targets:
- Lossless: `wav`, `aiff`, `flac`
- Lossy: `mp3`, `ogg`

Key build option:
- `ENABLE_BUNDLED_LAME=ON|OFF` (default: `ON`)

MP3 fallback discovery order:
- Bundled lookup roots (when `ENABLE_BUNDLED_LAME=ON`) from current working directory and executable-relative ancestors.
- `lame(.exe)` beside the app executable (`./`, `./bin`, `./lame`).
- `LAME_BIN` environment variable.
- `PATH` entries (quoted entries are supported).
- Downloaded fallback cache (`<user app data>/AutoMixMaster/codecs/lame/<platform>/`).

MP3 downloader controls:
- `AUTOMIX_LAME_SKIP_DOWNLOAD=1` to disable downloader fallback.
- `AUTOMIX_LAME_FORCE_DOWNLOAD=1` to force re-download.
- `AUTOMIX_LAME_VERSION=<version>` to override default (`3.100`).
- `AUTOMIX_LAME_DOWNLOAD_URL=<url>` to override source with a direct archive/binary URL.

## Developer Tools (`automix_dev_tools`)

Built when `-DBUILD_TOOLS=ON`.

```bash
automix_dev_tools export-features --session <session.json> --out <features.jsonl>
automix_dev_tools export-segments --session <session.json> --out-dir <dir> [--segment-seconds <sec>]
automix_dev_tools validate-modelpack --pack <modelpack_dir>
automix_dev_tools validate-external-limiter --binary <path> [--json]
automix_dev_tools list-supported-models
automix_dev_tools install-supported-model --id <model_id> [--dest <assets/models>]
automix_dev_tools list-supported-limiters
automix_dev_tools install-supported-limiter --id <limiter_id> [--dest <assets/limiters>]
automix_dev_tools install-lame-fallback [--force] [--json]
```

## External Limiter Contract

Schema:
- `docs/external_limiter_contract.schema.json`

Validation behavior:
- The renderer sends a minimal validation request using `--validate --request <json>`.
- Expected response fields:
  - `schemaVersion` (major version 1)
  - `version` (string)
  - `supportedFeatures` (string array)
- Validation now emits explicit error taxonomy codes (for tooling and registry diagnostics), including:
  - `binary_missing`, `launch_failed`, `timeout`, `exit_code`, `invalid_json`, `missing_version`, `missing_supported_features`, `schema_incompatible`.
- Invalid binaries are surfaced as unavailable in renderer discovery and fallback to BuiltIn at render time.

## AI Model Packs

`ModelManager` scans from:
- configured roots
- `assets/models`
- `assets/modelpacks`
- `assets/ModelPacks`
- `Assets/ModelPacks`
- env var `AUTOMIX_MODELPACK_PATHS`

Packs are schema-gated and validated for required metadata (`license`, `source`, `feature_schema_version`) and model-file presence/checksum.
Additional optional runtime specialization metadata is supported:
- `preferredPrecision`
- `providerAffinity`
- `defaultIntraOpThreads`
- `defaultInterOpThreads`
- `enableProfiling`

## Key CMake Options

- `BUILD_TESTING=ON|OFF`
- `BUILD_TOOLS=ON|OFF`
- `ENABLE_ONNX=ON|OFF`
- `ENABLE_RTNEURAL=ON|OFF`
- `RTNEURAL_XSIMD=ON|OFF`
- `RTNEURAL_USE_AVX=ON|OFF`
- `ENABLE_LIBEBUR128=ON|OFF`
- `ENABLE_PHASELIMITER=ON|OFF`
- `ENABLE_EXTERNAL_TOOL_SUPPORT=ON|OFF`
- `ENABLE_BUNDLED_LAME=ON|OFF` (default: `ON`)
- `DISTRIBUTION_MODE=OSS|PROPRIETARY`

## Known Limits

- ONNX inference remains a deterministic adapter layer in this branch (not a linked native ONNX Runtime session integration).
- External limiter contract still enforces major schema compatibility (`1.x`) rather than strict minor-version negotiation.

## License

AutoMixMaster is GPLv3 in OSS mode (`DISTRIBUTION_MODE=OSS`).

Third-party dependencies include JUCE, nlohmann/json, Catch2, and libebur128, each under their respective licenses.
