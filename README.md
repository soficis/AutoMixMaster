# AutoMixMaster

AutoMixMaster is a JUCE/CMake desktop app for deterministic stem mixing and mastering:

`Analysis -> MixPlan -> MasterPlan -> Renderer -> Audio + JSON report`

This branch implements the **v2 roadmap** in `docs/roadmap.md` including native ONNX session support (when linked), advanced separation variants, precision transport UX, trust-policy renderer discovery, and expanded developer tooling for evaluation, catalog processing, and collaboration.

## Since Last Published `ai_dev` Commit

Compared to `origin/ai_dev` commit `7311aa8`, this working tree adds:

1. Native ONNX Runtime session wiring
- CMake now auto-detects ONNX Runtime headers/libs when `ENABLE_ONNX=ON` and enables true native sessions (`AUTOMIX_HAS_NATIVE_ORT=1`) when found.
- Runtime provider canonicalization/tuning and native profiling artifact capture are now included.

2. Advanced separator variants + QA bundle
- `StemSeparator` now supports 2/4/6 stem variants with per-run `targetStemCount`, `gpuMemoryBudgetMb`, and `maxStreams` controls.
- Variant discovery supports `separator_pack.json` and fallback model filenames.
- Separation runs now emit `separation_qa_report.json` with `energyLeakage`, `residualDistortion`, and `transientRetention`.

3. Session/profile/trust-policy data model expansion
- Added project profile catalog support and default profile loader: `assets/profiles/project_profiles.json`.
- Added renderer trust policy support: `assets/renderers/trust_policy.json`.
- Session serialization now includes timeline state, profile/safety identifiers, and MP3 mode fields.

4. UI/transport and responsiveness hardening
- Main window is now resizable (`1280x720` default).
- Added loop in/out controls, timeline zoom, fine-scrub behavior, and loop overlay in waveform preview.
- Session loading, stem import/separation, and export preflight checks are pushed off the UI thread with stale-result guards.
- Task center history and progress updates are throttled to reduce UI stalls.

5. Render/export pipeline upgrades
- Added `Final` / `Balanced` / `Quick` export speed modes with Quick mode defaults for fast turnaround.
- MP3 now supports CBR and true VBR paths (`mp3UseVbr`, `mp3VbrQuality`), including external LAME VBR mode.
- Export metadata is preserved from original mix (or stem fallback) and mapped to MP3 ID3 tags.
- Offline render pipeline adds stem/raw mix caching and adaptive block-sizing for repeated renders.

6. Renderer trust/compliance/reporting parity
- External descriptor trust evaluation supports signature metadata (`fnv1a64`) and optional signed-only enforcement.
- Discovered external renderers now emit capability snapshots (`*.capabilities.snapshot.json`).
- `ExternalLimiter` and `PhaseLimiter` now apply original-mix soft-target guidance when no master plan is pinned, matching built-in behavior.
- Render reports now include plan-source and decision-log provenance metadata.

7. Dev tools and test coverage expansion
- `automix_dev_tools` now includes comparator, catalog processing, session diff/merge, external compatibility, golden eval, plan diff, model/limiter catalog installers, and LAME fallback installer.
- Added/expanded tests for project profile loading, trust-policy enforcement, transport looping, separator variants/QA output, and session schema fields.

## Implemented Scope (v2)

1. Native ONNX runtime sessions (Phase G)
- `OnnxModelInference` supports real ONNX Runtime C++ sessions when `ENABLE_ONNX=ON` and runtime headers/libs are present.
- Provider-aware tuning matrix (CPU/CUDA/DirectML/CoreML), thread controls, and optional profiling artifact export.
- Deterministic adapter fallback remains available when native runtime is unavailable.

2. Advanced separation model workflow (Phase H)
- Multi-variant separation packs (2/4/6 stem targets) with automatic selection and override support.
- Chunked overlap-add processing with GPU memory budget and stream scheduling controls.
- Separation QA report bundle with `energyLeakage`, `residualDistortion`, and `transientRetention`.

3. Precision UX and timeline state (Phase I)
- Loop in/out markers, looped transport playback, timeline zoom, and fine-scrub behavior.
- Timeline loop/zoom/fine-scrub state persists in session JSON.
- Task center panel with timestamped background-task history.

4. External DSP marketplace foundation (Phase J)
- External renderer descriptor signature metadata and trust policy loading.
- Signed/unsigned policy handling and trusted-signer filtering.
- Capability snapshot artifacts written for discovered external renderers.

5. Evaluation/training operations (Phase K)
- Golden corpus evaluator command backed by regression harness baselines.
- Heuristic vs model plan diff report generation.
- Dataset lineage/manifest output for feature export workflows.

## Product Suggestions Implemented

1. Project Profiles
- Data-driven profile bundles (platform target, renderer, codec, model packs, safety policy, preferred stem count).
- Profile selector in UI with auto-application to render/model settings.
- Strict safety policy pinning blocks export when renderer is not profile-pinned.

2. Multi-Render Comparator
- `automix_dev_tools compare-renders` renders multiple targets and ranks by loudness/compliance/artifact risk score.
- JSON and CSV comparator reports are generated.

3. Stem Health Assistant
- Pre-export diagnostics for masking, pumping, harshness, and mono risk.
- Integrated into app export flow and available via CLI (`stem-health`).

4. Catalog Processing Mode
- `catalog-process` headless queue runner with JSON/CSV deliverables.
- Resumable checkpoints via `--checkpoint` and `--resume`.
- Per-item failure handling (unreadable files are marked failed, not fatal to whole run).

5. Creator Collaboration Mode
- `session-diff` for deterministic JSON patch output.
- `session-merge` three-way deterministic merge with conflict reporting and left/right preference policy.

## Build

### Configure + build (Linux/WSL/macOS)

```bash
cmake -S . -B build-codex -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DBUILD_TOOLS=ON
cmake --build build-codex --parallel
```

When `ENABLE_ONNX=ON`, CMake attempts to locate native ONNX Runtime headers/library and enables native session support automatically when found.

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
ctest --test-dir build-codex --output-on-failure -j4
```

As of **February 15, 2026**, this branch passes **55/55 tests**.
Coverage includes project profile loading, renderer trust-policy enforcement, transport loop behavior, separator variant/QA outputs, and session schema round-trip fields.

## GUI Workflow

1. Import stems (or one mixed track with `AI-separated stems` enabled).
2. Optional: load original mix for A/B and residual blend.
3. Select project profile, AI packs, and ML provider (`auto/cpu/directml/coreml/cuda`).
4. Run `Auto Mix`.
5. Run `Auto Master` and choose master/platform preset.
6. Use transport/loop/zoom/fine-scrub for preview.
7. Review task center and stem health diagnostics.
8. Export single song or batch folder (`Quick` mode for speed, strict safety policy can block non-pinned renderers).

Outputs:
- `<name>.<ext>`
- `<name>.<ext>.report.json`

## Session Schema Additions

- `renderSettings`: `exportSpeedMode`, `mp3UseVbr`, `mp3VbrQuality`
- `timeline`: `loopEnabled`, `loopInSeconds`, `loopOutSeconds`, `zoom`, `fineScrub`
- `session`: `projectProfileId`, `safetyPolicyId`, `preferredStemCount`

## Renderers

- `BuiltIn`: always available in-process renderer.
- `PhaseLimiter`: external binary discovery with fallback-safe behavior.
- Descriptor-driven external renderers from:
  - `assets/limiters/**/renderer.json`
  - `assets/renderers/**/renderer.json`
  - `Assets/Limiters/**/renderer.json`

Trust policy candidates:
- `assets/renderers/trust_policy.json`
- `assets/limiters/trust_policy.json`

Discovery side-effect:
- External renderer validation writes capability snapshots next to binaries as `<rendererId>.capabilities.snapshot.json`.

## Codec Availability

Runtime export codec probing is surfaced in the GUI.

Supported format targets:
- Lossless: `wav`, `aiff`, `flac`
- Lossy: `mp3`, `ogg`

MP3 export modes:
- `CBR`: target bitrate via `lossyBitrateKbps` (default `320 kbps`)
- `VBR`: quality ladder via `mp3VbrQuality` (0 best .. 9 smallest)

Export speed modes:
- `Final`: default quality path (`blockSize=1024`, `24-bit`)
- `Balanced`: moderate throughput (`blockSize=2048`, `24-bit`)
- `Quick`: fast-turnaround preset (`blockSize=4096`, `16-bit`) with default codec target `MP3 VBR`
  - Stem-health preflight is skipped in Quick mode to reduce turnaround time.
  - If MP3 is unavailable at runtime, Quick mode falls back to the first available codec.

MP3 fallback order:
- JUCE MP3/LAME writer path (when available)
- External `lame` binary path discovery (`LAME_BIN`, app-adjacent, `PATH`)
- Optional on-demand LAME downloader cache

Metadata retention:
- Exports preserve source metadata from the original mix (or first valid stem fallback when original mix is not set).
- External LAME MP3 export maps common metadata to ID3 tags (`title`, `artist`, `album`, `year`, `track`, `genre`, `comment`).

## Developer Tools (`automix_dev_tools`)

Built when `-DBUILD_TOOLS=ON`.

```bash
automix_dev_tools export-features --session <session.json> --out <features.jsonl> [--manifest <manifest.json>] [--dataset-id <id>] [--source-tag <tag>] [--lineage-parents <id,id,...>]
automix_dev_tools export-segments --session <session.json> --out-dir <dir> [--segment-seconds <sec>]
automix_dev_tools validate-modelpack --pack <modelpack_dir>
automix_dev_tools validate-external-limiter --binary <path> [--json]
automix_dev_tools stem-health --session <session.json> [--out <path>] [--json]
automix_dev_tools compare-renders --session <session.json> [--renderers <id,id,...>] [--out-dir <dir>] [--format <fmt>] [--external-binary <path>] [--json]
automix_dev_tools catalog-process --input <folder> --output <folder> [--checkpoint <path>] [--resume] [--renderer <id>] [--format <fmt>] [--analysis-threads <n>] [--render-parallelism <n>] [--csv <path>] [--json <path>]
automix_dev_tools session-diff --base <session.json> --head <session.json> [--out <patch.json>] [--summary]
automix_dev_tools session-merge --base <session.json> --left <session.json> --right <session.json> --out <session.json> [--prefer <left|right>] [--report <report.json>] [--json]
automix_dev_tools external-limiter-compat --binary <path> [--timeout-ms <ms>] [--required-features <f1,f2>] [--out <report.json>] [--json]
automix_dev_tools golden-eval [--baseline <baselines.json>] [--work-dir <dir>] [--out <report.json>] [--json]
automix_dev_tools plan-diff --session <session.json> [--mix-model <path>] [--master-model <path>] [--out <report.json>] [--json]
automix_dev_tools list-supported-models
automix_dev_tools install-supported-model --id <model_id> [--dest <assets/models>]
automix_dev_tools list-supported-limiters
automix_dev_tools install-supported-limiter --id <limiter_id> [--dest <assets/limiters>]
automix_dev_tools install-lame-fallback [--force] [--json]
```

## External Limiter Contract

Schema:
- `docs/external_limiter_contract.schema.json`

Validation taxonomy:
- `binary_missing`
- `launch_failed`
- `timeout`
- `exit_code`
- `invalid_json`
- `missing_version`
- `missing_supported_features`
- `schema_incompatible`

## AI Model Packs

`ModelManager` scans:
- configured roots
- `assets/models`
- `assets/modelpacks`
- `assets/ModelPacks`
- `Assets/ModelPacks`
- `AUTOMIX_MODELPACK_PATHS`

Runtime specialization metadata supported:
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
- `ENABLE_BUNDLED_LAME=ON|OFF`
- `DISTRIBUTION_MODE=OSS|PROPRIETARY`

## Known Limits

- Native ONNX sessions require external ONNX Runtime headers/library at configure time; otherwise the deterministic adapter path is used.
- Signature verification currently uses descriptor-embedded `fnv1a64` checks (policy and signer lists are data-driven but not PKI-backed yet).
- Collaboration merge is deterministic and semantic for core session structures, but not yet CRDT-based real-time merge.

## License

AutoMixMaster is GPLv3 in OSS mode (`DISTRIBUTION_MODE=OSS`).

Third-party dependencies include JUCE, nlohmann/json, Catch2, and libebur128 under their respective licenses.
