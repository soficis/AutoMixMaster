# AutoMixMaster

AutoMixMaster is a small, cross-platform **desktop app** (JUCE + CMake) that helps you:

- **Import stems** (individual tracks like vocals, drums, bass...)
- **Analyze** them (levels, simple spectrum split, stereo correlation, "artifact risk")
- Generate an **automatic mix plan** (gain/pan/filter/dynamics suggestions)
- Generate an **automatic master plan** (loudness + true-peak safe-ish targets)
- **Export** a mastered stereo WAV + a machine-readable **JSON report**

It's intentionally built as a "plan-based" system:

> Analysis -> `MixPlan` / `MasterPlan` -> deterministic DSP render -> export/report

That makes it easier to debug, test, and (later) let AI propose parameters without turning the whole app into a black box.

---

## What you can do in the current app (feature tour)

The GUI (`AutoMixMasterApp`) is a simple offline workflow:

1. **Import** one or more stem files (`.wav`, `.aiff/.aif`, `.flac`)
2. (Optional) load an **Original Mix** stereo file (used for "residual blend" and mastering reference)
3. (Optional) mark imported stems as **AI-separated** (enables safer mixing heuristics)
4. Click **Auto Mix** to:
   - analyze stems
   - generate a `MixPlan`
   - show a per-stem metric table + JSON report + decision log
5. Click **Auto Master** to:
   - render a raw stereo mix from stems
   - generate a `MasterPlan` (optionally nudged toward the Original Mix)
6. Choose a **Renderer**:
   - `BuiltIn` (default; fully in-process)
   - `PhaseLimiter` (optional; external binary if available, otherwise auto-fallback)
7. Click **Export** to produce:
   - `output.wav`
   - `output.wav.report.json` (metrics + logs)

---

## Build & run (novice-friendly)

### Prerequisites

- **CMake** >= 3.24
- A **C++20** compiler
  - Windows: Visual Studio 2022 "Desktop development with C++"
  - macOS: Xcode command line tools
  - Linux: GCC/Clang + dev packages (see below)

This project fetches dependencies at configure time via CMake `FetchContent`:

- JUCE `8.0.8`
- nlohmann/json `v3.11.3`
- Catch2 `v3.7.1` (tests only)

### Configure + build

#### Windows (PowerShell)

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DBUILD_TOOLS=ON
cmake --build build --config Release --parallel
```

#### macOS / Linux (bash)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DBUILD_TOOLS=ON
cmake --build build --parallel
```

### What gets built (CMake targets)

- `automix_core` (library): domain + analysis + engine + renderers + AI scaffolding
- `AutoMixMasterApp` (GUI): the JUCE desktop app
- `automix_dev_tools` (CLI, optional): dataset export + model pack validation (`-DBUILD_TOOLS=ON`)
- `automix_tests` (unit tests, optional): Catch2 test binary (`-DBUILD_TESTING=ON`)
- `automix_regression_cli` (optional): renders a deterministic fixture suite (`-DBUILD_TESTING=ON`)

### Linux packages (if you see missing X11/ALSA headers)

Ubuntu/Debian equivalent (mirrors CI):

```bash
sudo apt-get update
sudo apt-get install -y \
  libasound2-dev \
  libx11-dev libxext-dev libxrandr-dev libxinerama-dev libxcursor-dev \
  libfreetype6-dev libfontconfig1-dev \
  libglu1-mesa-dev mesa-common-dev
```

### Run the app

JUCE CMake builds usually place GUI app artefacts under:

- `build/AutoMixMasterApp_artefacts/<Config>/` (multi-config generators like Visual Studio)
- or directly under `build/` (single-config generators)

Look for an executable / `.app` bundle with a name like **AutoMixMaster** and run it.

---

## Using the GUI (step-by-step)

### 1) Import stems

- Click **Import**
- Select one or more audio files
- If these stems came from a source-separation model (Spleeter/Demucs/etc), enable **AI-separated stems** *before importing* so the `StemOrigin` is set to `Separated`.

What this affects:

- Separated stems get **more conservative gain changes** and **less aggressive compression**.
- High "artifact risk" can trigger **soft expansion** instead of compression.

### 2) (Optional) Load Original Mix

- Click **Original Mix**
- Select your original stereo mix (`wav/aiff/flac`)

This enables two features:

1. **Residual Blend** (mix stage): add back a tiny bit of what your stems *don't* explain.
2. **Soft reference targeting** (master plan): nudge loudness / tilt / glue settings toward the reference.

### 3) Residual Blend (%)

The slider is intentionally small: **0.0 to 10.0** means **0% to 10%**.

Under the hood:

- The app aligns the Original Mix to the stem sum using cross-correlation.
- Computes `residual = originalMixAligned - stemSum`.
- Blends `stemSum + residual * (blendPercent / 100)`.
- Enforces a conservative ceiling (defaults to `-1 dBTP` in current pipeline call sites).

This is most useful when:

- you have separated stems with "holes" or watery artifacts
- you want a *little* of the original glue back without fully replacing your stem mix

### 4) Auto Mix

Click **Auto Mix** to:

- run per-stem analysis
- choose a role (by filename heuristics + simple spectrum rules)
- generate a `MixPlan`:
  - gain staging toward role-based target RMS levels
  - panning templates by role
  - simple high-pass filter suggestions
  - conservative compressor/expander toggles depending on stem origin + artifact risk

The UI shows:

- A table of metrics (Peak/RMS/Crest, Low/Mid/High energy split, Silence ratio)
- A JSON analysis report
- A plain-English decision log (what it decided and why)

### 5) Auto Master

Click **Auto Master** to generate a `MasterPlan`.

Important: **Auto Master only creates a plan**. The plan is applied during export by the `BuiltIn` renderer.

If you loaded an Original Mix, the plan is "soft targeted" toward it:

- loudness target and pre-gain are nudged toward the reference loudness
- glue ratio is adjusted based on crest factor differences
- EQ tilt is enabled when the spectral tilt differs enough

### 6) Export

Click **Export** to render offline and write files:

- **WAV audio**
- **JSON report** next to it: `yourfile.wav.report.json`

Renderer choices:

- **BuiltIn**
  - Uses the app's offline mix + mastering chain.
  - Uses `Session.mixPlan` and `Session.masterPlan` if present.
  - Generates a detailed report including the master decision log.
- **PhaseLimiter**
  - If PhaseLimiter is found, it runs the external `phase_limiter` binary on the rendered raw mix.
  - If not found (or it crashes), it **falls back** to the BuiltIn renderer.
  - Current limitation: it does **not** apply `Session.masterPlan` (it's effectively "mix + PhaseLimiter", not "mix + master plan").

---

## Outputs: what gets written

### Exported audio

- Always WAV.
- `BuiltIn` renderer writes at `RenderSettings.outputBitDepth` (GUI currently exports 24-bit).
- `PhaseLimiter` renderer currently forces **16-bit WAV output** (it invokes PhaseLimiter with `-bit_depth=16`).

### JSON report (`.report.json`)

The renderer writes a JSON file next to your WAV. Typical fields include:

- renderer metadata (`renderer`, output path, PhaseLimiter binary path if used)
- mastering metrics (integrated loudness estimate, true peak estimate)
- spectrum summary (low/mid/high energy ratios, stereo correlation)
- decision log (BuiltIn only)
- render logs (pipeline stages, residual blend alignment details, etc.)

This report is intended to be:

- easy to inspect manually
- stable enough for regression tests and dataset generation

---

## Session JSON format (developer-focused, but beginner-usable)

There's an explicit `Session` domain model and JSON serializer. The GUI currently doesn't expose "Save/Load session", but the format is already supported in code (and used by tools/tests).

Minimal example:

```json
{
  "schemaVersion": 2,
  "sessionName": "My Session",
  "residualBlend": 5.0,
  "originalMixPath": "C:/path/to/original_mix.wav",
  "stems": [
    {
      "id": "stem_1",
      "name": "Vox",
      "filePath": "C:/path/to/vocals.wav",
      "role": "unknown",
      "origin": "separated",
      "enabled": true
    }
  ],
  "buses": [],
  "renderSettings": {
    "outputSampleRate": 44100,
    "blockSize": 1024,
    "outputBitDepth": 24,
    "outputPath": "",
    "rendererName": "BuiltIn"
  }
}
```

Fields are forgiving:

- Missing `originalMixPath`, `mixPlan`, `masterPlan` are fine.
- `residualBlend` is clamped to `0.0 .. 10.0` on load.

---

## Developer tools (datasets + model pack validation)

If you build with `-DBUILD_TOOLS=ON`, you get `automix_dev_tools` (a CLI utility).

### Export analysis features (`.jsonl`)

```bash
automix_dev_tools export-features --session path/to/session.json --out path/to/features.jsonl
```

Each line is a JSON object containing stem identity + safety metadata + analysis metrics.

### Export short audio segments (for listening tests / training)

```bash
automix_dev_tools export-segments --session path/to/session.json --out-dir path/to/dataset --segment-seconds 5
```

Outputs:

- `stems/*.wav`: first `N` seconds per enabled stem
- `mix_segment.wav`: first `N` seconds of the offline raw mix
- `manifest.json`: summary metadata

### Validate a model pack

```bash
automix_dev_tools validate-modelpack --pack path/to/ModelPacks/my_pack
```

Validation checks:

- `model.json` parses
- model file exists
- optional checksum matches
- if the backend is enabled, runs a sample inference and checks expected outputs

---

## AI model packs (implemented scaffolding)

The codebase is ready for drop-in model packs, but the GUI currently only **lists** them and stores the active selection in memory.

### Where the app looks

By default it scans a `ModelPacks/` folder next to your working directory:

```
ModelPacks/
  my_pack/
    model.json
    model.onnx
```

### `model.json` (minimum supported shape)

```json
{
  "schema_version": 1,
  "id": "mix-v1",
  "type": "mix_parameters",
  "engine": "onnxruntime",
  "version": "1.0.0",
  "model_file": "model.onnx",
  "input_feature_count": 5,
  "output_keys": ["confidence", "global_gain_db", "global_pan_bias"]
}
```

Inference task names used by the codebase:

- `role_classifier` (stem role prediction)
- `mix_parameters` (per-stem mix parameter prediction)
- `master_parameters` (mastering parameter prediction)
- `mix_master_override` (global override hook used by `ModelStrategy`)

Supported `engine` values in current code:

- `onnxruntime` (compile-time optional; currently a stub backend)
- `rtneural` (compile-time optional; currently a small deterministic stub)
- `unknown` (schema-only validation)

### Important reality check (as of this repo state)

The "inference backends" are deliberately lightweight **stubs**:

- `OnnxModelInference` currently **does not run ONNX Runtime**; it returns fixed placeholder outputs to prove plumbing.
- `RtNeuralInference` currently **does not load real RTNeural weights**; it returns deterministic probabilities from the input features.

The architecture is real; the heavy runtime integrations are intentionally left as future work.

---

## How it works (codebase walkthrough)

This section maps user-visible features to the actual modules/classes in `src/`.

### 1) Domain model (`src/domain`)

Think of this as the "data contract" of the app:

- `Session`: everything about a project (stems, optional original mix path, plans, render settings).
- `Stem`: one audio file + metadata (`StemOrigin` and `StemRole`).
- `MixPlan`: per-stem decisions (gain/pan/high-pass + simple dynamics flags).
- `MasterPlan`: mastering targets + parameters (LUFS target, true-peak ceiling, glue settings, dither).
- JSON serialization: `nlohmann::json` conversions for the whole graph.

Design note:

- These structs are dependency-light and easy to test/serialize.
- The UI is supposed to *edit* these objects; the engine reads them.

### 2) Analysis (`src/analysis`)

`StemAnalyzer` computes:

- `peakDb`, `rmsDb`, `crestDb`
- low/mid/high energy ratios (simple one-pole filters, not FFT)
- `silenceRatio` (fraction of samples below a small threshold)
- stereo correlation + derived "width"
- `artifactRisk` (heuristic roughness + HF energy + width + silence)

This analysis is used for:

- displaying a useful "what's in these stems?" table
- driving heuristic mixing decisions
- exporting training features for ML workflows

### 3) AutoMix (heuristic) (`src/automix`)

`HeuristicAutoMixStrategy` produces a `MixPlan` using:

- role-based loudness targets (e.g., vocals a bit louder than FX)
- panning templates by role (e.g., guitars alternate L/R)
- high-pass defaults by role
- origin-aware "safety caps":
  - separated stems get smaller gain moves
  - separated stems avoid aggressive compression
  - high artifact risk can enable a gentle expander

Role inference:

- `StemRoleClassifierAI` can use a model backend, but defaults to:
  - filename keyword heuristics (`vox`, `kick`, `bass`, `fx`, ...)
  - simple spectrum rules if names are ambiguous

### 4) Offline mixing/rendering (`src/engine`)

`OfflineRenderPipeline::renderRawMix()` is the heart of the app:

1. Read each enabled stem (`AudioFileIO`, JUCE readers)
2. Resample to project rate if needed (`AudioResampler`, linear)
3. Apply per-stem processing from `StemMixDecision`:
   - one-pole high-pass filter
   - simple compressor (peak detector + ratio + release)
   - simple expander (for separated/artifacty content)
   - gain
   - constant-power panning
   - optional "dry/wet" blend between unprocessed and processed stem audio
4. Sum stems block-by-block (supports cancellation)
5. Apply mix-bus headroom normalization (`MixPlan.mixBusHeadroomDb`)
6. If Original Mix + residual blend enabled:
   - align original mix to stem sum (cross-correlation)
   - compute residual and blend it back (`ResidualBlendProcessor`)

Notes:

- This is offline processing: it loads entire files into memory right now.
- There is a `cancelFlag` in the API, but the GUI doesn't currently expose a cancel button.

### 5) Mastering (heuristic) (`src/automaster`)

`HeuristicAutoMasterStrategy` implements a basic chain:

- pre-gain toward target LUFS (simple integrated loudness estimate)
- optional gentle tonal tilt (very small low trim + high lift)
- glue compressor
- limiter (sample clamp; then iterative loudness + true peak correction)
- optional dither for <24-bit exports

`OriginalMixReference` can "soft target" a plan toward the Original Mix:

- nudges `targetLufs` and `preGainDb`
- adjusts glue ratio based on crest factor differences
- enables EQ tilt if spectral tilt differs enough

### 6) Renderers (`src/renderers`)

Renderers are the "final export" abstraction (`IRenderer`):

- `BuiltInRenderer`
  - renders raw mix
  - applies `MasterPlan` (or creates one if missing)
  - writes WAV + JSON report
- `PhaseLimiterRenderer`
  - discovers a PhaseLimiter binary (`PhaseLimiterDiscovery`)
  - renders raw mix
  - writes a temp input WAV and runs PhaseLimiter as a child process
  - validates output exists; otherwise falls back to BuiltIn
  - writes a JSON report with measured output metrics

PhaseLimiter discovery rules:

1. If `PHASELIMITER_BIN` is set, use it (file path or directory to scan).
2. Otherwise, scan "assets-ish" folders near the working directory / executable:
   - `assets/phaselimiter`, `Assets/PhaseLimiter`, etc.

### 7) Tests and regression (`tests/`)

There are two layers:

- **Unit tests** (Catch2): analysis math, serialization, resampling, residual alignment, PhaseLimiter discovery, etc.
- **Regression suite**: renders a deterministic synthetic fixture through:
  - `heuristic` pipeline
  - `ai` pipeline (uses deterministic inference stubs)
  - compares measured metrics against `tests/regression/baselines.json`

Run tests after building:

```bash
ctest --test-dir build --output-on-failure
```

Run the regression CLI:

```bash
automix_regression_cli --baseline ./tests/regression/baselines.json
```

On Windows with multi-config generators, you may need:

```powershell
ctest --test-dir build -C Release --output-on-failure
.\build\Release\automix_regression_cli.exe --baseline .\tests\regression\baselines.json
```

---

## CMake options you can toggle

- `-DBUILD_TESTING=ON|OFF` (default ON)
- `-DBUILD_TOOLS=ON|OFF` (default ON)
- `-DENABLE_ONNX=ON|OFF` (default OFF) - builds the ONNX inference stub
- `-DENABLE_RTNEURAL=ON|OFF` (default OFF) - enables the RTNeural inference stub
- `-DENABLE_PHASELIMITER=ON|OFF` (default OFF) - currently only sets a compile definition; PhaseLimiter renderer is built regardless

---

## Known limitations (honest notes)

This repo is intentionally small and testable, but that means some things are not wired up yet:

- The GUI does not currently **save/load sessions**, even though the serializer exists.
- There is no **audio preview/playback**; everything is offline render + export.
- The mix plan includes fields like `mudCutDb` that are **not applied** by the current render pipeline.
- Buses (`Session.buses`) exist in the domain model but are not used for routing yet.
- The GUI lists model packs but does not yet run an end-to-end AI pipeline export from selected packs.
- The PhaseLimiter renderer currently ignores `MasterPlan` (it post-processes the raw mix).

If you're extending the project, these are high-value first improvements.

---

## License

The AutoMixMaster project is licensed under the **GNU General Public License v3 (GPLv3)**.

In plain English (not legal advice):

- If you distribute the app (or modified versions), you need to provide the corresponding source under GPLv3.
- You can use it privately without distribution obligations.

### Third-party components

This repo also includes/uses third-party software with its own licensing, including:

- JUCE (fetched at build time)
- nlohmann/json (fetched at build time)
- Catch2 (fetched at build time for tests)
- PhaseLimiter binaries and resources under `assets/phaselimiter/` (see that folder for its license files)
