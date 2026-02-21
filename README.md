<div align="center">

# AutoMixMaster

**Version 0.4.0 | Status: Operational**

<img src="assets/AutoMixMaster.jpg" alt="AutoMixMaster application interface" width="860">

**FIXED-RULE AUDIO WORKFLOW FOR MIXING AND MASTERING MUSIC STEMS**

***Designed for amateur music producers and hobbyists***

</div>

<p align="center">
<a href="#overview">Overview</a> •
<a href="#quick-start">Quick Start</a> •
<a href="#releases">Releases</a> •
<a href="#first-session">First Session</a> •
<a href="#batch-processing">Batch Processing</a> •
<a href="#feature-set">Feature Set</a> •
<a href="#estimated-system-requirements">System Requirements</a> •
<a href="#build--install">Build + Install</a> •
<a href="#licensing">Licensing</a>
</p>

---

## Overview

AutoMixMaster is a robust automation utility for repetitive stem-level audio preparation. It employs a fixed, deterministic workflow for level balancing, gain staging, and output preparation, acting as a personal assistant for your mixing and mastering needs.

The project prioritizes experimentation and rapid iteration, making it ideal for creators working with raw multitrack material or AI-generated stems. Recent updates include a responsive interface overhaul, a one-click Mix + Master pipeline, AI stem separation pre-checks before Auto Mix, a task-scoped model browser (install/uninstall/set active), renderer chaining, bundled external renderer integrations (FFmpeg/SoX/rsgain), and optional export report sidecar control in Settings.

---

## Quick Start

Getting started with AutoMixMaster is simple.

> ⚠️ **Testing disclaimer:** only the **Windows** version has been manually tested end-to-end so far.  
> Linux, macOS, and ARM64 artifacts are currently provided as best-effort builds.

### Windows (Pre-compiled Executable)

Windows users can download the portable release zip (`AutoMixMaster-windows-<arch>.zip`), extract it, and run `AutoMixMaster.exe`.

### macOS (Pre-compiled App Bundle)

macOS users can download the release zip (`AutoMixMaster-macos-<arch>.zip`), extract it, and open `AutoMixMaster.app`.

### Linux (Prebuilt Packages)

Linux users can download either:

- **AppImage** (`AutoMixMaster-<version>-<arch>.AppImage`) for a portable one-file launch.
- **Debian package** (`automixmaster_<version>_<arch>.deb`) for Ubuntu/Debian install.
- **Flatpak bundle** (`AutoMixMaster-linux-<arch>.flatpak`) for Flatpak-based installs.

### Build From Source

If you are on Linux, or prefer to build the application from source on Windows, refer to the [Build + Install](#build--install) section below for verified instructions.

---

## Releases

Every GitHub Release publishes desktop artifacts for all supported platforms:

- **Linux**: `.deb`, `AppImage`, and `.flatpak` for **x64 + ARM64**
- **Windows**: portable `.zip` for **x64 + ARM64**
- **macOS**: `.app` bundle `.zip` for **x64 + ARM64**

Release packaging is automated by:

- `.github/workflows/release_packages.yml`

> ⚠️ **Testing disclaimer:** only the **Windows** version has been manually tested end-to-end so far.  
> Linux, macOS, and ARM64 artifacts are currently provided as best-effort builds.

---

## First Session

Welcome to your first mixing and mastering session. AutoMixMaster simplifies the process into a few core steps:

1. **Import audio**: Drag and drop files onto the waveform area, or click `Import`. Supported formats: WAV, AIFF, FLAC, MP3, OGG.
2. **(Optional) enable AI Stem Separation**: Toggle **AI Stem Separation** and check the badge beside it (`Separation model: <name/none>`).  
   - If exactly one full-mix track is loaded, separation runs before Auto Mix.  
   - If multiple files are loaded, they are treated as regular stems and separation is skipped.
3. **Manage models in Model Browser**: Open **Models** to fetch catalog entries, install/uninstall models, and set active packs per task (`mix`, `master`, `analysis`, `separation`) using **Set Active** or **Use Selected for Task**.
4. **Auto Mix**: Click **Auto Mix** to analyze stems and apply deterministic balancing rules.
5. **Auto Master**: Click **Auto Master** to apply mastering strategy and limiting.
6. **One-click pipeline**: Click **Mix + Master** (`Ctrl+Shift+M`) to run Auto Mix → Auto Master → Export. If AI Stem Separation is enabled and one full mix is loaded, separation is performed first, then the pipeline continues automatically.
7. **Export**: Use **Export** (`Ctrl+E`) for manual output control, or rely on pipeline export.

---

## Batch Processing

AutoMixMaster isn't limited to a single track. You can process an entire catalog using **Batch Mode**:

1. Select **Batch** from the main dashboard.
2. Choose a source folder containing your stems or full tracks.
3. Apply a shared profile or preset. The Task Orchestrator will manage the queue asynchronously.
4. The system groups stems into songs by filename role patterns (for example `_vocals`, `_bass`, `(drums)`, `(guitar)`), then renders **one mastered song per grouped set**.

Batch outputs are auto-named as:

- `<song>_AutoMixMaster_YYYYMMDD_XX.<ext>`

**Recursive folder support:** use the **Recursive Batch** toggle in the Control Deck to include subfolders during batch discovery.  
For headless/advanced runs, `AUTOMIX_BATCH_RECURSIVE=1` is also supported.

---

## Feature Set

| Module | Description |
| :--- | :--- |
| **Auto Mix** | Rapid algorithmic rules for level balancing and spatial placement. |
| **Auto Master** | Automated gain staging and peak limiting tailored for final output goals. |
| **Mix + Master Pipeline** | One-click button (`Ctrl+Shift+M`) runs Auto Mix → Auto Master → Export to a chosen folder. |
| **AI Stem Separation Gate** | When enabled, a single full-mix import is split into stems before Auto Mix/Mix + Master. |
| **Separation Readiness Badge** | Always-visible status (`Separation model: <name/none>`) beside AI Stem Separation toggle. |
| **Live Progress Tracking** | Task Center shows real-time progress bars and percentages for active operations. |
| **Task Log Utilities** | Task Center includes **Copy Log** for quick report sharing/debugging. |
| **Drag and Drop** | Drop WAV, AIFF, FLAC, MP3, or OGG files directly onto the waveform area to import. |
| **Task Orchestrator** | Real-time monitoring of asynchronous tasks with timestamped activity log. |
| **Model Browser (Task Scoped)** | Curated or raw catalog fetch, install/uninstall, compatibility gating, and active model assignment by task scope. |
| **Dual Model Sources** | Catalog/install supports both Hugging Face and GitHub Release-backed models. |
| **Batch Mode** | Processes multiple tracks/folders sequentially with unified settings. |
| **Renderer Chain** | Optional staged renderer chain (`Logical All` or `Master + rsgain`) with live chain preview. |
| **Renderer Availability Guardrails** | Renderer selector only lists currently available integrations (unavailable entries are hidden). |
| **Bundled Renderer Integrations** | Built-in registry/discovery for PhaseLimiter, FFmpeg, SoX, and rsgain (`*_BIN` env overrides supported). |
| **Export Report Sidecar Toggle** | Settings can disable per-export `.report.json` generation. |
| **Export Verification Report** | After export, a verification pass confirms mix/master processing and reports objective difference metrics. |
| **Batch Verification Summary** | Batch completion triggers a summary report confirming processed outputs and average mix/master deltas. |
| **Analysis Tools** | Real-time GlowMeters exposing LUFS and peak measurements for visual aid. |

---

## Estimated System Requirements

These are **practical estimates** for AI-heavy workflows (especially ONNX-based separation/mix/master inference), not strict hard limits.

AutoMixMaster is designed to benefit from **GPU acceleration** via ONNX Runtime providers.

### Minimum workable

- **CPU:** modern **6-core / 12-thread** desktop CPU (Ryzen 5 5600 / Core i5-12400 class)
- **RAM:** **16 GB minimum**
- **GPU:** compatible acceleration path with ~**6 GB VRAM**
  - Windows DirectML path: **DirectX 12-capable GPU**
  - CUDA path: **NVIDIA CUDA-capable GPU**
- **Storage:** ~10 GB free (models, temp files, exports)

### Recommended (smoother)

- **CPU:** **8 cores / 16 threads or better** (Ryzen 7 / Core i7 class)
- **RAM:** **32 GB**
- **GPU:** **8–12 GB VRAM**

### Heavy batch / long sessions

- **CPU:** **12 cores+** strongly recommended
- **RAM:** **32–64 GB**
- **GPU:** **12 GB+ VRAM**

### Why these estimates

- ONNX Runtime DirectML requires a **DirectX 12-capable** GPU and supports broad AMD/NVIDIA/Intel device ranges ([DirectML EP requirements](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)).
- ONNX Runtime CUDA EP is built for **NVIDIA CUDA-enabled** GPUs ([CUDA EP docs](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)).
- Demucs guidance indicates GPU memory can be around **3 GB minimum** and around **7 GB** at default settings, so 8 GB VRAM is a safer practical target ([Demucs README memory notes](https://github.com/facebookresearch/demucs/blob/main/README.md)).
- CPU-only inference can be significantly slower than GPU-backed runs, so a practical baseline is a modern **6C/12T** CPU, with **8C/16T+** recommended for smoother interactive use and batch work ([Demucs README](https://github.com/facebookresearch/demucs/blob/main/README.md)).

---

## Build + Install

### Windows (Visual Studio 2026)

1. Configure

```bash
cmake -S . -B build -G "Visual Studio 18 2026" -A x64
```

1. Build

```bash
cmake --build build --config Release --parallel
```

### Ubuntu Linux (24.04+)

1. Install dependencies

```bash
sudo apt-get install -y \
  libasound2-dev libfreetype6-dev libx11-dev libxcomposite-dev \
  libxcursor-dev libxext-dev libxinerama-dev libxrandr-dev \
  libxrender-dev libwebkit2gtk-4.1-dev libglu1-mesa-dev mesa-common-dev
```

1. Configure + build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

1. Run tests (optional but recommended)

```bash
ctest --test-dir build --output-on-failure
```

### Linux Package Builds (.deb + AppImage)

After building, create distributable Linux packages with:

```bash
./tools/package_linux.sh
```

Output artifacts are written to `dist/linux/`.

### Flatpak

Manifest path:

`packaging/flatpak/io.automixmaster.AutoMixMaster.yml`

Install Flatpak tooling:

```bash
sudo apt-get install -y flatpak flatpak-builder
```

Add Flathub and install required runtime/SDK:

```bash
flatpak remote-add --user --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo
flatpak --user install -y flathub org.freedesktop.Platform//24.08 org.freedesktop.Sdk//24.08
```

Build bundle:

```bash
./tools/build_flatpak.sh
```

Output:

- `dist/flatpak/AutoMixMaster.flatpak`

### Bundled Renderer Tools (Optional)

The renderer registry can auto-discover optional CLI tools if you place binaries under:

- `assets/ffmpeg/bin/ffmpeg(.exe)` or set `FFMPEG_BIN`
- `assets/sox/bin/sox(.exe)` or set `SOX_BIN`
- `assets/rsgain/bin/rsgain(.exe)` or set `RSGAIN_BIN`

If a tool is missing, it is hidden from selectable available renderers automatically.

---

## Licensing

AutoMixMaster is distributed under the **GNU General Public License v3 (GPLv3)**.

| Component | License | Role |
| :--- | :--- | :--- |
| JUCE 8.0.8 | AGPLv3 / Commercial | Framework |
| libebur128 | MIT | Metering |
| nlohmann/json | MIT | Metadata |
| Catch2 3.7.1 | BSL-1.0 | Testing |
| PhaseLimiter | GPL / Custom | Limiting |
| FFmpeg | GPL-compatible | Optional renderer |
| SoX | GPL-2.0-or-later | Optional renderer |
| rsgain | BSD-2-Clause | Optional ReplayGain tagging stage |
