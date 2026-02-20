<div align="center">

# AutoMixMaster

**Version 0.3.0 | Status: Operational**

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
<a href="#build--install">Build + Install</a> •
<a href="#licensing">Licensing</a>
</p>

---

## Overview

AutoMixMaster is a robust automation utility for repetitive stem-level audio preparation. It employs a fixed, deterministic workflow for level balancing, gain staging, and output preparation, acting as a personal assistant for your mixing and mastering needs.

The project prioritizes experimentation and rapid iteration, making it ideal for creators working with raw multitrack material or AI-generated stems. Recent updates include a completely overhauled interface with a modern responsive layout, live drag-and-drop import, a one-click Mix+Master pipeline button, a larger activity log, integrated AI model downloading via the Model Browser Panel, and a dedicated Task Orchestrator to track operations seamlessly.

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

1. **Import Stems**: Drag and drop audio files directly onto the waveform area, or click the `Import` button on the Control Deck. Supported formats: WAV, AIFF, FLAC, MP3, OGG. A blue highlight confirms the drop zone is active.
2. **Download a Model**: Open the new Model Browser Panel to browse and download an AI model for stem separation or processing. The Task Orchestrator will show real-time download progress.
3. **Auto Mix**: Once your stems are loaded, click the **Auto Mix** button. The application will analyze the stems and apply consistent level balancing and spatial placement algorithms based on fixed rules.
4. **Auto Master**: After mixing, apply **Auto Master**. This applies automated gain staging, EQ, and peak limiting to finalize the track for streaming platforms like Spotify or YouTube.
5. **One-click pipeline**: Skip steps 3–5 by clicking **Mix + Master** (or pressing `Ctrl+Shift+M`). You will be prompted to select an output folder; the application then runs Auto Mix → Auto Master → Export automatically and writes a timestamped file.
6. **Analyze and Export**: Monitor the GlowMeters to ensure LUFS and true peak levels match your target. Use **Export** (`Ctrl+E`) for manual control over the output file name and format.

---

## Batch Processing

AutoMixMaster isn't limited to a single track. You can process an entire catalog using **Batch Mode**:

1. Select **Batch** from the main dashboard.
2. Choose a source folder containing your stems or full tracks.
3. Apply a shared profile or preset. The Task Orchestrator will manage the queue asynchronously.
4. The system groups stems into songs by filename role patterns (for example `_vocals`, `_bass`, `(drums)`, `(guitar)`), then renders **one mastered song per grouped set**.

**Recursive folder support:** use the **Recursive Batch** toggle in the Control Deck to include subfolders during batch discovery.  
For headless/advanced runs, `AUTOMIX_BATCH_RECURSIVE=1` is also supported.

---

## Feature Set

| Module | Description |
| :--- | :--- |
| **Auto Mix** | Rapid algorithmic rules for level balancing and spatial placement. |
| **Auto Master** | Automated gain staging and peak limiting tailored for final output goals. |
| **Mix + Master Pipeline** | One-click button (`Ctrl+Shift+M`) runs Auto Mix → Auto Master → Export to a chosen folder. |
| **Live Progress Tracking** | Task Center shows real-time progress bars and percentages for active operations. |
| **Drag and Drop** | Drop WAV, AIFF, FLAC, MP3, or OGG files directly onto the waveform area to import. |
| **Task Orchestrator** | Real-time monitoring of asynchronous tasks with timestamped activity log. |
| **Model Hub** | Built-in browser to download, manage, and utilize Hugging Face AI models. |
| **Batch Mode** | Processes multiple tracks/folders sequentially with unified settings. |
| **Renderer Availability Guardrails** | Renderer selector only lists currently available integrations (unavailable entries are hidden). |
| **PhaseLimiter Integration** | Supports bundled binaries and managed download/discovery flows for external rendering. |
| **Export Verification Report** | After export, a verification pass confirms mix/master processing and reports objective difference metrics. |
| **Batch Verification Summary** | Batch completion triggers a summary report confirming processed outputs and average mix/master deltas. |
| **Analysis Tools** | Real-time GlowMeters exposing LUFS and peak measurements for visual aid. |

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
  libxrender-dev libwebkit2gtk-4.0-dev libglu1-mesa-dev mesa-common-dev
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

### CI Release Packaging

On each published GitHub Release, CI automatically builds and uploads:

- `automixmaster_<version>_amd64.deb` and `automixmaster_<version>_arm64.deb`
- `AutoMixMaster-<version>-x86_64.AppImage` and `AutoMixMaster-<version>-aarch64.AppImage`
- `AutoMixMaster-linux-x64.flatpak` and `AutoMixMaster-linux-arm64.flatpak`
- `AutoMixMaster-windows-x64.zip` and `AutoMixMaster-windows-arm64.zip`
- `AutoMixMaster-macos-x64.zip` and `AutoMixMaster-macos-arm64.zip`

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

<div align="center">
  <br />
  <em>[ THIS APPLICATION IS A WORK IN PROGRESS ]</em>
</div>
