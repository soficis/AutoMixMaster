<div align="center">

```
 ______________________________________________________________________________
 [ SYSTEM: AUTOMIXMASTER ] [ VERSION: 0.2.0 ] [ STATUS: OPERATIONAL ]
 ______________________________________________________________________________

                  _        __  __ _      __  __           _
       /\        | |      |  \/  (_)    |  \/  |         | |
      /  \  _   _| |_ ___ | \  / |_  _ _| \  / | __ _ ___| |_ ___ _ __
     / /\ \| | | | __/ _ \| |\/| | \ \/ / |\/| |/ _` / __| __/ _ \ '__|
    / ____ \ |_| | || (_) | |  | | |>  <| |  | | (_| \__ \ ||  __/ |
   /_/    \_\__,_|\__\___/|_|  |_|_/_/\_\_|  |_|\__,_|___/\__\___|_|

 ------------------------------------------------------------------------------
```

<img src="assets/AutoMixMaster.jpg" alt="AutoMixMaster application interface" width="860">

```
 -------- FIXED-RULE AUDIO WORKFLOW FOR MIXING AND MASTERING MUSIC STEMS --------
```

**Designed for amateur music producers and hobbyists**

</div>

<p align="center">
<a href="#overview">Overview</a> •
<a href="#workflows">Workflows</a> •
<a href="#feature-set">Feature Set</a> •
<a href="#build--install">Build + Install</a> •
<a href="#licensing">Licensing</a>
</p>

---

## Overview

AutoMixMaster is an automation utility for repetitive stem-level audio prep.
It uses a fixed, deterministic workflow for level balancing, gain staging, and output preparation.

The project is aimed at personal experimentation and iteration speed, especially for creators working with raw multitrack material or AI-generated stems.

---

## Workflows

| Workflow | What it does | Typical use |
| :--- | :--- | :--- |
| AI seed processing | Applies consistent level balancing to separated stems | Quick listening passes on Udio/Suno-style stems |
| Songwriter prototyping | Moves raw multitrack recordings toward a reference balance | Faster idea review without manual fader passes |
| Bulk folder processing | Processes entire directories to a configured loudness target | Keeping catalog loudness consistent |

---

## Feature Set

| Module | Description |
| :--- | :--- |
| Auto Mix | Applies fixed algorithmic rules for level balancing and basic spatial placement |
| Auto Master | Applies automated gain staging and peak limiting to final output |
| Batch Mode | Processes multiple tracks/folders sequentially with shared settings |
| Analysis Tools | Exposes LUFS and peak measurements for visual monitoring |

---

## Build + Install

### Windows (Visual Studio 2026)

1. Configure

```bash
cmake -S . -B build -G "Visual Studio 18 2026" -A x64
```

2. Build

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

2. Configure + build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

3. Run tests (optional but recommended)

```bash
ctest --test-dir build --output-on-failure
```

---

## Licensing

AutoMixMaster is distributed under **GNU General Public License v3 (GPLv3)**.

| Component | License | Role |
| :--- | :--- | :--- |
| JUCE 8.0.8 | AGPLv3 / Commercial | Framework |
| libebur128 | MIT | Metering |
| nlohmann/json | MIT | Metadata |
| Catch2 3.7.1 | BSL-1.0 | Testing |
| PhaseLimiter | GPL / Custom | Limiting |

<div align="center">

```
 ______________________________________________________________________________
                     [ THIS APPLICATION IS A WORK IN PROGRESS ]
 ______________________________________________________________________________
```

</div>
