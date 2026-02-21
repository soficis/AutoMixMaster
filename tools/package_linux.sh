#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build_linux_release}"
DIST_DIR="${DIST_DIR:-$REPO_ROOT/dist/linux}"
APP_NAME="AutoMixMaster"
PACKAGE_NAME="automixmaster"

BUILD_DEB=1
BUILD_APPIMAGE=1
SKIP_BUILD=0

usage() {
  cat <<USAGE
Usage: tools/package_linux.sh [options]

Options:
  --skip-build           Reuse existing build output.
  --deb-only             Build only the .deb package.
  --appimage-only        Build only the AppImage package.
  --build-dir <path>     Override build directory (default: build_linux_release).
  --dist-dir <path>      Override output directory (default: dist/linux).
  -h, --help             Show this help text.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-build)
      SKIP_BUILD=1
      ;;
    --deb-only)
      BUILD_DEB=1
      BUILD_APPIMAGE=0
      ;;
    --appimage-only)
      BUILD_DEB=0
      BUILD_APPIMAGE=1
      ;;
    --build-dir)
      BUILD_DIR="$2"
      shift
      ;;
    --dist-dir)
      DIST_DIR="$2"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

if [[ $BUILD_DEB -eq 0 && $BUILD_APPIMAGE -eq 0 ]]; then
  echo "Nothing to do: both .deb and AppImage builds are disabled." >&2
  exit 1
fi

need_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Required command not found: $cmd" >&2
    exit 1
  fi
}

need_cmd cmake
need_cmd cp
need_cmd install
need_cmd awk
need_cmd sed
need_cmd find
need_cmd grep
need_cmd sha256sum

if [[ $BUILD_DEB -eq 1 ]]; then
  need_cmd dpkg-deb
  need_cmd fakeroot
fi

if [[ $BUILD_APPIMAGE -eq 1 ]]; then
  need_cmd curl
fi

if [[ $SKIP_BUILD -eq 0 ]]; then
  cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_TOOLS=OFF
  cmake --build "$BUILD_DIR" --config Release --target AutoMixMasterApp --parallel
fi

BINARY_PATH="$BUILD_DIR/AutoMixMasterApp_artefacts/Release/$APP_NAME"
SOURCE_ASSETS_PATH="$REPO_ROOT/assets"

if [[ ! -x "$BINARY_PATH" ]]; then
  echo "Expected executable not found: $BINARY_PATH" >&2
  exit 1
fi
if [[ ! -d "$SOURCE_ASSETS_PATH" ]]; then
  echo "Expected assets directory not found: $SOURCE_ASSETS_PATH" >&2
  exit 1
fi

VERSION=""
if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  VERSION="$(awk -F= '/^CMAKE_PROJECT_VERSION:STATIC=/{print $2; exit}' "$BUILD_DIR/CMakeCache.txt" || true)"
fi
if [[ -z "$VERSION" ]]; then
  VERSION="$(sed -n 's/^project(AutoMixMaster VERSION \([0-9][0-9.]*\).*/\1/p' "$REPO_ROOT/CMakeLists.txt" | head -n1)"
fi
if [[ -z "$VERSION" ]]; then
  VERSION="0.0.0"
fi

mkdir -p "$DIST_DIR"

create_desktop_file() {
  local output="$1"
  local exec_name="$2"
  local icon_name="$3"
  cat > "$output" <<DESKTOP
[Desktop Entry]
Type=Application
Name=AutoMixMaster
Comment=Deterministic auto mix and mastering utility
Exec=${exec_name}
Icon=${icon_name}
Terminal=false
Categories=AudioVideo;Audio;Music;
StartupNotify=true
DESKTOP
}

sanitize_linux_assets() {
  local app_root="$1"
  local phase_bin_dir="$app_root/assets/phaselimiter/bin"
  if [[ ! -d "$phase_bin_dir" ]]; then
    return
  fi

  find "$phase_bin_dir" -type f \( -iname '*.dll' -o -iname '*.exe' \) -delete

  if ! find "$phase_bin_dir" -maxdepth 1 -type f \( -name 'phase_limiter' -o -name 'phaselimiter' -o -name 'phase_limiter.bin' -o -name 'phaselimiter.bin' \) | grep -q .; then
    rm -rf "$phase_bin_dir"
  fi
}

build_deb() {
  local deb_arch
  deb_arch="$(dpkg --print-architecture)"

  local stage_dir
  stage_dir="$(mktemp -d "${TMPDIR:-/tmp}/automix-deb.XXXXXX")"
  local app_root="$stage_dir/opt/$PACKAGE_NAME"
  mkdir -p "$stage_dir/DEBIAN" "$app_root" "$stage_dir/usr/bin" "$stage_dir/usr/share/applications" "$stage_dir/usr/share/icons/hicolor/scalable/apps"

  install -Dm755 "$BINARY_PATH" "$app_root/$APP_NAME"
  cp -a "$SOURCE_ASSETS_PATH" "$app_root/assets"
  sanitize_linux_assets "$app_root"

  install -Dm755 /dev/null "$stage_dir/usr/bin/$PACKAGE_NAME"
  cat > "$stage_dir/usr/bin/$PACKAGE_NAME" <<'WRAPPER'
#!/bin/sh
cd /opt/automixmaster || exit 1
exec ./AutoMixMaster "$@"
WRAPPER
  chmod 0755 "$stage_dir/usr/bin/$PACKAGE_NAME"

  install -Dm644 "$REPO_ROOT/packaging/linux/automixmaster.svg" "$stage_dir/usr/share/icons/hicolor/scalable/apps/$PACKAGE_NAME.svg"
  create_desktop_file "$stage_dir/usr/share/applications/$PACKAGE_NAME.desktop" "$PACKAGE_NAME" "$PACKAGE_NAME"

  local installed_size
  installed_size="$(du -sk "$app_root" | awk '{print $1}')"

  cat > "$stage_dir/DEBIAN/control" <<CONTROL
Package: $PACKAGE_NAME
Version: $VERSION
Section: sound
Priority: optional
Architecture: $deb_arch
Maintainer: AutoMixMaster
Depends: libc6 (>= 2.31), libstdc++6 (>= 11), libgcc-s1, libasound2, libfontconfig1, libfreetype6, libexpat1, zlib1g, libbz2-1.0, libpng16-16, libbrotli1
Installed-Size: $installed_size
Description: Deterministic auto mix and mastering desktop app
 AutoMixMaster provides one-click mix/master workflows, batch processing,
 and verification reporting for audio files.
CONTROL

  local deb_output="$DIST_DIR/${PACKAGE_NAME}_${VERSION}_${deb_arch}.deb"
  fakeroot dpkg-deb --build "$stage_dir" "$deb_output" >/dev/null
  rm -rf "$stage_dir"
  echo "Built .deb: $deb_output"
}

appimage_arch() {
  local machine
  machine="$(uname -m)"
  case "$machine" in
    x86_64|amd64)
      echo "x86_64"
      ;;
    aarch64|arm64)
      echo "aarch64"
      ;;
    *)
      echo "Unsupported architecture for AppImage: $machine" >&2
      exit 1
      ;;
  esac
}

build_appimage() {
  local arch
  arch="$(appimage_arch)"

  local stage_dir
  stage_dir="$(mktemp -d "${TMPDIR:-/tmp}/automix-appdir.XXXXXX")"
  mkdir -p "$stage_dir/usr/bin" "$stage_dir/usr/share/applications" "$stage_dir/usr/share/icons/hicolor/scalable/apps"

  install -Dm755 "$BINARY_PATH" "$stage_dir/usr/bin/$APP_NAME"
  cp -a "$SOURCE_ASSETS_PATH" "$stage_dir/usr/bin/assets"
  sanitize_linux_assets "$stage_dir/usr/bin"

  install -Dm644 "$REPO_ROOT/packaging/linux/automixmaster.svg" "$stage_dir/$PACKAGE_NAME.svg"
  install -Dm644 "$REPO_ROOT/packaging/linux/automixmaster.svg" "$stage_dir/usr/share/icons/hicolor/scalable/apps/$PACKAGE_NAME.svg"

  create_desktop_file "$stage_dir/$PACKAGE_NAME.desktop" "$APP_NAME" "$PACKAGE_NAME"
  cp "$stage_dir/$PACKAGE_NAME.desktop" "$stage_dir/usr/share/applications/$PACKAGE_NAME.desktop"

  ln -sf "$PACKAGE_NAME.svg" "$stage_dir/.DirIcon"

  cat > "$stage_dir/AppRun" <<'APPRUN'
#!/bin/sh
set -eu
HERE="$(dirname "$(readlink -f "$0")")"
cd "$HERE/usr/bin" || exit 1
exec ./AutoMixMaster "$@"
APPRUN
  chmod 0755 "$stage_dir/AppRun"

  local tools_dir="$DIST_DIR/tools"
  mkdir -p "$tools_dir"
  local appimagetool="$tools_dir/appimagetool-${arch}.AppImage"

  # Pinned release of appimagetool (AppImage/appimagetool v1.9.1, 2025-11-18).
  # Update APPIMAGETOOL_VERSION and the matching SHA-256 entries when upgrading.
  local APPIMAGETOOL_VERSION="1.9.1"
  local appimagetool_url="https://github.com/AppImage/appimagetool/releases/download/${APPIMAGETOOL_VERSION}/appimagetool-${arch}.AppImage"

  # Expected SHA-256 checksums for each supported architecture.
  local expected_sha256
  case "$arch" in
    x86_64)  expected_sha256="ed4ce84f0d9caff66f50bcca6ff6f35aae54ce8135408b3fa33abfc3cb384eb0" ;;
    aarch64) expected_sha256="f0837e7448a0c1e4e650a93bb3e85802546e60654ef287576f46c71c126a9158" ;;
    *)
      echo "No known checksum for appimagetool on arch '${arch}'" >&2
      exit 1
      ;;
  esac

  if [[ ! -x "$appimagetool" ]]; then
    echo "Downloading appimagetool ${APPIMAGETOOL_VERSION} (${arch})..."
    curl -L --fail --retry 3 --output "$appimagetool" "$appimagetool_url"
    chmod 0755 "$appimagetool"
  fi

  echo "Verifying appimagetool checksum..."
  local actual_sha256
  actual_sha256="$(sha256sum "$appimagetool" | awk '{print $1}')"
  if [[ "$actual_sha256" != "$expected_sha256" ]]; then
    echo "Checksum mismatch for appimagetool-${arch}.AppImage" >&2
    echo "  expected: $expected_sha256" >&2
    echo "  got:      $actual_sha256" >&2
    rm -f "$appimagetool"
    exit 1
  fi

  local output="$DIST_DIR/${APP_NAME}-${VERSION}-${arch}.AppImage"
  APPIMAGE_EXTRACT_AND_RUN=1 ARCH="$arch" "$appimagetool" "$stage_dir" "$output" >/dev/null
  rm -rf "$stage_dir"
  chmod 0755 "$output"
  sha256sum "$output" > "$output.sha256"
  echo "Built AppImage: $output"
}

if [[ $BUILD_DEB -eq 1 ]]; then
  build_deb
fi

if [[ $BUILD_APPIMAGE -eq 1 ]]; then
  build_appimage
fi
