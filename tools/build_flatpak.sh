#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
MANIFEST="$REPO_ROOT/packaging/flatpak/io.automixmaster.AutoMixMaster.yml"
DIST_DIR="${DIST_DIR:-$REPO_ROOT/dist/flatpak}"
BUILD_DIR="$DIST_DIR/build-dir"
REPO_DIR="$DIST_DIR/repo"
APP_ID="io.automixmaster.AutoMixMaster"
FLATHUB_REMOTE_URL="https://flathub.org/repo/flathub.flatpakrepo"

if ! command -v flatpak-builder >/dev/null 2>&1; then
  echo "flatpak-builder is required (sudo apt install flatpak-builder flatpak)." >&2
  exit 1
fi
if ! command -v flatpak >/dev/null 2>&1; then
  echo "flatpak is required (sudo apt install flatpak)." >&2
  exit 1
fi

mkdir -p "$DIST_DIR"

if ! flatpak remote-list --user --columns=name | grep -qx "flathub"; then
  flatpak remote-add --user --if-not-exists flathub "$FLATHUB_REMOTE_URL"
fi

if ! flatpak remote-ls --user flathub 2>/dev/null | grep -q "org.freedesktop.Platform"; then
  flatpak remote-delete --user -f flathub >/dev/null 2>&1 || true
  flatpak remote-add --user --if-not-exists flathub "$FLATHUB_REMOTE_URL"
fi

EXTRA_ARGS=()
if flatpak-builder --help 2>&1 | grep -q -- "--disable-rofiles-fuse"; then
  EXTRA_ARGS+=(--disable-rofiles-fuse)
fi
if [[ "${AUTOMIX_FLATPAK_DISABLE_SANDBOX:-0}" == "1" ]]; then
  if flatpak-builder --help 2>&1 | grep -q -- "--disable-sandbox"; then
    EXTRA_ARGS+=(--disable-sandbox)
  else
    echo "Warning: this flatpak-builder does not support --disable-sandbox; continuing without it."
  fi
fi

flatpak-builder \
  --user \
  --force-clean \
  --repo="$REPO_DIR" \
  --install-deps-from=flathub \
  "${EXTRA_ARGS[@]}" \
  "$BUILD_DIR" \
  "$MANIFEST"

BUNDLE_PATH="$DIST_DIR/AutoMixMaster.flatpak"
flatpak build-bundle "$REPO_DIR" "$BUNDLE_PATH" "$APP_ID"

echo "Built Flatpak bundle: $BUNDLE_PATH"
