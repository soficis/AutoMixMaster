#!/usr/bin/env bash
set -euo pipefail

APP_BUNDLE="${1:-}"
if [[ -z "$APP_BUNDLE" || ! -d "$APP_BUNDLE" ]]; then
  echo "Usage: $0 <path-to-AutoMixMaster.app>" >&2
  exit 1
fi

FRAMEWORKS_DIR="$APP_BUNDLE/Contents/Frameworks"
if [[ -d "$FRAMEWORKS_DIR" ]]; then
  echo "Re-signing bundled dylibs in $FRAMEWORKS_DIR..."
  find "$FRAMEWORKS_DIR" -type f \( -name "*.dylib" -o -name "*.dylib.*" \) | while read -r dylib; do
    echo "  Signing $dylib"
    codesign --force --sign - --options runtime "$dylib"
  done
else
  echo "No Frameworks directory found in $APP_BUNDLE; skipping dylib signing."
fi
