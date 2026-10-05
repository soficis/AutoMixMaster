#!/usr/bin/env python3
"""Regenerate assets/lame-pins.json, the list of LAME downloads the app trusts.

The app fetches that file from the default branch before downloading an MP3
encoder, so merging a change to it updates every installed copy without a
release. Run by .github/workflows/lame_pins.yml, which opens a pull request
when the output changes; a person reviews it before it is merged.

Where each hash comes from:
  - Debian: the SHA256 field of the `lame` entry in Debian stable's package index.
  - Homebrew: the bottle digests in the GHCR image index for the current version.
  - rarewares.org (Windows): no index or published hash exists, so the known
    files are downloaded and hashed. A changed hash there has no independent
    confirmation and needs a careful look before merging.

A source that cannot be refreshed keeps its current entry and is reported.
"""

import hashlib
import json
import lzma
import pathlib
import sys
import urllib.request

MANIFEST = pathlib.Path(__file__).resolve().parent.parent / "assets" / "lame-pins.json"
GHCR = "https://ghcr.io/v2/homebrew/core/lame"
DEBIAN = "https://deb.debian.org/debian/"

# platform key -> Debian architecture / Homebrew (os, architecture) / rarewares file
DEBIAN_ARCH = {"linux-x64": "amd64", "linux-arm64": "arm64", "linux-arm": "armhf"}
BOTTLE = {
    "linux-x64": ("linux", "amd64"),
    "linux-arm64": ("linux", "arm64"),
    "darwin-x64": ("darwin", "amd64"),
    "darwin-arm64": ("darwin", "arm64"),
}
WINDOWS_ZIP = {
    "win32-x64": "https://www.rarewares.org/files/mp3/lame3.100.1-x64.zip",
    "win32-arm64": "https://www.rarewares.org/files/mp3/lame3.100.1-x64.zip",
    "win32-ia32": "https://www.rarewares.org/files/mp3/lame3.100.1-win32.zip",
}


def fetch(url, headers=None):
    request = urllib.request.Request(url, headers={"User-Agent": "automix-lame-pins", **(headers or {})})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def debian_sources():
    sources = []
    for platform, arch in DEBIAN_ARCH.items():
        index = lzma.decompress(fetch(f"{DEBIAN}dists/stable/main/binary-{arch}/Packages.xz")).decode("utf-8")
        stanza = next(s for s in index.split("\n\n") if s.startswith("Package: lame\n"))
        fields = dict(line.split(": ", 1) for line in stanza.splitlines() if ": " in line and not line.startswith(" "))
        sources.append({"platform": platform, "type": "deb", "url": DEBIAN + fields["Filename"], "sha256": fields["SHA256"]})
    return sources


def os_version(manifest):
    digits = "".join(c if c.isdigit() or c == "." else " " for c in manifest["platform"].get("os.version", "0"))
    return tuple(int(part) for part in digits.split()[0].split(".") if part) if digits.split() else (0,)


def homebrew_sources():
    version = json.loads(fetch("https://formulae.brew.sh/api/formula/lame.json"))["versions"]["stable"]
    token = json.loads(fetch("https://ghcr.io/token?service=ghcr.io&scope=repository:homebrew/core/lame:pull"))["token"]
    auth = {"Authorization": f"Bearer {token}"}
    index = json.loads(fetch(f"{GHCR}/manifests/{version}", {**auth, "Accept": "application/vnd.oci.image.index.v1+json"}))

    sources = []
    for platform, (os_name, arch) in BOTTLE.items():
        bottles = [m for m in index["manifests"]
                   if m["platform"]["os"] == os_name and m["platform"]["architecture"] == arch]
        # The bottle built for the oldest OS release runs on the widest range of machines.
        bottle = min(bottles, key=os_version)
        sources.append({"platform": platform, "type": "ghcr", "sha256": bottle["annotations"]["sh.brew.bottle.digest"]})
    return sources


def windows_sources(current):
    sources = []
    hashes = {}
    for platform, default_url in WINDOWS_ZIP.items():
        url = next((s["url"] for s in current if s["platform"] == platform and s["type"] == "zip"), default_url)
        if url not in hashes:
            hashes[url] = hashlib.sha256(fetch(url)).hexdigest()
        sources.append({"platform": platform, "type": "zip", "url": url, "sha256": hashes[url]})
    return sources


def main():
    current = json.loads(MANIFEST.read_text(encoding="utf-8"))["sources"] if MANIFEST.exists() else []
    sources = []
    failed = False
    for name, kind, refresh in (("Debian", "deb", debian_sources),
                                ("Homebrew", "ghcr", homebrew_sources),
                                ("rarewares.org", "zip", lambda: windows_sources(current))):
        try:
            sources += refresh()
        except Exception as error:  # keep the entries we already trust
            failed = True
            print(f"warning: could not refresh {name} pins ({error}); keeping the current ones", file=sys.stderr)
            sources += [s for s in current if s["type"] == kind]

    for old in current:
        new = next((s for s in sources if (s["platform"], s["type"]) == (old["platform"], old["type"])), None)
        if new is not None and new["sha256"] != old["sha256"]:
            note = " -- NO independent confirmation, review before merging" if old["type"] == "zip" else ""
            print(f"changed: {old['platform']} {old['type']} {old['sha256'][:12]} -> {new['sha256'][:12]}{note}")

    sources.sort(key=lambda s: (s["platform"], s["type"]))
    MANIFEST.write_text(json.dumps({"schema": 1, "sources": sources}, indent=2) + "\n", encoding="utf-8", newline="\n")
    return 1 if failed and not sources else 0


if __name__ == "__main__":
    sys.exit(main())
