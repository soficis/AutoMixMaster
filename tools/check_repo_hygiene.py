#!/usr/bin/env python3
"""
tools/check_repo_hygiene.py

Repository hygiene verification script.
Ensures no forbidden files, sensitive IP patterns, private tokens,
or developer notes are committed or staged in the repository.
"""

import argparse
import os
import re
import subprocess
import sys

# Paths/patterns that must NEVER exist in the repo
FORBIDDEN_PATH_PATTERNS = [
    re.compile(r"^diff_output\.txt$", re.IGNORECASE),
    re.compile(r"CLAUDE\.md$", re.IGNORECASE),
    re.compile(r"AGENTS\.md$", re.IGNORECASE),
    re.compile(r"^\.omo(/|$)", re.IGNORECASE),
    re.compile(r"^assets/phaselimiter/assets(/|$)", re.IGNORECASE),
    # Under docs/, ONLY model-licensing-audit.json is permitted
    re.compile(r"^docs/(?!model-licensing-audit\.json$).+", re.IGNORECASE),
]

# Sensitive content regex patterns
SENSITIVE_CONTENT_PATTERNS = [
    ("Private Key", re.compile(r"-----BEGIN (RSA|OPENSSH|EC|DSA) PRIVATE KEY-----")),
    ("GitHub Personal Access Token (classic)", re.compile(r"\bghp_[A-Za-z0-9]{36}\b")),
    ("GitHub Fine-Grained PAT", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{82}\b")),
    ("Slack Token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]+\b")),
    ("Local Network IP", re.compile(r"\b192\.168\.0\.\d{1,3}\b")),
    ("Windows User Path", re.compile(r"[C-Zc-z]:[/\\]Users[/\\](?!runneradmin|Default|Public)[A-Za-z0-9_]+")),
]

# Files/extensions exempt from binary or content inspection
EXEMPT_EXTENSIONS = {
    ".onnx", ".png", ".jpg", ".jpeg", ".ico", ".wav", ".mp3", ".zip", ".tar",
    ".gz", ".xz", ".exe", ".dll", ".lib", ".so", ".dylib", ".bin", ".dat"
}

EXEMPT_PATHS = {
    "tools/check_repo_hygiene.py",  # self-exempt for pattern literals
}


def get_git_files(staged_only=False):
    """Retrieve list of files from git."""
    cmd = ["git", "diff", "--cached", "--name-only"] if staged_only else ["git", "ls-files"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def check_path_hygiene(file_path):
    """Check if a path violates repository hygiene rules."""
    normalized = file_path.replace("\\", "/").lstrip("/")
    for pat in FORBIDDEN_PATH_PATTERNS:
        if pat.search(normalized):
            return f"Forbidden path pattern '{pat.pattern}' matched: {file_path}"
    return None


def check_file_content(file_path):
    """Scan a text file for forbidden content patterns."""
    normalized = file_path.replace("\\", "/")
    if normalized in EXEMPT_PATHS:
        return []

    _, ext = os.path.splitext(file_path)
    if ext.lower() in EXEMPT_EXTENSIONS:
        return []

    if not os.path.isfile(file_path):
        return []

    findings = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, start=1):
                for name, pat in SENSITIVE_CONTENT_PATTERNS:
                    if pat.search(line):
                        findings.append(f"{file_path}:{line_no}: detected {name}")
    except Exception as e:
        findings.append(f"{file_path}: unable to read file: {e}")

    return findings


def main():
    parser = argparse.ArgumentParser(description="Check repository hygiene and scan for sensitive content.")
    parser.add_argument("--staged", action="store_true", help="Scan only git staged files.")
    parser.add_argument("--path", nargs="*", help="Specific file paths to scan.")
    args = parser.parse_args()

    files = args.path if args.path else get_git_files(staged_only=args.staged)

    path_errors = []
    content_errors = []

    for file_path in files:
        err = check_path_hygiene(file_path)
        if err:
            path_errors.append(err)

        content_errs = check_file_content(file_path)
        content_errors.extend(content_errs)

    has_errors = False
    if path_errors:
        has_errors = True
        print("[ERROR] Hygiene: Forbidden paths found:", file=sys.stderr)
        for err in path_errors:
            print(f"  - {err}", file=sys.stderr)

    if content_errors:
        has_errors = True
        print("[ERROR] Hygiene: Sensitive patterns found in file contents:", file=sys.stderr)
        for err in content_errors:
            print(f"  - {err}", file=sys.stderr)

    if has_errors:
        sys.exit(1)

    print(f"[OK] Repository hygiene check passed: {len(files)} files scanned, 0 violations.")
    sys.exit(0)


if __name__ == "__main__":
    main()
