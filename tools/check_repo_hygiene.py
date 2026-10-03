#!/usr/bin/env python3
"""
tools/check_repo_hygiene.py

Repository hygiene verification script.
Ensures no forbidden files, sensitive IP patterns, private tokens,
or developer notes are committed or staged in the repository.
Also verifies GitHub Actions workflow pinning and failure mask rules.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# Paths/patterns that must NEVER exist in the repo
FORBIDDEN_PATH_PATTERNS = [
    re.compile(r"^diff_output\.txt$", re.IGNORECASE),
    re.compile(r"CLAUDE\.md$", re.IGNORECASE),
    re.compile(r"AGENTS\.md$", re.IGNORECASE),
    re.compile(r"^\.omo(/|$)", re.IGNORECASE),
    re.compile(r"^assets/phaselimiter/assets(/|$)", re.IGNORECASE),
    # Under docs/, ONLY model-licensing-audit.json is permitted
    re.compile(r"^docs/(?!model-licensing-audit\.json$).+", re.IGNORECASE),
    # Extended patterns from Round 5
    re.compile(r"(^|/)\.superpowers(/|$)", re.IGNORECASE),
    re.compile(r"(^|/)\.anchor(/|$)", re.IGNORECASE),
    re.compile(r"(^|/)memory(/|$)", re.IGNORECASE),
    re.compile(r"(^|/)\.claude(/|$)", re.IGNORECASE),
    re.compile(r"(^|/)\.gemini(/|$)", re.IGNORECASE),
    re.compile(r"(^|/)\.opencode(/|$)", re.IGNORECASE),
    re.compile(r"(^|/)\.codex(/|$)", re.IGNORECASE),
    re.compile(r"(^|/)GEMINI\.md$", re.IGNORECASE),
    re.compile(r"(^|/)\.cursorrules$", re.IGNORECASE),
    re.compile(r"(^|/)[^/]*handoff[^/]*\.md$", re.IGNORECASE),
    re.compile(r"(^|/)job-logs.*\.txt$", re.IGNORECASE),
    re.compile(r"\.jsonl$", re.IGNORECASE),
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
    "tests/tools/test_check_repo_hygiene.py",
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
            return f"FORBIDDEN: {file_path} (matched '{pat.pattern}')"
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


def check_workflow_file(file_path):
    """Check a GitHub Actions workflow file for failure masks and unpinned actions."""
    findings = []
    normalized_path = os.path.normpath(file_path).replace("\\", "/")
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, start=1):
                # 1. Mask detection: || true, continue-on-error: true, if: false
                if (re.search(r"\|\|\s*true\b", line) or
                    re.search(r"continue-on-error:\s*true\b", line, re.IGNORECASE) or
                    re.search(r"^\s*if:\s*false\b", line, re.IGNORECASE)):
                    findings.append(f"MASK: {normalized_path}:{line_no}: {line.strip()}")

                # 2. Unpinned action detection: uses: owner/repo@ref
                m = re.search(r"uses:\s*([^\s#]+)", line)
                if m:
                    action_ref = m.group(1)
                    if not action_ref.startswith("./"):
                        if "@" in action_ref:
                            _, ref = action_ref.split("@", 1)
                            if not re.fullmatch(r"[0-9a-fA-F]{40}", ref):
                                findings.append(f"UNPINNED: {normalized_path}:{line_no}: {action_ref}")
                        else:
                            findings.append(f"UNPINNED: {normalized_path}:{line_no}: {action_ref}")
    except Exception as e:
        findings.append(f"ERROR: {normalized_path}: unable to read file: {e}")

    return findings


def check_workflows_dir(workflows_dir):
    """Scan all YAML workflow files in workflows_dir."""
    p = Path(workflows_dir)
    if not p.is_dir():
        return [f"ERROR: Workflows directory '{workflows_dir}' not found"]

    findings = []
    yaml_files = sorted(list(p.glob("*.yml")) + list(p.glob("*.yaml")))
    for yml in yaml_files:
        findings.extend(check_workflow_file(str(yml)))
    return findings


def main():
    parser = argparse.ArgumentParser(description="Check repository hygiene and scan for sensitive content.")
    parser.add_argument("--staged", action="store_true", help="Scan only git staged files.")
    parser.add_argument("--path", nargs="*", help="Specific file paths to scan.")
    parser.add_argument("--paths-from", type=str, help="File containing newline-separated paths to scan.")
    parser.add_argument("--check-workflows", action="store_true", help="Check workflows for failure masks and unpinned actions.")
    parser.add_argument("--workflows-dir", type=str, default=".github/workflows", help="Directory containing workflow files to check.")
    args = parser.parse_args()

    has_errors = False

    if args.check_workflows:
        wf_errors = check_workflows_dir(args.workflows_dir)
        if wf_errors:
            has_errors = True
            for err in wf_errors:
                print(err)

    files_to_check = None
    if args.paths_from:
        with open(args.paths_from, "r", encoding="utf-8") as f:
            files_to_check = [line.strip() for line in f if line.strip()]
    elif args.path:
        files_to_check = args.path
    elif not args.check_workflows:
        files_to_check = get_git_files(staged_only=args.staged)

    if files_to_check is not None:
        path_errors = []
        content_errors = []

        for file_path in files_to_check:
            err = check_path_hygiene(file_path)
            if err:
                path_errors.append(err)

            content_errs = check_file_content(file_path)
            content_errors.extend(content_errs)

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

    if args.check_workflows and files_to_check is None:
        print("[OK] Workflow hygiene check passed: 0 masks, 0 unpinned actions.")
    else:
        print(f"[OK] Repository hygiene check passed: {len(files_to_check or [])} files scanned, 0 violations.")
    sys.exit(0)


if __name__ == "__main__":
    main()
