import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "tools" / "check_repo_hygiene.py"


class TestCheckRepoHygiene(unittest.TestCase):
    def run_hygiene(self, args):
        cmd = [sys.executable, str(SCRIPT_PATH)] + args
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        return proc.returncode, proc.stdout

    def test_mask_detected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wf_path = Path(tmpdir) / "test_mask.yml"
            wf_path.write_text("name: Test\njobs:\n  build:\n    steps:\n      - run: ctest || true\n")
            rc, out = self.run_hygiene(["--check-workflows", "--workflows-dir", tmpdir])
            self.assertEqual(rc, 1, f"Expected rc 1, got {rc}. Output:\n{out}")
            self.assertIn("MASK:", out)

    def test_unpinned_action_detected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wf_path = Path(tmpdir) / "test_unpinned.yml"
            wf_path.write_text("name: Test\njobs:\n  build:\n    steps:\n      - uses: actions/checkout@v4\n")
            rc, out = self.run_hygiene(["--check-workflows", "--workflows-dir", tmpdir])
            self.assertEqual(rc, 1, f"Expected rc 1, got {rc}. Output:\n{out}")
            self.assertIn("UNPINNED:", out)

    def test_pinned_action_ok(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wf_path = Path(tmpdir) / "test_pinned.yml"
            wf_path.write_text("name: Test\njobs:\n  build:\n    steps:\n      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4\n")
            rc, out = self.run_hygiene(["--check-workflows", "--workflows-dir", tmpdir])
            self.assertEqual(rc, 0, f"Expected rc 0, got {rc}. Output:\n{out}")

    def test_forbidden_paths(self):
        forbidden_samples = [
            ".superpowers/x",
            ".anchor/x",
            "memory/x.md",
            ".claude/x",
            ".gemini/x",
            ".opencode/x",
            ".codex/x",
            "GEMINI.md",
            ".cursorrules",
            "notes/SSH_HANDOFF.md",
            "x/handoff-2.md",
            "job-logs1.txt",
            "a/b.jsonl",
        ]
        allowed_samples = [
            "docs/model-licensing-audit.json",
            "README.md",
            "assets/sox/README.md",
            "tools/training/README.md",
        ]
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as f:
            for p in forbidden_samples + allowed_samples:
                f.write(p + "\n")
            list_file = f.name

        try:
            rc, out = self.run_hygiene(["--paths-from", list_file])
            self.assertEqual(rc, 1, f"Expected rc 1, got {rc}. Output:\n{out}")
            for p in forbidden_samples:
                self.assertIn(f"FORBIDDEN: {p}", out)
            for p in allowed_samples:
                self.assertNotIn(f"FORBIDDEN: {p}", out)
        finally:
            os.remove(list_file)


if __name__ == "__main__":
    unittest.main()
