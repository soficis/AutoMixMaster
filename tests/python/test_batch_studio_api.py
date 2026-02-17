#!/usr/bin/env python3
"""Unit tests for batch_studio_api path-containment helpers."""

from __future__ import annotations

import importlib.util
import pathlib
import tempfile
import unittest


def _load_module():
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    module_path = repo_root / "tools" / "batch_studio_api.py"
    spec = importlib.util.spec_from_file_location("batch_studio_api", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BatchStudioApiPathContainmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module()

    def test_allows_root_and_descendants(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir) / "allowed"
            nested = root / "nested" / "folder"
            nested.mkdir(parents=True)

            self.assertTrue(self.module._is_path_contained(str(root), root))
            self.assertTrue(self.module._is_path_contained(str(nested), root))

    def test_rejects_path_escape_via_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = pathlib.Path(temp_dir)
            root = workspace / "allowed"
            outside = workspace / "outside"
            root.mkdir(parents=True)
            outside.mkdir(parents=True)

            escaped_path = root / ".." / "outside"
            self.assertFalse(self.module._is_path_contained(str(escaped_path), root))

    def test_allows_nonexistent_child_path_inside_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir) / "allowed"
            root.mkdir(parents=True)
            future_path = root / "future" / "file.json"

            self.assertTrue(self.module._is_path_contained(str(future_path), root))

    def test_rejects_invalid_path_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir) / "allowed"
            root.mkdir(parents=True)

            self.assertFalse(self.module._is_path_contained("bad\0path", root))


if __name__ == "__main__":
    unittest.main()
