"""tests/test_ci_scope.py — CI base-ref scope semantics (P0 fixes).

- GIT_BASE_SHA set → three-dot PR diff, untracked/generator files excluded
- base ref + zero relevant changes → scope_mode "none" (never full review)
- no base ref + clean tree → scope_mode "full" (local semantics unchanged)
- code_review skips on scope_mode "none"
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent))

from game_agent_pipeline import (
    _default_state,
    _get_git_changed_files,
    code_review_node,
    resolve_scope_node,
)


@contextmanager
def _repo():
    d = Path(tempfile.mkdtemp()) / "game"
    d.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=d, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=d, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=d, check=True)
    (d / "project.godot").write_text("; engine config\n", encoding="utf-8")
    (d / "a.gd").write_text("extends Node\nvar x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=d, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=d, capture_output=True, check=True)
    try:
        yield d
    finally:
        shutil.rmtree(d.parent, ignore_errors=True)


def _commit(d: Path, rel: str, content: str, msg: str) -> str:
    (d / rel).write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", rel], cwd=d, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", msg], cwd=d, capture_output=True, check=True)
    return subprocess.run(["git", "rev-parse", "HEAD~1"], cwd=d,
                          capture_output=True, text=True).stdout.strip()


class TestBaseRefDiff(unittest.TestCase):
    def test_pr_diff_finds_changed_files(self):
        with _repo() as d:
            base = _commit(d, "b.gd", "extends Node\nvar y = 2\n", "pr work")
            changed = _get_git_changed_files(d, base_ref=base)
            self.assertIn("b.gd", changed)
            self.assertNotIn("a.gd", changed)

    def test_generated_files_excluded(self):
        with _repo() as d:
            (d / ".godot").mkdir()
            base = _commit(d, ".godot/cache.gd", "extends Node\n", "generated")
            changed = _get_git_changed_files(d, base_ref=base)
            self.assertNotIn(str(Path(".godot/cache.gd")), changed)
            self.assertEqual(changed, set())

    def test_import_files_excluded_by_suffix(self):
        with _repo() as d:
            base = _commit(d, "tex.png.import", "stub=1\n", "import artifact")
            changed = _get_git_changed_files(d, base_ref=base)
            self.assertEqual(changed, set())

    def test_base_ref_no_changes_means_none_not_full(self):
        with _repo() as d:
            base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=d,
                                  capture_output=True, text=True).stdout.strip()
            with mock.patch.dict("os.environ", {"GIT_BASE_SHA": base}):
                state = _default_state(str(d), no_display=True)
                result = resolve_scope_node(state)
            self.assertEqual(result["scope_mode"], "none")
            self.assertEqual(result["agent_status"]["resolve_scope"],
                             "ok:no_changes")

    def test_no_base_ref_clean_tree_means_full(self):
        with _repo() as d:
            with mock.patch.dict("os.environ", {}, clear=False):
                import os
                os.environ.pop("GIT_BASE_SHA", None)
                state = _default_state(str(d), no_display=True)
                result = resolve_scope_node(state)
            self.assertEqual(result["scope_mode"], "full")


class TestCodeReviewSkip(unittest.TestCase):
    def test_scope_none_skips_llm_review(self):
        state = _default_state("/nonexistent", no_display=True)
        state["scope_mode"] = "none"
        result = code_review_node(state)
        self.assertEqual(result["findings"], [])
        self.assertEqual(result["agent_status"]["code_review"],
                         "skipped:no_changes")


if __name__ == "__main__":
    unittest.main()
