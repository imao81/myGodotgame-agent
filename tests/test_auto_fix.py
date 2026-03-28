"""Unit tests for agents/auto_fix.py.

All external I/O is mocked:
  - agents.llm_client.LLMClient.chat (via patch.object)
  - agents.auto_fix._git_stash
  - agents.auto_fix._validate_gd_file

File reads/writes use a real temporary directory (tmpdir fixture).
"""

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
import os

from agents.llm_client import LLMClient


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_state(project_path: str, issues: list[dict]) -> dict:
    return {"project_path": project_path, "code_issues": issues}


def _make_issue(file: str, line: int = 10, severity: str = "error",
                rule: str = "uninit_var", message: str = "Uninitialized variable") -> dict:
    return {"file": file, "line": line, "severity": severity, "rule": rule, "message": message}


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestAutoFixDryRun(unittest.TestCase):
    """Dry-run mode must never write files, call LLM, or stash."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        gd = Path(self.tmpdir) / "player.gd"
        gd.write_text("extends Node\nvar speed = 0\n", encoding="utf-8")
        self.gd_path = "player.gd"

    def test_dry_run_prints_plan_no_writes(self):
        from agents.auto_fix import run

        issues = [_make_issue(self.gd_path)]
        state = _make_state(self.tmpdir, issues)

        with patch.object(LLMClient, "chat") as mock_chat, \
             patch("agents.auto_fix._git_stash") as mock_stash:

            result = run(state, apply=False)

        # LLM and stash must never be called in dry-run
        mock_chat.assert_not_called()
        mock_stash.assert_not_called()

        self.assertEqual(len(result["fix_results"]), 1)
        self.assertEqual(result["fix_results"][0]["status"], "would_fix")
        self.assertIn("Dry-run", result["fix_summary"])

    def test_dry_run_file_is_unchanged(self):
        from agents.auto_fix import run

        gd = Path(self.tmpdir) / "player.gd"
        original = gd.read_text(encoding="utf-8")
        issues = [_make_issue(self.gd_path)]
        state = _make_state(self.tmpdir, issues)

        with patch.object(LLMClient, "chat"), \
             patch("agents.auto_fix._git_stash"):
            run(state, apply=False)

        self.assertEqual(gd.read_text(encoding="utf-8"), original)


class TestAutoFixApply(unittest.TestCase):
    """Apply mode — happy path: LLM returns valid code, validation passes."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.gd = Path(self.tmpdir) / "player.gd"
        self.gd.write_text("extends Node\nvar speed  # uninit\n", encoding="utf-8")
        self.gd_path = "player.gd"

    def test_apply_writes_fixed_file_when_validation_passes(self):
        from agents.auto_fix import run

        fixed_code = "extends Node\nvar speed: float = 0.0\n"
        issues = [_make_issue(self.gd_path)]
        state = _make_state(self.tmpdir, issues)

        with patch.object(LLMClient, "chat", return_value=fixed_code), \
             patch("agents.auto_fix._git_stash", return_value=True), \
             patch("agents.auto_fix._validate_gd_file", return_value=True):

            result = run(state, apply=True)

        self.assertEqual(result["fix_results"][0]["status"], "fixed")
        self.assertEqual(self.gd.read_text(encoding="utf-8"), fixed_code)
        self.assertIn("1 file(s) fixed", result["fix_summary"])

    def test_apply_skips_file_when_validation_fails(self):
        from agents.auto_fix import run

        original_code = "extends Node\nvar speed  # uninit\n"
        self.gd.write_text(original_code, encoding="utf-8")
        issues = [_make_issue(self.gd_path)]
        state = _make_state(self.tmpdir, issues)

        with patch.object(LLMClient, "chat", return_value="invalid gdscript !!!"), \
             patch("agents.auto_fix._git_stash", return_value=True), \
             patch("agents.auto_fix._validate_gd_file", return_value=False):

            result = run(state, apply=True)

        self.assertEqual(self.gd.read_text(encoding="utf-8"), original_code)
        self.assertEqual(result["fix_results"][0]["status"], "skipped")
        self.assertIn("check-only failed", result["fix_results"][0]["reason"])

    def test_tmp_file_is_deleted_after_apply(self):
        """Temp file must be cleaned up regardless of validation outcome."""
        from agents.auto_fix import run

        issues = [_make_issue(self.gd_path)]
        state = _make_state(self.tmpdir, issues)

        with patch.object(LLMClient, "chat", return_value="bad code"), \
             patch("agents.auto_fix._git_stash", return_value=True), \
             patch("agents.auto_fix._validate_gd_file", return_value=False):

            run(state, apply=True)

        tmp_files = list(Path(self.tmpdir).glob("*__autofix_tmp.gd"))
        self.assertEqual(tmp_files, [])


class TestAutoFixEmptyInput(unittest.TestCase):
    """With no error issues, agent must do nothing (no stash, no API call)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_empty_issues_no_stash_no_api_call(self):
        from agents.auto_fix import run

        state = _make_state(self.tmpdir, [])

        with patch.object(LLMClient, "chat") as mock_chat, \
             patch("agents.auto_fix._git_stash") as mock_stash:

            result = run(state, apply=True)

        mock_chat.assert_not_called()
        mock_stash.assert_not_called()
        self.assertEqual(result["agent_status"]["auto_fix"], "nothing_to_fix")

    def test_warnings_only_not_fixed(self):
        """Warning-severity issues must be silently ignored."""
        from agents.auto_fix import run

        issues = [_make_issue("player.gd", severity="warning")]
        state = _make_state(self.tmpdir, issues)

        with patch("agents.auto_fix._git_stash") as mock_stash:
            result = run(state, apply=True)

        mock_stash.assert_not_called()
        self.assertEqual(result["agent_status"]["auto_fix"], "nothing_to_fix")


class TestAutoFixCodeFenceStripping(unittest.TestCase):
    """LLM sometimes wraps output in ```gdscript ... ```. Strip them."""

    def test_strips_code_fences_from_response(self):
        from agents.auto_fix import _strip_code_fences

        wrapped = "```gdscript\nextends Node\nvar x = 0\n```"
        self.assertEqual(_strip_code_fences(wrapped), "extends Node\nvar x = 0")

    def test_strips_plain_code_fence(self):
        from agents.auto_fix import _strip_code_fences

        wrapped = "```\nextends Node\n```"
        self.assertEqual(_strip_code_fences(wrapped), "extends Node")

    def test_passthrough_when_no_fence(self):
        from agents.auto_fix import _strip_code_fences

        plain = "extends Node\nvar x = 0"
        self.assertEqual(_strip_code_fences(plain), plain)


if __name__ == "__main__":
    unittest.main()
