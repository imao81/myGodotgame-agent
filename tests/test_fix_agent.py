"""tests/test_fix_agent.py — bounded fix loop with scripted fake clients."""

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

from agents.fix_agent import fix_file
from tools.sandbox import GateReport, Sandbox

GOOD_BLOCK = (
    "<<<<<<< ORIGINAL\n"
    "var speed = 100\n"
    "=======\n"
    "var speed = 200\n"
    ">>>>>>> FIXED\n"
)
BAD_BLOCK = GOOD_BLOCK.replace("var speed = 100", "var speed = 999")

GREEN = GateReport(syntax_ok=True, assertions_resolved=True, regression_ok=True)


class FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.last_usage = {"prompt_tokens": 10, "completion_tokens": 5}
        self.prompts: list[str] = []

    def chat(self, messages, max_tokens=1024):
        self.prompts.append(messages[0]["content"])
        if not self._responses:
            raise AssertionError("no scripted response left")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@contextmanager
def _tmpdir_repo():
    d = Path(tempfile.mkdtemp()) / "proj"
    d.mkdir(parents=True)
    (d / "a.gd").write_text(
        "extends Node2D\n\nvar speed = 100\nvar name_ = \"ship\"\n",
        encoding="utf-8")
    subprocess.run(["git", "init"], cwd=d, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=d, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=d, check=True)
    subprocess.run(["git", "add", "."], cwd=d, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=d, capture_output=True, check=True)
    try:
        yield d
    finally:
        shutil.rmtree(d.parent, ignore_errors=True)


ISSUES = [{"file": "a.gd", "line": 2, "rule": "test", "message": "speed wrong"}]


@mock.patch("tools.sandbox.Sandbox.run_gates", return_value=GREEN)
class TestFixLoop(unittest.TestCase):
    def test_success_round_one(self, _gates):
        with _tmpdir_repo() as repo:
            client = FakeClient([GOOD_BLOCK])
            out = fix_file("a.gd", ISSUES, repo, client=client)
            self.assertEqual(out["status"], "fixed")
            self.assertEqual(out["rounds_used"], 1)
            self.assertIn("speed = 200", out["fixed_code"])
            self.assertEqual(out["usage"]["prompt_tokens"], 10)
            # real tree untouched; sandbox kept for the caller
            self.assertIn("speed = 100", (repo / "a.gd").read_text())
            Sandbox.discard_at(repo, out["sandbox_path"])

    def test_feedback_loop_recovers(self, _gates):
        with _tmpdir_repo() as repo:
            client = FakeClient([BAD_BLOCK, GOOD_BLOCK])
            out = fix_file("a.gd", ISSUES, repo, client=client)
            self.assertEqual(out["status"], "fixed")
            self.assertEqual(out["rounds_used"], 2)
            # round-2 prompt carries round-1 failure feedback
            self.assertIn("not found", client.prompts[1])
            Sandbox.discard_at(repo, out["sandbox_path"])

    def test_exhaustion_is_honest_failure(self, _gates):
        with _tmpdir_repo() as repo:
            client = FakeClient([BAD_BLOCK] * 5)
            out = fix_file("a.gd", ISSUES, repo, client=client, max_rounds=3)
            self.assertEqual(out["status"], "failed")
            self.assertEqual(out["rounds_used"], 3)
            self.assertIn("exhausted", out["error"])
            self.assertNotIn("sandbox_path", out)  # sandbox discarded

    def test_gate_failure_feeds_back(self, _gates):
        # gates green by class-level mock; here override with one red round
        with _tmpdir_repo() as repo:
            red = GateReport(syntax_ok=False, details=["syntax: parse error"])
            with mock.patch("tools.sandbox.Sandbox.run_gates",
                            side_effect=[red, GREEN]):
                client = FakeClient([GOOD_BLOCK, GOOD_BLOCK])
                out = fix_file("a.gd", ISSUES, repo, client=client)
            self.assertEqual(out["status"], "fixed")
            self.assertEqual(out["rounds_used"], 2)
            self.assertIn("parse error", client.prompts[1])
            Sandbox.discard_at(repo, out["sandbox_path"])

    def test_llm_api_error_no_crash(self, _gates):
        with _tmpdir_repo() as repo:
            client = FakeClient([RuntimeError("api down")])
            out = fix_file("a.gd", ISSUES, repo, client=client)
            self.assertEqual(out["status"], "failed")
            self.assertIn("api down", out["error"])

    def test_whole_file_small_diff_accepted(self, _gates):
        with _tmpdir_repo() as repo:
            client = FakeClient(
                ["extends Node2D\n\nvar speed = 200\nvar name_ = \"ship\"\n"])
            out = fix_file("a.gd", ISSUES, repo, client=client)
            self.assertEqual(out["status"], "fixed")
            self.assertEqual(out["boundary"]["mode"], "whole_file")
            Sandbox.discard_at(repo, out["sandbox_path"])

    def test_whole_file_large_diff_rejected_then_blocks(self, _gates):
        with _tmpdir_repo() as repo:
            big = "\n".join(f"var x{i} = {i}" for i in range(20))
            client = FakeClient([big, GOOD_BLOCK])
            out = fix_file("a.gd", ISSUES, repo, client=client)
            self.assertEqual(out["status"], "fixed")
            self.assertEqual(out["rounds_used"], 2)
            self.assertIn("edit blocks", client.prompts[1])
            Sandbox.discard_at(repo, out["sandbox_path"])


if __name__ == "__main__":
    unittest.main()
