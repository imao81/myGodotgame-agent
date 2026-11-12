"""tests/test_sandbox_wiring.py — sandbox-wired auto_fix flow.

Covers the gate-forward wiring: auto_fix_prepare generates and gate-verifies
fixes inside a sandbox, human_gate shows the evidence, auto_fix merges from
the sandbox only after approval. Real git fixtures; no LLM, no Godot, no network.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent))

from game_agent_pipeline import (
    _auto_fix_apply_from_sandbox,
    _default_state,
    auto_fix_node,
    auto_fix_prepare_node,
    build_graph,
    human_gate_node,
)
from tools.sandbox import GateReport

GREEN_GATES = GateReport(syntax_ok=True, assertions_resolved=True, regression_ok=True)


def _git_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "a.gd").write_text("extends Node2D\nvar speed = 100\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    subprocess.run(["git", "add", "."], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=path, capture_output=True, check=True)
    return path


def _state_with_error(repo: Path) -> dict:
    state = _default_state(str(repo), no_display=True)
    state["deduplicated_findings"] = [{
        "file": "a.gd", "line": 2, "rule": "test_rule", "message": "broken",
        "final_severity": "error",
        "assertion": {"type": "identifier_not_defined", "identifier": "x"},
    }]
    return state


class TestAutoFixPrepare(unittest.TestCase):
    def test_prepare_generates_candidates_in_sandbox(self):
        with mock.patch("agents.auto_fix._generate_fixed_code",
                        return_value="extends Node2D\nvar speed = 200\n"), \
             mock.patch("tools.sandbox.Sandbox.run_gates", return_value=GREEN_GATES), \
             mock.patch("agents.llm_client.LLMClient", return_value=mock.Mock()):
            with _tmpdir_repo() as repo:
                state = _state_with_error(repo)
                result = auto_fix_prepare_node(state)

                self.assertEqual(result["agent_status"]["auto_fix_prepare"], "ok")
                self.assertEqual(result["fix_candidates"][0]["status"], "ready")
                self.assertTrue(result["fix_gate_report"]["ok"])
                # sandbox holds the fix; the real tree is untouched
                sandbox_file = Path(result["sandbox_path"]) / "a.gd"
                self.assertIn("speed = 200", sandbox_file.read_text())
                self.assertIn("speed = 100", (repo / "a.gd").read_text())
                # cleanup
                from tools.sandbox import Sandbox
                Sandbox.discard_at(repo, result["sandbox_path"])

    def test_prepare_sandbox_unavailable_falls_back(self):
        from tools.sandbox import SandboxError
        with mock.patch("tools.sandbox.Sandbox.create",
                        side_effect=SandboxError("boom")):
            with _tmpdir_repo() as repo:
                result = auto_fix_prepare_node(_state_with_error(repo))
        self.assertEqual(result["sandbox_path"], "")
        self.assertIn("sandbox_unavailable",
                      result["agent_status"]["auto_fix_prepare"])

    def test_prepare_no_errors_skips(self):
        state = _default_state("/nonexistent", no_display=True)
        result = auto_fix_prepare_node(state)
        self.assertEqual(result["agent_status"]["auto_fix_prepare"],
                         "skipped:no_errors")


class TestHumanGateEvidence(unittest.TestCase):
    def test_gate_payload_carries_sandbox_evidence(self):
        state = _default_state("/nonexistent", no_display=True)
        state["deduplicated_findings"] = [{
            "file": "a.gd", "line": 1, "final_severity": "error",
        }]
        state["fix_gate_report"] = {"ok": True, "syntax_ok": True}
        state["fix_candidates"] = [{"file": "a.gd", "status": "ready"}]

        with mock.patch("game_agent_pipeline.interrupt", return_value=True) as intr:
            result = human_gate_node(state)

        payload = intr.call_args.args[0]
        self.assertEqual(payload["sandbox_gates"], {"ok": True, "syntax_ok": True})
        self.assertEqual(payload["fix_candidates"], [{"file": "a.gd", "status": "ready"}])
        self.assertTrue(result["auto_fix_approved"])


class TestApplyFromSandbox(unittest.TestCase):
    def _prepare_sandbox(self, repo: Path) -> str:
        """Real sandbox with a modified a.gd, ready to merge."""
        from tools.sandbox import Sandbox
        sb = Sandbox(repo)
        sb.create()
        (Path(sb.path) / "a.gd").write_text(
            "extends Node2D\nvar speed = 200\n", encoding="utf-8")
        return sb.path

    def test_apply_merges_and_cleans_up(self):
        with _tmpdir_repo() as repo:
            sandbox_path = self._prepare_sandbox(repo)
            state = _state_with_error(repo)
            state["auto_fix_approved"] = True
            state["sandbox_path"] = sandbox_path
            state["fix_candidates"] = [{"file": "a.gd", "status": "ready", "issues": 1}]

            with mock.patch("agents.auto_fix._validate_gd_file", return_value=True):
                result = auto_fix_node(state)

            self.assertEqual(result["fix_results"][0]["status"], "fixed")
            self.assertTrue(result["fix_results"][0]["sandbox_verified"])
            self.assertIn("speed = 200", (repo / "a.gd").read_text())
            self.assertFalse(Path(sandbox_path).exists())

    def test_apply_restores_on_failed_validation(self):
        with _tmpdir_repo() as repo:
            sandbox_path = self._prepare_sandbox(repo)
            state = _state_with_error(repo)
            state["auto_fix_approved"] = True
            state["sandbox_path"] = sandbox_path
            state["fix_candidates"] = [{"file": "a.gd", "status": "ready", "issues": 1}]

            with mock.patch("agents.auto_fix._validate_gd_file", return_value=False):
                result = auto_fix_node(state)

            self.assertEqual(result["fix_results"][0]["status"], "skipped")
            self.assertIn("speed = 100", (repo / "a.gd").read_text())

    def test_legacy_fallback_when_sandbox_missing(self):
        with _tmpdir_repo() as repo:
            state = _state_with_error(repo)
            state["auto_fix_approved"] = True
            state["sandbox_path"] = str(repo / "gone")
            state["fix_candidates"] = [{"file": "a.gd", "status": "ready"}]

            fake = {"fix_results": [{"file": "a.gd", "status": "fixed"}],
                    "fix_summary": "legacy", "agent_status": {"auto_fix": "ok"}}
            with mock.patch("agents.auto_fix.run", return_value=fake) as m:
                result = auto_fix_node(state)
            m.assert_called_once()
            self.assertEqual(result["fix_results"][0]["status"], "fixed")


class TestGraphWiring(unittest.TestCase):
    def test_auto_fix_graph_contains_prepare_node(self):
        app = build_graph(with_auto_fix=True)
        node_names = set(app.get_graph().nodes)
        self.assertIn("auto_fix_prepare", node_names)
        self.assertIn("human_gate", node_names)


from contextlib import contextmanager
import shutil
import tempfile


@contextmanager
def _tmpdir_repo():
    d = Path(tempfile.mkdtemp()) / "proj"
    try:
        yield _git_repo(d)
    finally:
        # read-only git objects on Windows need a writable flag to delete
        shutil.rmtree(d.parent, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
