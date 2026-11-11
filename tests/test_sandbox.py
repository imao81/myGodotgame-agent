"""Tests for tools/sandbox.py — real git fixtures, zero Godot/LLM/network."""

import subprocess
from pathlib import Path

import pytest

from tools.sandbox import GateReport, Sandbox, SandboxError

GOOD_DIFF = """\
diff --git a/player.gd b/player.gd
--- a/player.gd
+++ b/player.gd
@@ -1,2 +1,2 @@
 extends Node2D
-var speed = 100
+var speed = 200
"""

CONFLICT_DIFF = """\
diff --git a/player.gd b/player.gd
--- a/player.gd
+++ b/player.gd
@@ -1,2 +1,2 @@
 extends Node2D
-var speed = 999
+var speed = 500
"""


def _git_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "player.gd").write_text("extends Node2D\nvar speed = 100\n", encoding="utf-8")
    (path / "enemies").mkdir()
    (path / "enemies" / "mine.gd").write_text("extends Area2D\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "add", "."], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-m", "init"], cwd=path, capture_output=True, check=True)
    return path


@pytest.fixture
def repo(tmp_path):
    return _git_repo(tmp_path / "proj")


@pytest.fixture
def sandbox_dir(tmp_path):
    return tmp_path / "sandboxes"


class TestLifecycle:
    def test_worktree_create_apply_promote_discard(self, repo, sandbox_dir):
        sb = Sandbox(repo, sandbox_dir)
        sb.create()
        assert sb.mode == "worktree"
        assert (sb.path / "player.gd").exists()
        assert sb.apply_patch(GOOD_DIFF) is True
        assert "speed = 200" in (sb.path / "player.gd").read_text()
        artifact = sb.promote()
        assert "speed = 200" in artifact.read_text(encoding="utf-8")
        sb.discard()
        assert not sb.path.exists()
        # real working tree untouched throughout
        assert "speed = 100" in (repo / "player.gd").read_text()

    def test_conflict_rejected_without_pollution(self, repo, sandbox_dir):
        with Sandbox(repo, sandbox_dir) as sb:
            assert sb.apply_patch(CONFLICT_DIFF) is False
            assert "speed = 100" in (sb.path / "player.gd").read_text()
        assert not sb.path.exists()

    def test_context_manager_discards_unpromoted(self, repo, sandbox_dir):
        with Sandbox(repo, sandbox_dir) as sb:
            p = sb.path
            sb.apply_patch(GOOD_DIFF)
        assert not p.exists()

    def test_promote_keeps_sandbox_readable(self, repo, sandbox_dir):
        with Sandbox(repo, sandbox_dir) as sb:
            sb.apply_patch(GOOD_DIFF)
            artifact = sb.promote()
            assert artifact.exists()
        # promoted sandbox is not auto-discarded on exit
        sb.discard()

    def test_discard_idempotent(self, repo, sandbox_dir):
        sb = Sandbox(repo, sandbox_dir)
        sb.create()
        sb.discard()
        sb.discard()

    def test_missing_project_raises(self, tmp_path, sandbox_dir):
        with pytest.raises(SandboxError):
            Sandbox(tmp_path / "nope", sandbox_dir)

    def test_patch_before_create_raises(self, repo, sandbox_dir):
        sb = Sandbox(repo, sandbox_dir)
        with pytest.raises(SandboxError):
            sb.apply_patch(GOOD_DIFF)


class TestCopyFallback:
    def test_non_git_project_uses_copy(self, tmp_path, sandbox_dir):
        proj = tmp_path / "plain"
        proj.mkdir()
        (proj / "player.gd").write_text("extends Node2D\nvar speed = 100\n", encoding="utf-8")
        sb = Sandbox(proj, sandbox_dir)
        sb.create()
        assert sb.mode == "copy"
        assert sb.apply_patch(GOOD_DIFF) is True
        artifact = sb.promote()
        assert artifact.exists()
        sb.discard()
        assert not sb.path.exists()


class TestGates:
    def test_gates_all_skipped_or_pass_on_empty(self, repo, sandbox_dir, monkeypatch):
        monkeypatch.setattr("tools.sandbox.godot_runner.GODOT_EXECUTABLE", None)
        with Sandbox(repo, sandbox_dir) as sb:
            sb.apply_patch(GOOD_DIFF)
            report = sb.run_gates()
        assert report.ok  # None (skipped) never counts as failure
        assert any("syntax" in d for d in report.details)

    def test_syntax_failure_blocks(self, repo, sandbox_dir, monkeypatch):
        import sys
        # exe must "exist" so the gate actually runs (missing Godot = infra skip)
        monkeypatch.setattr("tools.sandbox.godot_runner.GODOT_EXECUTABLE", sys.executable)
        monkeypatch.setattr(
            "tools.sandbox.godot_runner.run_script",
            lambda *a, **k: {"success": False, "output": "err", "errors": ["parse error"], "returncode": 1},
        )
        with Sandbox(repo, sandbox_dir) as sb:
            sb.apply_patch(GOOD_DIFF)
            report = sb.run_gates()
        assert report.syntax_ok is False
        assert not report.ok

    def test_assertion_still_reproduces_blocks(self, repo, sandbox_dir, monkeypatch):
        monkeypatch.setattr("tools.sandbox.godot_runner.GODOT_EXECUTABLE", None)
        monkeypatch.setattr("tools.index_builder.build_project_index", lambda p: {})
        monkeypatch.setattr(
            "tools.assertion_verifier.verify_batch",
            lambda assertions, index: [{"verified": "confirmed"} for _ in assertions],
        )
        with Sandbox(repo, sandbox_dir) as sb:
            report = sb.run_gates(target_assertions=[{"type": "file_not_exists", "path": "x"}])
        assert report.assertions_resolved is False
        assert not report.ok

    def test_assertion_resolved_passes(self, repo, sandbox_dir, monkeypatch):
        monkeypatch.setattr("tools.sandbox.godot_runner.GODOT_EXECUTABLE", None)
        monkeypatch.setattr("tools.index_builder.build_project_index", lambda p: {})
        monkeypatch.setattr(
            "tools.assertion_verifier.verify_batch",
            lambda assertions, index: [{"verified": "refuted"} for _ in assertions],
        )
        with Sandbox(repo, sandbox_dir) as sb:
            report = sb.run_gates(target_assertions=[{"type": "file_not_exists", "path": "x"}])
        assert report.assertions_resolved is True
        assert report.ok


class TestGateReport:
    def test_none_is_not_failure(self):
        assert GateReport(syntax_ok=None, regression_ok=None).ok

    def test_any_false_fails(self):
        assert not GateReport(syntax_ok=True, regression_ok=False).ok
