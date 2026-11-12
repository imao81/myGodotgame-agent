"""Isolated execution sandbox for agent-generated patches.

A patch is applied inside a throwaway git worktree (falling back to a plain
directory copy), must pass all acceptance gates there, and only then may be
promoted to a patch artifact. The real working tree is never touched.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from tools import godot_runner

_AGENT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SANDBOX_ROOT = _AGENT_ROOT / ".kimi" / "sandbox"


class SandboxError(RuntimeError):
    pass


@dataclass
class GateReport:
    # None means "gate skipped (infra unavailable)", not failure.
    syntax_ok: bool | None = None
    assertions_resolved: bool | None = None
    regression_ok: bool | None = None
    details: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(v is not False for v in (self.syntax_ok, self.assertions_resolved, self.regression_ok))


def _git(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if check and proc.returncode != 0:
        raise SandboxError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc


def _patch_files(unified_diff: str) -> list[str]:
    files = []
    for line in unified_diff.splitlines():
        m = re.match(r"\+\+\+ b/(.+)", line)
        if m:
            files.append(m.group(1))
    return files


class Sandbox:
    """One-shot isolated environment: create -> apply_patch -> run_gates -> promote/discard."""

    def __init__(self, project_root: str | Path, sandbox_root: str | Path | None = None):
        self.project_root = Path(project_root).resolve()
        if not self.project_root.is_dir():
            raise SandboxError(f"project not found: {self.project_root}")
        root = Path(sandbox_root) if sandbox_root else _DEFAULT_SANDBOX_ROOT
        self.run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.path = root / self.run_id
        self.mode: str | None = None  # "worktree" | "copy"
        self._promoted = False
        self._discarded = False

    # ── lifecycle ────────────────────────────────────────────────────────────

    def create(self) -> Path:
        if (self.project_root / ".git").exists():
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                _git(["worktree", "add", "--detach", str(self.path)], cwd=self.project_root)
                self.mode = "worktree"
                return self.path
            except SandboxError:
                shutil.rmtree(self.path, ignore_errors=True)
        try:
            ignore = shutil.ignore_patterns(".git", ".godot", "__pycache__")
            shutil.copytree(self.project_root, self.path, ignore=ignore)
            self.mode = "copy"
            return self.path
        except OSError as e:
            raise SandboxError(f"sandbox copy failed: {e}") from e

    def apply_patch(self, unified_diff: str) -> bool:
        """Apply a unified diff. Returns False on conflict; never applies partially."""
        if not self.mode:
            raise SandboxError("create() first")
        patch_file = self.path / "_sandbox.patch"
        patch_file.write_text(unified_diff, encoding="utf-8")
        try:
            # git apply also works on plain trees (no repo required).
            check = _git(["apply", "-p1", "--whitespace=nowarn", "--check", "_sandbox.patch"],
                         cwd=self.path, check=False)
            if check.returncode != 0:
                return False
            applied = _git(["apply", "-p1", "--whitespace=nowarn", "_sandbox.patch"],
                           cwd=self.path, check=False)
            if applied.returncode != 0:
                return False
            if self.mode == "copy":
                # No git history in copy mode: keep applied diffs for promote()/changed_files.
                stamp = self.path / f"{len(list(self.path.glob('*.applied.patch'))):03d}.applied.patch"
                stamp.write_text(unified_diff, encoding="utf-8")
            return True
        finally:
            patch_file.unlink(missing_ok=True)

    def run_gates(self, target_assertions: list[dict] | None = None) -> GateReport:
        if not self.mode:
            raise SandboxError("create() first")
        report = GateReport()
        self._gate_syntax(report)
        self._gate_assertions(report, target_assertions)
        self._gate_regression(report)
        return report

    def promote(self) -> Path:
        """Produce the patch artifact. Never writes to the real working tree."""
        if not self.mode:
            raise SandboxError("create() first")
        artifact = self.path / "fix.patch"
        if self.mode == "worktree":
            diff = _git(["diff", "HEAD"], cwd=self.path, check=False)
            artifact.write_text(diff.stdout, encoding="utf-8")
        else:
            # Copy mode has no git history; the artifact records what was applied.
            applied = sorted(self.path.glob("*.applied.patch"))
            artifact.write_text(
                "\n".join(p.read_text(encoding="utf-8") for p in applied)
                or "# copy-mode sandbox: see changed_files.txt\n",
                encoding="utf-8",
            )
        self._promoted = True
        return artifact

    def discard(self) -> None:
        if self._discarded:
            return
        self._discarded = True
        self.discard_at(self.project_root, self.path)

    @staticmethod
    def discard_at(project_root: str | Path, path: str | Path) -> None:
        """Best-effort cleanup of a sandbox directory created in an earlier session."""
        path = Path(path)
        if not path.exists():
            return
        subprocess.run(["git", "worktree", "remove", "--force", str(path)],
                       cwd=str(project_root), capture_output=True, text=True)
        subprocess.run(["git", "worktree", "prune"],
                       cwd=str(project_root), capture_output=True, text=True)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)

    def __enter__(self) -> "Sandbox":
        self.create()
        return self

    def __exit__(self, *exc) -> None:
        if not self._promoted:
            self.discard()

    # ── gates ────────────────────────────────────────────────────────────────

    def _gate_syntax(self, report: GateReport) -> None:
        changed = [f for f in self._changed_files() if f.endswith(".gd")]
        if not changed:
            report.details.append("syntax: no .gd files changed")
            return
        exe = godot_runner.GODOT_EXECUTABLE
        if not exe or not Path(exe).exists() and not shutil.which(str(exe)):
            report.syntax_ok = None
            report.details.append("syntax: Godot executable unavailable, gate skipped (infra)")
            return
        failures = []
        for rel in changed:
            r = godot_runner.run_script(str(self.path), str(self.path / rel))
            if not r["success"] or r["errors"]:
                failures.append(f"{rel}: {'; '.join(r['errors'][:3]) or r['output'][:200]}")
        report.syntax_ok = not failures
        report.details.append(f"syntax: {len(changed)} file(s) checked, {len(failures)} failed")
        report.details.extend(failures)

    def _gate_assertions(self, report: GateReport, assertions: list[dict] | None) -> None:
        if not assertions:
            return
        from tools.assertion_verifier import verify_batch
        from tools.index_builder import build_project_index

        index = build_project_index(self.path)
        results = verify_batch(assertions, index)
        still_valid = [r for r in results if r.get("verified") == "confirmed"]
        report.assertions_resolved = not still_valid
        report.details.append(
            f"assertions: {len(assertions)} re-verified, {len(still_valid)} still reproduce"
        )

    def _gate_regression(self, report: GateReport) -> None:
        controller = self.path / "test_controller.gd"
        if not controller.exists():
            report.details.append("regression: no test_controller.gd in sandbox, skipped")
            return
        r = godot_runner.run_test_controller(str(self.path), str(controller))
        if r["returncode"] == -1 and "not found" in r["output"].lower():
            report.regression_ok = None
            report.details.append("regression: Godot unavailable, gate skipped (infra)")
            return
        failed = [j for j in r["json_results"] if not j.get("passed", True)]
        report.regression_ok = r["success"] and not failed
        report.details.append(
            f"regression: {len(r['json_results'])} scenario result(s), {len(failed)} failed"
        )

    def _changed_files(self) -> list[str]:
        if self.mode == "worktree":
            proc = _git(["diff", "--name-only", "HEAD"], cwd=self.path, check=False)
            return [f for f in proc.stdout.splitlines() if f]
        return [p.name.replace(".applied.patch", "") for p in self.path.glob("*.applied.patch")]
