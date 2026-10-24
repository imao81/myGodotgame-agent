"""tests/test_pipeline_nodes.py — LangGraph node unit tests.

Covers the nodes that drive the CI gate but had no direct tests:
  - report_merger_node: red/yellow/green health classification (exit-code source)
  - resolve_scope_node / _expand_dependency_scope: git-diff scoping + BFS expansion
  - preflight_route: abort/continue edge

No LLM, no Godot binary, no real game project required.

Run:
    python -m pytest tests/test_pipeline_nodes.py -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from game_agent_pipeline import (
    _default_state,
    _expand_dependency_scope,
    preflight_route,
    report_merger_node,
)


def _state(**overrides) -> dict:
    return {**_default_state(".", True), **overrides}


def _finding(sev: str) -> dict:
    return {"file": "a.gd", "line": 1, "rule": "x", "severity": sev,
            "final_severity": sev, "message": "m"}


class TestReportMergerHealth(unittest.TestCase):

    def test_red_on_pipeline_issues(self):
        out = report_merger_node(_state(pipeline_issues=[{"type": "syntax"}]))
        self.assertEqual(out["overall_health"], "red")

    def test_red_on_error_finding(self):
        out = report_merger_node(_state(deduplicated_findings=[_finding("error")]))
        self.assertEqual(out["overall_health"], "red")

    def test_red_on_critical_in_correlated(self):
        out = report_merger_node(_state(
            correlated_findings=[{"severity": "critical", "final_severity": "critical"}]))
        self.assertEqual(out["overall_health"], "red")

    def test_yellow_on_warning_only(self):
        out = report_merger_node(_state(deduplicated_findings=[_finding("warning")]))
        self.assertEqual(out["overall_health"], "yellow")

    def test_yellow_when_files_skipped(self):
        """Skipped files = incomplete coverage — must not report green."""
        out = report_merger_node(_state(skipped_files=["a.gd"]))
        self.assertEqual(out["overall_health"], "yellow")

    def test_green_when_clean(self):
        out = report_merger_node(_state())
        self.assertEqual(out["overall_health"], "green")

    def test_summary_contains_health(self):
        out = report_merger_node(_state())
        self.assertIn("GREEN", out["summary"])


class TestPreflightRoute(unittest.TestCase):

    def test_abort_routes_to_end(self):
        self.assertEqual(preflight_route(_state(abort=True)), "abort")

    def test_continue_by_default(self):
        self.assertEqual(preflight_route(_state()), "continue")


class TestExpandDependencyScope(unittest.TestCase):

    def test_bfs_expands_through_res_refs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "b.gd").write_text(
                'extends Node\nconst A = preload("res://a.gd")\n', encoding="utf-8")
            (root / "c.gd").write_text(
                'extends Node\nconst B = preload("res://b.gd")\n', encoding="utf-8")
            (root / "unrelated.gd").write_text("extends Node\n", encoding="utf-8")

            scope = _expand_dependency_scope(root, {"a.gd"})
            self.assertEqual(scope, {"a.gd", "b.gd", "c.gd"})

    def test_no_referrers_keeps_seed_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a.gd").write_text("extends Node\n", encoding="utf-8")
            scope = _expand_dependency_scope(root, {"a.gd"})
            self.assertEqual(scope, {"a.gd"})

    def test_libraries_dir_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            lib = root / "libraries"
            lib.mkdir()
            (lib / "lib.gd").write_text(
                'extends Node\nconst A = preload("res://a.gd")\n', encoding="utf-8")
            (root / "a.gd").write_text("extends Node\n", encoding="utf-8")
            scope = _expand_dependency_scope(root, {"a.gd"})
            self.assertNotIn("libraries/lib.gd", scope)


if __name__ == "__main__":
    unittest.main()
