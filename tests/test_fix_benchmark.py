"""tests/test_fix_benchmark.py — benchmark orchestration with a fake fixer."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.fix_benchmark import (
    evaluate_outcome,
    finding_for_mutant,
    load_mutants,
    run_benchmark,
    summarize,
)

MUTANT = {
    "mutant_id": "mT1", "file": "ships/x.gd", "operator": "rename_emitted_signal",
    "line": 10, "expected_assertion": "signal_not_declared",
    "expected_entity": "sig_broken",
    "description": "Emits Events.sig_broken which is not declared",
}


class TestFindingSpec(unittest.TestCase):
    def test_finding_carries_assertion(self):
        f = finding_for_mutant(MUTANT)
        self.assertEqual(f["assertion"], {
            "type": "signal_not_declared",
            "signal_name": "sig_broken",
            "file": "res://ships/x.gd",
        })
        self.assertEqual(f["line"], 10)

    def test_unknown_assertion_type_gets_no_entity(self):
        m = {**MUTANT, "expected_assertion": "node_not_found"}
        f = finding_for_mutant(m)
        self.assertEqual(f["assertion"]["type"], "node_not_found")
        self.assertNotIn("signal_name", f["assertion"])


class TestEvaluateOutcome(unittest.TestCase):
    def _mk(self, **kw):
        base = {
            "status": "fixed",
            "gate_report": {"ok": True, "assertions_resolved": True},
            "boundary": {"boundary_ok": True},
        }
        base.update(kw)
        return base

    def test_clean_fix_counts(self):
        self.assertTrue(evaluate_outcome(self._mk()))

    def test_unfixed_never_counts(self):
        self.assertFalse(evaluate_outcome(self._mk(status="failed")))

    def test_red_gates_never_count(self):
        self.assertFalse(evaluate_outcome(
            self._mk(gate_report={"ok": False, "assertions_resolved": True})))

    def test_assertion_still_reproducing_never_counts(self):
        self.assertFalse(evaluate_outcome(self._mk(
            gate_report={"ok": True, "assertions_resolved": False})))

    def test_boundary_violation_never_counts(self):
        self.assertFalse(evaluate_outcome(
            self._mk(boundary={"boundary_ok": False})))


class TestRunBenchmark(unittest.TestCase):
    def test_fake_fixer_metrics_and_resume(self):
        out = Path(tempfile.mkdtemp()) / "bench.jsonl"
        mutants = [dict(MUTANT), {**MUTANT, "mutant_id": "mT2"},
                   {**MUTANT, "mutant_id": "mT3"}]
        outcomes = iter([
            {"status": "fixed", "rounds_used": 1,
             "gate_report": {"ok": True, "assertions_resolved": True},
             "boundary": {"boundary_ok": True, "mode": "edit_blocks"},
             "usage": {"prompt_tokens": 100, "completion_tokens": 50},
             "error": None},
            {"status": "failed", "rounds_used": 3, "gate_report": {},
             "boundary": {}, "usage": {"prompt_tokens": 300,
                                       "completion_tokens": 90},
             "error": "exhausted"},
        ])

        def fake_fix(file, issues, root, *, client):
            return next(outcomes)

        # first run: 2 mutants; second run resumes and does only the third
        summary1 = run_benchmark(mutants[:2], Path("/tmp"), None, fake_fix, out)
        self.assertEqual(summary1["n"], 2)
        self.assertEqual(summary1["fix_rate"], 0.5)
        self.assertEqual(summary1["mean_rounds"], 2.0)

        outcomes2 = iter([
            {"status": "fixed", "rounds_used": 2,
             "gate_report": {"ok": True, "assertions_resolved": None},
             "boundary": {"boundary_ok": True, "mode": "whole_file"},
             "usage": {"prompt_tokens": 50, "completion_tokens": 20},
             "error": None},
        ])
        summary2 = run_benchmark(mutants, Path("/tmp"), None,
                                 lambda *a, **k: next(outcomes2), out)
        self.assertEqual(summary2["n"], 3)  # includes resumed rows
        # mT3's assertions gate never ran (None) — unverified, doesn't count
        self.assertAlmostEqual(summary2["fix_rate"], 1 / 3, places=3)
        self.assertEqual(summary2["whole_file_fallbacks"], 1)
        self.assertEqual(summary2["total_tokens"], 100 + 50 + 300 + 90 + 70)
        self.assertEqual(summary2["by_operator"]["rename_emitted_signal"],
                         {"n": 3, "fixed": 1})

    def test_real_mutant_set_loads(self):
        mutants = load_mutants()
        self.assertGreaterEqual(len(mutants), 20)
        for m in mutants:
            finding_for_mutant(m)  # never raises


if __name__ == "__main__":
    unittest.main()
