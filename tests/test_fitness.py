"""tests/test_fitness.py — golden-set fitness scorer + GEPA pure logic.

Covers:
  - eval/fitness: golden loading (disputed excluded), entity/line matching,
    scoring sign conventions, feedback text
  - agents/gepa: placeholder validation, fence stripping, mutation gating
    (mocked mutator provider — zero LLM calls)

Run:
    python -m pytest tests/test_fitness.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.gepa import (
    REQUIRED_PLACEHOLDERS,
    _strip_fences,
    mutate_template,
    template_is_valid,
)
from eval.fitness import (
    feedback_text,
    load_golden,
    match_findings,
    score_match,
)


def _golden(recs: list[dict]) -> list[dict]:
    return recs


G = [
    {"id": "t1", "file": "a.gd", "line": 10, "human_label": "tp",
     "message": "real bug",
     "assertion": {"type": "identifier_not_defined", "identifier": "_timer"}},
    {"id": "t2", "file": "a.gd", "line": 40, "human_label": "partial",
     "message": "half right", "assertion": {"type": "unverifiable"}},
    {"id": "f1", "file": "b.gd", "line": 5, "human_label": "fp",
     "message": "not a bug", "assertion": {"type": "signal_not_declared",
                                           "signal_name": "docked"}},
    {"id": "d1", "file": "c.gd", "line": 1, "human_label": "disputed",
     "message": "contested", "assertion": {"type": "unverifiable"}},
]


class TestLoadGolden(unittest.TestCase):

    def test_disputed_excluded(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "g.jsonl"
            p.write_text("\n".join(json.dumps(r) for r in G) + "\n",
                         encoding="utf-8")
            recs = load_golden(p)
        self.assertEqual(len(recs), 3)
        self.assertNotIn("disputed", {r["human_label"] for r in recs})


class TestMatching(unittest.TestCase):

    def test_entity_match(self):
        produced = [{"file": "a.gd", "line": 99,
                     "assertion": {"type": "identifier_not_defined",
                                   "identifier": "_timer"}}]
        m = match_findings(produced, G[:3])
        self.assertEqual(len(m["matched"]), 1)
        self.assertEqual(m["matched"][0][1]["id"], "t1")

    def test_line_tolerance_match(self):
        produced = [{"file": "a.gd", "line": 42,
                     "assertion": {"type": "unverifiable"}}]
        m = match_findings(produced, G[:3])
        self.assertEqual(m["matched"][0][1]["id"], "t2")  # line 40 ± 3

    def test_wrong_file_and_novel(self):
        produced = [{"file": "zzz.gd", "line": 1,
                     "assertion": {"type": "unverifiable"}}]
        m = match_findings(produced, G[:3])
        self.assertEqual(len(m["novel"]), 1)
        self.assertEqual(len(m["matched"]), 0)

    def test_missed_only_counts_true_labels(self):
        m = match_findings([], G[:3])
        self.assertEqual({g["id"] for g in m["missed"]}, {"t1", "t2"})


class TestScoring(unittest.TestCase):

    def test_perfect_score(self):
        produced = [
            {"file": "a.gd", "line": 10,
             "assertion": {"type": "identifier_not_defined",
                           "identifier": "_timer"}},
            {"file": "a.gd", "line": 40,
             "assertion": {"type": "unverifiable"}},
        ]
        m = match_findings(produced, G[:3])
        s = score_match(m, G[:3])
        self.assertAlmostEqual(s["recall_true"], 1.0)
        self.assertEqual(s["fp_repro_rate"], 0.0)
        self.assertAlmostEqual(s["fitness"], 1.0)

    def test_fp_reproduction_penalised(self):
        produced = [{"file": "b.gd", "line": 5,
                     "assertion": {"type": "signal_not_declared",
                                   "signal_name": "docked"}}]
        s = score_match(match_findings(produced, G[:3]), G[:3])
        self.assertEqual(s["fp_repro_rate"], 1.0)
        self.assertAlmostEqual(s["fitness"], -1.0)

    def test_novel_penalty_capped_and_small(self):
        produced = [{"file": f"n{i}.gd", "line": 1,
                     "assertion": {"type": "unverifiable"}} for i in range(40)]
        s = score_match(match_findings(produced, G[:3]), G[:3])
        self.assertEqual(s["novel_penalty"], 0.02 * 25)  # capped
        self.assertEqual(s["novel_count"], 40)

    def test_silence_scores_zero(self):
        s = score_match(match_findings([], G[:3]), G[:3])
        self.assertEqual(s["fitness"], 0.0)


class TestFeedbackText(unittest.TestCase):

    def test_mentions_missed_and_fp(self):
        produced = [{"file": "b.gd", "line": 5,
                     "assertion": {"type": "signal_not_declared",
                                   "signal_name": "docked"}}]
        g = [dict(r, human_notes="humans said no") for r in G[:3]]
        m = match_findings(produced, g)
        text = feedback_text(score_match(m, g), m)
        self.assertIn("FAILED to find", text)
        self.assertIn("FALSE POSITIVES", text)
        self.assertIn("humans said no", text)


class TestGepaPureLogic(unittest.TestCase):

    def test_template_validation(self):
        self.assertEqual(template_is_valid(
            "a {filename} b {autoload_list} c {context_and_code}"), [])
        self.assertEqual(len(template_is_valid("nothing here")), 3)

    def test_strip_fences(self):
        self.assertEqual(_strip_fences("```text\nhello\n```"), "hello")
        self.assertEqual(_strip_fences("plain"), "plain")

    def test_mutant_without_placeholders_rejected(self):
        provider = mock.Mock()
        provider.generate.return_value = "x" * 500  # long but placeholder-free
        self.assertIsNone(mutate_template("base", "feedback", provider))

    def test_mutant_with_placeholders_accepted(self):
        provider = mock.Mock()
        provider.generate.return_value = (
            "header {filename} {autoload_list} body " + "x" * 300
            + " {context_and_code}")
        out = mutate_template("base {filename}", "feedback", provider)
        self.assertIsNotNone(out)
        # mutator prompt must contain base template + feedback
        sent = provider.generate.call_args.args[0]
        self.assertIn("base {filename}", sent)
        self.assertIn("feedback", sent)

    def test_mutation_prompt_preserves_literal_placeholders(self):
        """Regression: .replace() would eat the placeholder instructions."""
        provider = mock.Mock()
        provider.generate.return_value = "too short"
        mutate_template("T", "F", provider)
        sent = provider.generate.call_args.args[0]
        for p in REQUIRED_PLACEHOLDERS:
            self.assertIn(p, sent)  # instruction block survived intact


if __name__ == "__main__":
    unittest.main()
