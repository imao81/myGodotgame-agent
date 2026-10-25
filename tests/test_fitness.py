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
    pareto_sample,
    template_is_valid,
)
from eval.fitness import (
    feedback_text,
    load_golden,
    match_findings,
    score_files,
    score_match,
    trace_feedback,
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


class TestScoreFiles(unittest.TestCase):
    """Per-instance scores — the data structure the Pareto frontier needs."""

    def test_per_file_independence(self):
        produced = [
            {"file": "a.gd", "line": 10,
             "assertion": {"type": "identifier_not_defined",
                           "identifier": "_timer"}},
            {"file": "b.gd", "line": 5,
             "assertion": {"type": "signal_not_declared",
                           "signal_name": "docked"}},
        ]
        s = score_files(produced, G[:3])
        self.assertEqual(s["a.gd"], 1.0)    # tp matched
        self.assertEqual(s["b.gd"], -1.0)   # fp reproduced

    def test_untouched_file_scores_zero(self):
        self.assertEqual(score_files([], G[:3])["a.gd"], 0.0)


class TestParetoSample(unittest.TestCase):
    """Paper Algorithm 2: per-instance winners, domination removal, sampling."""

    @staticmethod
    def _cand(cid: int, scores: dict) -> dict:
        return {"id": cid, "scores": scores}

    def test_each_instance_winner_stays_alive(self):
        # c1 dominates a.gd, c2 dominates b.gd — greedy mean would kill c2
        cands = [self._cand(1, {"a.gd": 2.0, "b.gd": 0.0}),
                 self._cand(2, {"a.gd": -1.0, "b.gd": 3.0}),
                 self._cand(3, {"a.gd": -5.0, "b.gd": -5.0})]  # wins nothing
        rng = __import__("random").Random(0)
        picks = {pareto_sample(cands, ["a.gd", "b.gd"], rng)["id"]
                 for _ in range(50)}
        self.assertEqual(picks, {1, 2})  # c3 never picked; both winners survive

    def test_dominated_candidate_removed(self):
        # c2's win-set {a.gd} is a strict subset of c1's {a.gd, b.gd}
        cands = [self._cand(1, {"a.gd": 2.0, "b.gd": 2.0}),
                 self._cand(2, {"a.gd": 2.0, "b.gd": 0.0})]
        rng = __import__("random").Random(0)
        picks = {pareto_sample(cands, ["a.gd", "b.gd"], rng)["id"]
                 for _ in range(30)}
        self.assertEqual(picks, {1})

    def test_greedy_would_collapse_pareto_does_not(self):
        """Regression for the first implementation's local-optimum failure."""
        # c1 has the best MEAN but loses b.gd badly; c2 owns b.gd
        cands = [self._cand(1, {"a.gd": 2.0, "b.gd": 0.0}),
                 self._cand(2, {"a.gd": 0.9, "b.gd": 1.0})]
        greedy = max(cands, key=lambda c: sum(c["scores"].values()))
        self.assertEqual(greedy["id"], 1)
        rng = __import__("random").Random(1)
        picks = {pareto_sample(cands, ["a.gd", "b.gd"], rng)["id"]
                 for _ in range(50)}
        self.assertIn(2, picks)  # the b.gd specialist is still sampled


class TestTraceFeedback(unittest.TestCase):

    def test_four_trace_categories(self):
        produced = [
            {"file": "a.gd", "line": 10, "verified": "confirmed",
             "message": "found the timer bug",
             "assertion": {"type": "identifier_not_defined",
                           "identifier": "_timer"}},
            {"file": "b.gd", "line": 5, "verified": "confirmed",
             "message": "claimed docked undeclared",
             "assertion": {"type": "signal_not_declared",
                           "signal_name": "docked"}},
            {"file": "n.gd", "line": 1, "verified": "refuted",
             "message": "novel but statically false",
             "assertion": {"type": "unverifiable"}},
        ]
        text = trace_feedback(produced, G[:3])
        self.assertIn("human-confirmed bugs", text)
        self.assertIn("human-REJECTED", text)
        self.assertIn("verifier REFUTED", text)
        self.assertIn("MISSED", text)

    def test_empty_production_reports_misses(self):
        text = trace_feedback([], G[:3])
        self.assertIn("MISSED", text)

    def test_truly_empty_is_explicit(self):
        self.assertEqual(trace_feedback([], []), "No findings produced at all.")


if __name__ == "__main__":
    unittest.main()
