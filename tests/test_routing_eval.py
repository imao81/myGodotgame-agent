"""tests/test_routing_eval.py — routing matrix + offline policy evaluation.

Covers (no LLM calls, no network):
  - routing_matrix.jsonl schema validation (on a toy matrix)
  - oracle policy: cheapest catching arm, fallback to cheapest when none catch
  - rule baseline: n_lines median threshold behaviour
  - LinUCB replay: same seed → identical plan (determinism)
  - 5-fold CV: no leakage (training folds never contain the evaluated file)
  - reward_of: consistency with agents.router.compute_reward formula

Run:
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/Scripts/python.exe -m pytest tests/test_routing_eval.py -q
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.router import compute_reward  # noqa: E402
from eval import routing_analysis as ra  # noqa: E402

# ── Toy matrix fixture ─────────────────────────────────────────────────────────
# 4 files; f1/f3 carry a mutant each. flash catches f1's bug only; k3 catches both.

def _rec(file: str, arm: str, n_lines: int, matched: list[str],
         prompt: int, completion: int, fp: int = 0) -> dict:
    return {
        "file": file, "arm": arm, "n_lines": n_lines,
        "findings_raw": fp + len(matched),
        "findings_verified": [{"line": 1, "verified": "confirmed",
                               "assertion_type": "signal_not_declared",
                               "message": "x"}] * (fp + len(matched)),
        "matched_mutant_ids": matched,
        "fp_count": fp,
        "prompt_tokens": prompt, "completion_tokens": completion,
        "latency_s": 1.0, "error": None,
    }


TOY_MATRIX = [
    _rec("f1.gd", "deepseek-flash", 10, ["m001"], 1000, 100),
    _rec("f1.gd", "kimi-standard", 10, ["m001"], 2000, 200),
    _rec("f2.gd", "deepseek-flash", 20, [], 1000, 50, fp=1),
    _rec("f2.gd", "kimi-standard", 20, [], 2000, 100),
    _rec("f3.gd", "deepseek-flash", 90, [], 1500, 60),
    _rec("f3.gd", "kimi-standard", 90, ["m002"], 3000, 300),
    _rec("f4.gd", "deepseek-flash", 100, [], 1600, 40),
    _rec("f4.gd", "kimi-standard", 100, [], 3200, 120),
]

FILES = sorted({r["file"] for r in TOY_MATRIX})
N_LINES = {r["file"]: r["n_lines"] for r in TOY_MATRIX
           if r["arm"] == "deepseek-flash"}
CONTEXTS = {f: np.arange(1, 10, dtype=float) * (i + 1)
            for i, f in enumerate(FILES)}


class TestMatrixSchema(unittest.TestCase):
    """The toy matrix satisfies the collector's record schema."""

    REQUIRED = {"file", "arm", "n_lines", "findings_raw", "findings_verified",
                "matched_mutant_ids", "fp_count", "prompt_tokens",
                "completion_tokens", "latency_s", "error"}

    def test_schema_keys(self):
        for rec in TOY_MATRIX:
            self.assertTrue(self.REQUIRED <= set(rec),
                            f"missing keys: {self.REQUIRED - set(rec)}")

    def test_schema_types(self):
        for rec in TOY_MATRIX:
            self.assertIsInstance(rec["file"], str)
            self.assertIn(rec["arm"], ra.REL_COST)
            self.assertIsInstance(rec["n_lines"], int)
            self.assertIsInstance(rec["findings_raw"], int)
            self.assertIsInstance(rec["findings_verified"], list)
            self.assertIsInstance(rec["matched_mutant_ids"], list)
            self.assertIsInstance(rec["fp_count"], int)
            self.assertGreaterEqual(rec["prompt_tokens"], 0)
            self.assertGreaterEqual(rec["completion_tokens"], 0)
            self.assertGreaterEqual(rec["latency_s"], 0.0)
            self.assertIsNone(rec["error"])

    def test_load_matrix_filters_errors(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "m.jsonl"
            bad = {**TOY_MATRIX[0], "error": "LLM call failed: boom"}
            p.write_text("\n".join(json.dumps(r)
                                   for r in [*TOY_MATRIX, bad]) + "\n",
                         encoding="utf-8")
            loaded = ra.load_matrix(p)
        self.assertEqual(len(loaded), len(TOY_MATRIX))
        self.assertTrue(all(r["error"] is None for r in loaded))


class TestOracle(unittest.TestCase):
    def setUp(self):
        self.lookup = ra.matrix_lookup(TOY_MATRIX)

    def test_cheapest_catching_arm(self):
        plan = ra.policy_oracle(self.lookup, FILES)
        # f1: both catch → flash (cheaper); f3: only k3 catches → k3
        self.assertEqual(plan["f1.gd"], "deepseek-flash")
        self.assertEqual(plan["f3.gd"], "kimi-standard")
        # f2/f4: nobody catches → cheapest
        self.assertEqual(plan["f2.gd"], "deepseek-flash")
        self.assertEqual(plan["f4.gd"], "deepseek-flash")

    def test_oracle_recall_is_maximum(self):
        plan = ra.policy_oracle(self.lookup, FILES)
        m = ra.evaluate_plan(plan, self.lookup, n_mutants=2, lam=0.05)
        self.assertEqual(m["detected"], 2)
        self.assertEqual(m["recall"], 1.0)
        # cheaper than all-k3
        m_k3 = ra.evaluate_plan(ra.policy_constant(FILES, "kimi-standard"),
                                self.lookup, n_mutants=2, lam=0.05)
        self.assertLess(m["usd"], m_k3["usd"])
        self.assertEqual(m_k3["recall"], 1.0)  # k3 also catches both here


class TestRuleBaseline(unittest.TestCase):
    def test_median_threshold(self):
        plan = ra.policy_rule(N_LINES)
        median = float(np.median(list(N_LINES.values())))  # (20+90)/2 = 55
        self.assertEqual(median, 55.0)
        for f, arm in plan.items():
            if N_LINES[f] > median:
                self.assertEqual(arm, "kimi-standard", f)
            else:
                self.assertEqual(arm, "deepseek-flash", f)
        # concrete: 10/20 → flash; 90/100 → k3
        self.assertEqual(plan["f1.gd"], "deepseek-flash")
        self.assertEqual(plan["f2.gd"], "deepseek-flash")
        self.assertEqual(plan["f3.gd"], "kimi-standard")
        self.assertEqual(plan["f4.gd"], "kimi-standard")


class TestReplay(unittest.TestCase):
    def setUp(self):
        self.lookup = ra.matrix_lookup(TOY_MATRIX)

    def test_same_seed_same_plan(self):
        for lam in (0.01, 0.05, 0.2):
            a = ra.run_replay(self.lookup, CONTEXTS, lam, seed=7)
            b = ra.run_replay(self.lookup, CONTEXTS, lam, seed=7)
            self.assertEqual(a, b)

    def test_plan_covers_all_files_and_arms_valid(self):
        plan = ra.run_replay(self.lookup, CONTEXTS, 0.05, seed=0)
        self.assertEqual(set(plan), set(FILES))
        self.assertTrue(set(plan.values()) <= set(ra.ARM_ORDER))

    def test_high_lambda_prefers_cheap_arm(self):
        # With λ huge the cost term dominates: after the 2 cold-start pulls
        # (one per arm), UCB routing must pick deepseek-flash for the rest.
        import random
        seed = 3
        order = sorted(CONTEXTS)
        random.Random(seed).shuffle(order)
        post_cold_start = order[len(ra.ARM_ORDER):]
        plan = ra.run_replay(self.lookup, CONTEXTS, 100.0, seed=seed)
        for f in post_cold_start:
            self.assertEqual(plan[f], "deepseek-flash", f)


class TestCrossValidation(unittest.TestCase):
    def setUp(self):
        self.lookup = ra.matrix_lookup(TOY_MATRIX)

    def test_folds_partition_files(self):
        folds = ra.make_folds(FILES, n_folds=2, seed=42)
        flat = [f for fold in folds for f in fold]
        self.assertEqual(sorted(flat), FILES)

    def test_no_leakage(self):
        # Held-out files must never appear in the fold's training set.
        folds = ra.make_folds(FILES, n_folds=2, seed=42)
        for k, held_out in enumerate(folds):
            train = [f for j, fold in enumerate(folds) if j != k
                     for f in fold]
            self.assertFalse(set(held_out) & set(train))
        # And run_cv produces a plan covering exactly the held-out union.
        plan = ra.run_cv(self.lookup, CONTEXTS, 0.05, n_folds=2, seed=42)
        self.assertEqual(set(plan), set(FILES))

    def test_cv_deterministic(self):
        a = ra.run_cv(self.lookup, CONTEXTS, 0.05, n_folds=2, seed=42)
        b = ra.run_cv(self.lookup, CONTEXTS, 0.05, n_folds=2, seed=42)
        self.assertEqual(a, b)


class TestRewardConsistency(unittest.TestCase):
    """reward_of == router.compute_reward with confirmed-count ≡ caught."""

    def test_caught_matches_confirmed(self):
        rec = _rec("f.gd", "kimi-standard", 50, ["m001"], 2000, 500)
        for lam in (0.01, 0.05, 0.2):
            expected = compute_reward(
                [{"verified": "confirmed"}],
                {"prompt_tokens": 2000, "completion_tokens": 500},
                rel_cost=ra.REL_COST["kimi-standard"], lam=lam)
            self.assertAlmostEqual(
                ra.reward_of(rec, ra.REL_COST["kimi-standard"], lam), expected)

    def test_miss_matches_empty_findings(self):
        rec = _rec("f.gd", "deepseek-flash", 50, [], 1000, 100)
        for lam in (0.01, 0.05, 0.2):
            expected = compute_reward(
                [], {"prompt_tokens": 1000, "completion_tokens": 100},
                rel_cost=ra.REL_COST["deepseek-flash"], lam=lam)
            self.assertAlmostEqual(
                ra.reward_of(rec, ra.REL_COST["deepseek-flash"], lam), expected)

    def test_cost_term_sign_and_scale(self):
        rec = _rec("f.gd", "deepseek-flash", 50, [], 4000, 1000)
        # caught=0 → r = −λ·(5000/1000)·0.7
        self.assertAlmostEqual(ra.reward_of(rec, 0.7, 0.05), -0.05 * 5 * 0.7)


if __name__ == "__main__":
    unittest.main()
