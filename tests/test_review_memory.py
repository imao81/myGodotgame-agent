"""tests/test_review_memory.py — ReviewMemory (verifier-gated test-time learning).

Synthetic data only — no LLM, no real game project required.

Run:
    python -m pytest tests/test_review_memory.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.review_memory import ReviewMemory
from review import build_prompt


def _finding(verdict: str, assertion: dict, message: str = "test finding",
             file: str = "autoload/events.gd") -> dict:
    return {
        "file": file,
        "severity": "error",
        "message": message,
        "assertion": assertion,
        "verified": verdict,
        "_verifier": {"details": f"verifier says {verdict}"},
    }


_SIGNAL_ASSERTION = {"type": "signal_not_declared", "signal_name": "damaged",
                     "file": "res://autoload/events.gd"}
_INDEX = {
    "scripts": {
        "res://autoload/events.gd": {
            "class_name": None,
            "signals": ["damaged", "player_died"],
            "methods": {},
        },
    },
}


class TestCurate(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.mem = ReviewMemory(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_refuted_becomes_false_positive_memory(self):
        stats = self.mem.curate([_finding("refuted", _SIGNAL_ASSERTION)], "run1")
        self.assertEqual(stats["added"], 1)
        e = self.mem.entries[0]
        self.assertEqual(e["kind"], "false_positive")
        self.assertEqual(e["entity"], "damaged@res://autoload/events.gd")

    def test_confirmed_becomes_pattern_memory(self):
        a = {"type": "identifier_not_defined", "identifier": "Foo"}
        stats = self.mem.curate([_finding("confirmed", a)], "run1")
        self.assertEqual(stats["added"], 1)
        self.assertEqual(self.mem.entries[0]["kind"], "confirmed_pattern")

    def test_unverifiable_is_skipped(self):
        stats = self.mem.curate([_finding("unverifiable", _SIGNAL_ASSERTION)], "run1")
        self.assertEqual(stats["skipped"], 1)
        self.assertEqual(len(self.mem.entries), 0)

    def test_unverifiable_assertion_type_is_skipped(self):
        stats = self.mem.curate(
            [_finding("confirmed", {"type": "unverifiable"})], "run1")
        self.assertEqual(stats["skipped"], 1)

    def test_duplicate_entity_merges_and_increments_hits(self):
        self.mem.curate([_finding("refuted", _SIGNAL_ASSERTION)], "run1")
        stats = self.mem.curate([_finding("refuted", _SIGNAL_ASSERTION)], "run2")
        self.assertEqual(stats["updated"], 1)
        self.assertEqual(len(self.mem.entries), 1)
        self.assertEqual(self.mem.entries[0]["hits"], 2)
        self.assertEqual(self.mem.entries[0]["evidence"]["run_id"], "run2")

    def test_persistence_roundtrip(self):
        self.mem.curate([_finding("refuted", _SIGNAL_ASSERTION)], "run1")
        mem2 = ReviewMemory(Path(self.tmp.name))
        self.assertEqual(len(mem2.entries), 1)
        self.assertEqual(mem2.entries[0]["assertion_type"], "signal_not_declared")


class TestRetrieve(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.mem = ReviewMemory(Path(self.tmp.name))
        self.mem.curate([_finding("refuted", _SIGNAL_ASSERTION,
                                  message="signal 'damaged' not declared in events.gd")],
                        "run1")

    def tearDown(self):
        self.tmp.cleanup()

    def test_retrieves_relevant_entry(self):
        hits = self.mem.retrieve("autoload/events.gd", _INDEX)
        self.assertEqual(len(hits), 1)
        self.assertIn("_score", hits[0])

    def test_irrelevant_file_returns_empty(self):
        hits = self.mem.retrieve("vfx/pop_effect.gd",
                                 {"scripts": {"res://vfx/pop_effect.gd": {
                                     "class_name": None, "signals": [],
                                     "methods": {"pop": {}}}}},
                                 min_score=0.99)
        self.assertEqual(hits, [])

    def test_confirmed_patterns_not_injected_by_default(self):
        a = {"type": "identifier_not_defined", "identifier": "events"}
        self.mem.curate([_finding("confirmed", a)], "run2")
        hits = self.mem.retrieve("autoload/events.gd", _INDEX, k=10)
        self.assertTrue(all(h["kind"] == "false_positive" for h in hits))

    def test_token_budget_truncates(self):
        mem = ReviewMemory(Path(self.tmp.name), max_inject_tokens=5)
        for i in range(6):
            a = {"type": "signal_not_declared", "signal_name": "damaged",
                 "file": "res://autoload/events.gd"}
            mem.curate([_finding("refuted", a, message=f"variant {'x' * 100} {i}")],
                       f"run{i}")
        hits = mem.retrieve("autoload/events.gd", _INDEX, k=6, min_score=0.0)
        self.assertLessEqual(len(hits), 1)  # 5 tokens ≈ 20 chars — at most one entry

    def test_recency_decay_lowers_score(self):
        old = datetime.now(timezone.utc) - timedelta(days=200)
        self.mem.entries[0]["last_seen_at"] = old.isoformat()
        hits_fresh = self.mem.retrieve("autoload/events.gd", _INDEX, min_score=0.0)
        # effective hits decayed: 1 * 0.95^200 ≈ tiny — score must drop vs fresh
        fresh_score = hits_fresh[0]["_score"] if hits_fresh else 0.0
        self.mem.entries[0]["last_seen_at"] = datetime.now(timezone.utc).isoformat()
        hits_now = self.mem.retrieve("autoload/events.gd", _INDEX, min_score=0.0)
        self.assertGreater(hits_now[0]["_score"], fresh_score)


class TestPromptInjection(unittest.TestCase):

    def test_memory_block_injected(self):
        prompt = build_prompt("autoload/events.gd", "extends Node\nvar x = 1\n",
                              {"autoloads": {}, "godot_version": "4.6"},
                              memory_hints=[{
                                  "assertion_type": "signal_not_declared",
                                  "summary": "signal 'damaged' not declared",
                                  "evidence": {"verifier_status": "refuted"},
                                  "hits": 2,
                              }])
        self.assertIn("KNOWN FALSE-POSITIVE PATTERNS", prompt)
        self.assertIn("signal 'damaged' not declared", prompt)

    def test_no_memory_no_block(self):
        prompt = build_prompt("autoload/events.gd", "extends Node\nvar x = 1\n",
                              {"autoloads": {}, "godot_version": "4.6"})
        self.assertNotIn("KNOWN FALSE-POSITIVE PATTERNS", prompt)


if __name__ == "__main__":
    unittest.main()
