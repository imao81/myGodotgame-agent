"""tests/test_router.py — LinUCB contextual bandit routing.

Covers:
  - LinUCB math: update/select consistency, cold-start round-robin,
    deterministic convergence on synthetic contexts
  - persistence round-trip
  - compute_reward: confirmed/refuted/cost sign conventions
  - available_arms / escalation_ladder: env-key filtering, ordering
  - route_files_node: skip modes, plan shape
  - _review_one_file_with_retry: tier escalation on retry
  - code_review_node: bandit update + stats + persistence (mocked LLM)

No LLM calls, no Godot binary, no network.

Run:
    python -m pytest tests/test_router.py -v
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from agents.router import (
    LinUCB,
    available_arms,
    compute_reward,
    escalation_ladder,
    file_features,
)
import game_agent_pipeline as gap


# ── LinUCB core ────────────────────────────────────────────────────────────────

class TestLinUCBMath(unittest.TestCase):

    def test_update_then_theta_matches_manual(self):
        b = LinUCB(["a"], n_features=2, alpha=0.5)
        x = [1.0, 2.0]
        b.update("a", x, 3.0)
        # A = I + x xᵀ = [[2, 2], [2, 5]]; b = 3x = [3, 6]
        a_inv = np.linalg.inv(np.array([[2.0, 2.0], [2.0, 5.0]]))
        theta = a_inv @ np.array([3.0, 6.0])
        ucb = b._ucb("a", x)
        expected = float(theta @ np.array(x)
                         + 0.5 * math.sqrt(np.array(x) @ a_inv @ np.array(x)))
        self.assertAlmostEqual(ucb, expected, places=6)

    def test_cold_start_round_robin(self):
        b = LinUCB(["cheap", "mid", "premium"])
        contexts = {f"f{i}.gd": [1.0] * 9 for i in range(6)}
        plan = b.select(contexts)
        # first 3 files (sorted order) cover all three arms exactly once
        first_three = [plan[f"f{i}.gd"] for i in range(3)]
        self.assertEqual(sorted(first_three), ["cheap", "mid", "premium"])

    def test_convergence_separates_easy_and_hard_files(self):
        """Small files do well on cheap arm, big files need premium arm."""
        b = LinUCB(["cheap", "premium"], n_features=2, alpha=0.3)
        small = np.array([1.0, 0.0])
        big = np.array([1.0, 5.0])

        def true_reward(arm: str, x: np.ndarray) -> float:
            is_big = x[1] > 2.0
            if is_big:
                return 1.0 if arm == "premium" else -1.0
            return 1.0 if arm == "cheap" else -0.5

        for i in range(300):
            x = big if i % 2 else small
            arm = b.select({"f.gd": x})["f.gd"]
            b.update(arm, x, true_reward(arm, x))

        self.assertEqual(b.select({"f.gd": small})["f.gd"], "cheap")
        self.assertEqual(b.select({"f.gd": big})["f.gd"], "premium")

    def test_persistence_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "linucb.json"
            b = LinUCB(["a", "b"])
            b.update("a", [1.0] * 9, 2.0)
            b.save(path)
            b2 = LinUCB.load(path, ["a", "b"])
            contexts = {"x.gd": [1.0] * 9}
            self.assertEqual(b.select(contexts), b2.select(contexts))
            self.assertEqual(b2.pulls["a"], 1)

    def test_load_resets_when_arm_set_changes(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "linucb.json"
            LinUCB(["a", "b"]).save(path)
            b = LinUCB.load(path, ["a", "c"])
            self.assertEqual(b.pulls, {"a": 0, "c": 0})

    def test_missing_file_gives_fresh_bandit(self):
        b = LinUCB.load(Path("/nonexistent/linucb.json"), ["a"])
        self.assertEqual(b.pulls, {"a": 0})


# ── Reward ─────────────────────────────────────────────────────────────────────

class TestComputeReward(unittest.TestCase):

    def test_confirmed_minus_refuted(self):
        findings = [{"verified": "confirmed"}, {"verified": "confirmed"},
                    {"verified": "refuted"}, {"verified": "unverifiable"}]
        self.assertAlmostEqual(compute_reward(findings, None, 1.0), 1.0)

    def test_cost_term(self):
        usage = {"prompt_tokens": 2000, "completion_tokens": 500}
        # 2.5k tokens × rel_cost 2.0 × λ 0.05 = 0.25
        self.assertAlmostEqual(
            compute_reward([], usage, rel_cost=2.0, lam=0.05), -0.25)

    def test_zero_findings_prefer_cheaper_arm(self):
        usage = {"prompt_tokens": 1000, "completion_tokens": 0}
        cheap = compute_reward([], usage, rel_cost=1.0)
        premium = compute_reward([], usage, rel_cost=8.0)
        self.assertGreater(cheap, premium)


# ── Arms / ladder ─────────────────────────────────────────────────────────────

class TestArms(unittest.TestCase):

    def test_available_arms_filtered_by_keys(self):
        with mock.patch.dict(os.environ, {"MOONSHOT_API_KEY": "k"}, clear=True):
            names = [a["name"] for a in available_arms()]
        self.assertEqual(names, ["kimi-highspeed", "kimi-standard", "kimi-k3"])

    def test_deepseek_key_adds_two_arms(self):
        with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "d"}, clear=True):
            names = [a["name"] for a in available_arms()]
        self.assertEqual(names, ["deepseek-flash", "deepseek-pro"])

    def test_deepseek_resolves_as_provider(self):
        from agents.llm_client import _resolve_provider
        with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "d"}, clear=True):
            self.assertEqual(_resolve_provider(), "deepseek")
        with mock.patch.dict(os.environ,
                             {"LLM_PROVIDER": "DeepSeek",
                              "MOONSHOT_API_KEY": "k"}, clear=True):
            self.assertEqual(_resolve_provider(), "deepseek")

    def test_both_providers_give_four_arms(self):
        with mock.patch.dict(os.environ,
                             {"MOONSHOT_API_KEY": "k", "ANTHROPIC_API_KEY": "c"},
                             clear=True):
            self.assertEqual(len(available_arms()), 4)

    def test_whitelist(self):
        with mock.patch.dict(os.environ,
                             {"MOONSHOT_API_KEY": "k", "ANTHROPIC_API_KEY": "c",
                              "ROUTER_ARMS": "kimi-highspeed,claude-sonnet"},
                             clear=True):
            self.assertEqual([a["name"] for a in available_arms()],
                             ["kimi-highspeed", "claude-sonnet"])

    def test_escalation_ladder(self):
        arms = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        self.assertEqual(escalation_ladder(arms, "b"), ["b", "c"])
        self.assertEqual(escalation_ladder(arms, "unknown"), ["a", "b", "c"])


# ── Features ──────────────────────────────────────────────────────────────────

class TestFileFeatures(unittest.TestCase):

    def test_feature_shape_and_extends_depth(self):
        index = {
            "class_names": {"PirateShip": "res://ships/pirate.gd"},
            "scripts": {
                "res://ships/pirate.gd": {"extends": "Node2D"},
                "res://ships/squad.gd": {
                    "extends": "PirateShip", "signals": ["a", "b"],
                    "methods": {"m": {}}, "preloads": ["res://x.tscn"],
                    "onready_vars": ["y"], "has_async_methods": True,
                    "is_tool": False,
                },
            },
        }
        x = file_features("ships/squad.gd", index, n_lines=100)
        self.assertEqual(x.shape, (9,))
        self.assertEqual(x[0], 1.0)                     # bias
        self.assertAlmostEqual(x[1], math.log1p(100))   # lines
        self.assertEqual(x[3], 2.0)                     # signals
        self.assertEqual(x[6], 1.0)                     # has_async
        self.assertEqual(x[8], 1.0)                     # extends depth


# ── route_files_node ───────────────────────────────────────────────────────────

class TestRouteFilesNode(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "a.gd").write_text("extends Node\n", encoding="utf-8")
        (root / "b.gd").write_text("extends Node\n" * 50, encoding="utf-8")
        self.state = {**gap._default_state(str(root), True),
                      "scope": [], "index_path": ""}

    def tearDown(self):
        self.tmp.cleanup()

    def test_static_only_skips(self):
        state = {**self.state, "fallback_mode": "static_only"}
        result = gap.route_files_node(state)
        self.assertEqual(result["routing_plan"], {})
        self.assertEqual(result["agent_status"]["route_files"],
                         "skipped:static_only")

    def test_no_router_skips(self):
        state = {**self.state, "no_router": True}
        result = gap.route_files_node(state)
        self.assertEqual(result["agent_status"]["route_files"], "skipped:disabled")

    def test_two_arms_route_all_files(self):
        with mock.patch.dict(os.environ,
                             {"MOONSHOT_API_KEY": "k", "ANTHROPIC_API_KEY": "c"},
                             clear=True):
            result = gap.route_files_node(self.state)
        plan = result["routing_plan"]
        self.assertEqual(set(plan), {"a.gd", "b.gd"})
        for entry in plan.values():
            self.assertIn(entry["arm"], entry["model"] + entry["provider"]
                          + entry["arm"])  # arm name is non-empty
            self.assertEqual(len(entry["context"]), 9)
        self.assertEqual(sum(result["routing_stats"]["assignment"].values()), 2)


# ── Escalation inside per-file retry ──────────────────────────────────────────

class TestEscalation(unittest.TestCase):

    def test_retry_escalates_to_next_tier(self):
        providers = {}

        def provider_for(arm):
            p = mock.Mock()
            p.last_usage = {"prompt_tokens": 100, "completion_tokens": 10}
            providers[arm] = p
            return p

        calls = []

        def fake_review_file(gd, root, index, provider, memory_hints=None):
            calls.append(provider)
            if len(calls) == 1:
                return [], "api error"          # attempt 1 fails → escalate
            return [{"verified": "confirmed"}], None

        with tempfile.TemporaryDirectory() as td:
            gd = Path(td) / "a.gd"
            gd.write_text("extends Node\n", encoding="utf-8")
            with mock.patch("review.review_file", side_effect=fake_review_file), \
                 mock.patch.object(gap.time, "sleep"):
                findings, err, usage, arm_used, escalated, failed_arms = (
                    gap._review_one_file_with_retry(
                        gd, Path(td), {}, mock.Mock(),
                        provider_for=provider_for,
                        arm_ladder=["kimi-highspeed", "kimi-standard"],
                    )
                )
        self.assertIsNone(err)
        self.assertEqual(findings, [{"verified": "confirmed"}])
        self.assertEqual(arm_used, "kimi-standard")
        self.assertTrue(escalated)
        self.assertEqual(len(calls), 2)
        self.assertEqual(failed_arms, ["kimi-highspeed"])

    def test_no_ladder_keeps_single_provider(self):
        provider = mock.Mock()
        provider.last_usage = None
        with tempfile.TemporaryDirectory() as td:
            gd = Path(td) / "a.gd"
            gd.write_text("extends Node\n", encoding="utf-8")
            with mock.patch("review.review_file", return_value=([], None)):
                findings, err, usage, arm_used, escalated, failed_arms = (
                    gap._review_one_file_with_retry(gd, Path(td), {}, provider)
                )
        self.assertIsNone(err)
        self.assertIsNone(arm_used)
        self.assertFalse(escalated)
        self.assertEqual(failed_arms, [])

    def test_auth_error_escalates_immediately_without_sleep(self):
        def provider_for(arm):
            p = mock.Mock()
            p.last_usage = None
            return p

        def always_auth(*args, **kwargs):
            return [], "LLM call failed: Error code: 401 - authentication_error"

        with tempfile.TemporaryDirectory() as td:
            gd = Path(td) / "a.gd"
            gd.write_text("extends Node\n", encoding="utf-8")
            with mock.patch("review.review_file", side_effect=always_auth), \
                 mock.patch.object(gap.time, "sleep") as mock_sleep:
                findings, err, usage, arm_used, escalated, failed_arms = (
                    gap._review_one_file_with_retry(
                        gd, Path(td), {}, mock.Mock(),
                        provider_for=provider_for,
                        arm_ladder=["k1", "k2", "k3"],
                    )
                )
        self.assertIsNotNone(err)
        self.assertEqual(findings, [])
        # all three tiers tried, zero backoff sleeps (auth never heals)
        self.assertEqual(failed_arms, ["k1", "k2", "k3"])
        mock_sleep.assert_not_called()

    def test_auth_error_without_ladder_fails_fast(self):
        provider = mock.Mock()
        provider.last_usage = None
        calls = []

        def always_auth(*args, **kwargs):
            calls.append(1)
            return [], "401 unauthorized"

        with tempfile.TemporaryDirectory() as td:
            gd = Path(td) / "a.gd"
            gd.write_text("extends Node\n", encoding="utf-8")
            with mock.patch("review.review_file", side_effect=always_auth), \
                 mock.patch.object(gap.time, "sleep"):
                _, err, *_ = gap._review_one_file_with_retry(
                    gd, Path(td), {}, provider)
        self.assertIsNotNone(err)
        self.assertEqual(len(calls), 1)  # one shot, no pointless retries


# ── code_review_node integration (mocked LLM) ─────────────────────────────────

class TestCodeReviewRouting(unittest.TestCase):

    def test_bandit_updated_and_stats_returned(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gd = root / "a.gd"
            gd.write_text("extends Node\n", encoding="utf-8")
            router_dir = root / "router"

            fake_memory = mock.Mock()
            fake_memory.retrieve.return_value = []
            fake_memory.curate.return_value = {"added": 0, "updated": 0,
                                               "total": 0}

            state = {
                **gap._default_state(str(root), True),
                "routing_plan": {
                    "a.gd": {"arm": "kimi-standard", "provider": "kimi",
                             "model": "kimi-k2.7-code", "rel_cost": 2.0,
                             "context": [1.0] * 9},
                },
                "routing_stats": {"arms": ["kimi-highspeed", "kimi-standard"]},
            }

            def fake_review_file(*args, **kwargs):
                return [{"verified": "confirmed", "final_severity": "error"}], None

            with mock.patch.dict(os.environ, {"MOONSHOT_API_KEY": "k"},
                                 clear=True), \
                 mock.patch.object(gap, "_ROUTER_DIR", router_dir), \
                 mock.patch.object(gap, "_discover_review_files",
                                   return_value=[gd]), \
                 mock.patch("review.review_file",
                            side_effect=fake_review_file), \
                 mock.patch("agents.review_memory.ReviewMemory",
                            return_value=fake_memory), \
                 mock.patch("agents.llm_client.LLMClient") as mock_client_cls:
                mock_client_cls.return_value.last_usage = {
                    "prompt_tokens": 1000, "completion_tokens": 200}
                result = gap.code_review_node(state)

            stats = result["routing_stats"]
            self.assertEqual(stats["pulls"], {"kimi-standard": 1})
            self.assertIn("kimi-standard", stats["mean_reward"])
            self.assertEqual(stats["escalations"], 0)
            # bandit persisted for the next run
            saved = json.loads((router_dir / "linucb.json").read_text())
            self.assertEqual(saved["pulls"]["kimi-standard"], 1)
            # reward = +1 confirmed − 0.05 × 1.2k tok × rel_cost 2.0 = 0.88
            self.assertAlmostEqual(stats["total_reward"]["kimi-standard"], 0.88)

    def test_failed_file_poisons_assigned_arm(self):
        """A skipped file must still update the bandit — negatively."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gd = root / "a.gd"
            gd.write_text("extends Node\n", encoding="utf-8")
            router_dir = root / "router"

            fake_memory = mock.Mock()
            fake_memory.retrieve.return_value = []
            fake_memory.curate.return_value = {"added": 0, "updated": 0,
                                               "total": 0}

            state = {
                **gap._default_state(str(root), True),
                "routing_plan": {
                    "a.gd": {"arm": "claude-sonnet", "provider": "claude",
                             "model": "claude-sonnet-4-6", "rel_cost": 8.0,
                             "context": [1.0] * 9},
                },
                "routing_stats": {},
            }

            def always_auth(*args, **kwargs):
                return [], "LLM call failed: Error code: 401 - invalid x-api-key"

            with mock.patch.dict(os.environ,
                                 {"MOONSHOT_API_KEY": "k",
                                  "ANTHROPIC_API_KEY": "bad"}, clear=True), \
                 mock.patch.object(gap, "_ROUTER_DIR", router_dir), \
                 mock.patch.object(gap, "_discover_review_files",
                                   return_value=[gd]), \
                 mock.patch("review.review_file", side_effect=always_auth), \
                 mock.patch("agents.review_memory.ReviewMemory",
                            return_value=fake_memory), \
                 mock.patch("agents.llm_client.LLMClient"), \
                 mock.patch.object(gap.time, "sleep"):
                result = gap.code_review_node(state)

            self.assertEqual(result["skipped_files"], ["a.gd"])
            stats = result["routing_stats"]
            self.assertEqual(stats["poisoned_arms"], {"claude-sonnet": 1})
            saved = json.loads((router_dir / "linucb.json").read_text())
            self.assertEqual(saved["pulls"]["claude-sonnet"], 1)
            self.assertLess(saved["total_reward"]["claude-sonnet"], 0.0)


class TestRouteFilesBrokenProvider(unittest.TestCase):

    def test_broken_provider_excluded_from_plan(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a.gd").write_text("extends Node\n", encoding="utf-8")
            (root / "b.gd").write_text("extends Node\n", encoding="utf-8")
            state = {**gap._default_state(str(root), True),
                     "broken_providers": ["claude"]}
            with mock.patch.dict(os.environ,
                                 {"MOONSHOT_API_KEY": "k",
                                  "ANTHROPIC_API_KEY": "bad"}, clear=True):
                result = gap.route_files_node(state)
        self.assertTrue(result["routing_plan"])
        for entry in result["routing_plan"].values():
            self.assertEqual(entry["provider"], "kimi")


if __name__ == "__main__":
    unittest.main()
