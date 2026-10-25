"""tests/test_runtime_layer.py — runtime verification layer.

Covers:
  - agents/runtime_verifier.verify_runtime: dormant re-judgement matrix
    (fired → refuted; context exercised + silent → confirmed upgrade;
    context not exercised → untouched; no playtest evidence → untouched)
  - agents/runtime_verifier.coverage_notes: res:// conversion, exclusions
  - tools/godot_runner.run_test_controller: TEST_RESULT/COVERAGE/SIGNALS parsing
  - playtest_node / runtime_verifier_node / fix_verify_node: state mapping,
    infra-failure tolerance, regression detection

No LLM, no Godot binary, no real game project required.

Run:
    python -m pytest tests/test_runtime_layer.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.runtime_verifier import coverage_notes, verify_runtime
from game_agent_pipeline import (
    _default_state,
    fix_verify_node,
    playtest_node,
    runtime_verifier_node,
)


def _runtime_finding(signal: str, verified: str = "unverifiable",
                     dormant: bool = True, **assertion_extra) -> dict:
    return {
        "file": "ships/player/player_ship.gd",
        "line": 42,
        "rule": "dead_emission",
        "severity": "error",
        "message": f"Events.{signal} is never emitted",
        "assertion": {"type": "runtime_signal_not_emitted",
                      "signal_name": signal, **assertion_extra},
        "verified": verified,
        "final_severity": "warning" if dormant else "error",
        "dormant_bug": dormant,
        "trigger_condition": "dead path" if dormant else None,
        "_verifier": {"confidence": 0.0, "details": "static stub",
                      "fix_suggestion": None},
    }


class TestVerifyRuntime(unittest.TestCase):

    def test_signal_fired_refutes(self):
        findings = [_runtime_finding("mine_finished")]
        updated, stats = verify_runtime(findings, {"mine_finished": 2})
        self.assertEqual(updated[0]["verified"], "refuted")
        self.assertEqual(updated[0]["final_severity"], "drop")
        self.assertFalse(updated[0]["dormant_bug"])
        self.assertIn("fired 2x", updated[0]["_verifier"]["details"])
        self.assertTrue(updated[0]["_verifier"]["runtime_evidence"])
        self.assertEqual(stats["runtime_refuted"], 1)

    def test_context_exercised_and_silent_confirms(self):
        findings = [_runtime_finding("mine_finished")]
        # mine_started fired (context exercised) but mine_finished never did
        updated, stats = verify_runtime(
            findings, {"mine_started": 3, "mine_finished": 0})
        self.assertEqual(updated[0]["verified"], "confirmed")
        self.assertEqual(updated[0]["final_severity"], "error")
        self.assertFalse(updated[0]["dormant_bug"])
        self.assertIsNone(updated[0]["trigger_condition"])
        self.assertIn("mine_started", updated[0]["_verifier"]["details"])
        self.assertEqual(stats["runtime_confirmed"], 1)

    def test_context_not_exercised_leaves_dormant(self):
        findings = [_runtime_finding("mine_finished")]
        # neither signal fired — scenario never ran; absence proves nothing
        updated, stats = verify_runtime(
            findings, {"mine_started": 0, "mine_finished": 0})
        self.assertEqual(updated[0]["verified"], "unverifiable")
        self.assertTrue(updated[0]["dormant_bug"])
        self.assertEqual(stats["runtime_unchecked"], 1)

    def test_unwatchable_signal_never_confirmed(self):
        # player_died has no reachable context in a spawn-phase playtest
        findings = [_runtime_finding("player_died")]
        updated, stats = verify_runtime(findings, {"player_died": 0})
        self.assertEqual(updated[0]["verified"], "unverifiable")
        self.assertEqual(stats["runtime_unchecked"], 1)

    def test_assertion_context_overrides_default(self):
        findings = [_runtime_finding("upgrade_unlocked",
                                     expected_context=["mine_finished"])]
        updated, stats = verify_runtime(
            findings, {"mine_finished": 1, "upgrade_unlocked": 0})
        self.assertEqual(updated[0]["verified"], "confirmed")
        self.assertEqual(stats["runtime_confirmed"], 1)

    def test_no_playtest_evidence_untouched(self):
        findings = [_runtime_finding("mine_finished")]
        for observed, ok in [({}, True), ({"mine_started": 1}, False)]:
            updated, stats = verify_runtime(findings, observed, playtest_ok=ok)
            self.assertEqual(updated[0]["verified"], "unverifiable")
            self.assertEqual(stats["runtime_unchecked"], 1)

    def test_non_runtime_assertions_untouched(self):
        findings = [
            {"assertion": {"type": "signal_not_declared"},
             "verified": "unverifiable", "dormant_bug": True},
            _runtime_finding("mine_finished", verified="confirmed",
                             dormant=False),
        ]
        updated, stats = verify_runtime(
            findings, {"mine_started": 1, "mine_finished": 0})
        self.assertEqual(updated, findings)
        self.assertEqual(stats["runtime_unchecked"], 0)

    def test_missing_signal_name_unchecked(self):
        f = _runtime_finding("")
        updated, stats = verify_runtime([f], {"x": 1})
        self.assertEqual(updated[0]["verified"], "unverifiable")
        self.assertEqual(stats["runtime_unchecked"], 1)


class TestCoverageNotes(unittest.TestCase):

    def test_uncovered_files_noted(self):
        notes = coverage_notes(
            ["ships/player/player_ship.gd", "world/spawners/station_spawner.gd"],
            ["res://ships/player/player_ship.gd"],
        )
        self.assertEqual(len(notes), 1)
        self.assertIn("station_spawner.gd", notes[0]["message"])
        self.assertEqual(notes[0]["severity"], "info")
        self.assertNotIn("dead code", notes[0]["message"])

    def test_exclusions_and_path_normalisation(self):
        notes = coverage_notes(
            ["autoload\\test_controller.gd", "a.gd"],  # Windows-style sep
            ["res://a.gd"],
        )
        self.assertEqual(notes, [])


class TestGodotRunnerParsing(unittest.TestCase):

    def test_parses_all_line_kinds(self):
        from tools import godot_runner
        fake_output = "\n".join([
            "Godot Engine v4.6.1",
            'TEST_RESULT:{"scenario": "station_spawns_on_load", "passed": true}',
            'SIGNALS:{"station_spawned": 1, "mine_finished": 0}',
            'COVERAGE:["res://ships/player/player_ship.gd"]',
            "TEST_RESULT:not-json",
            'SIGNALS:{"broken": ',
        ])
        with mock.patch.object(godot_runner, "_run", return_value={
                "success": True, "output": fake_output, "errors": []}):
            result = godot_runner.run_test_controller(".", "res://x.gd")
        self.assertEqual(len(result["json_results"]), 1)
        self.assertEqual(result["signals"]["station_spawned"], 1)
        self.assertEqual(result["signals"]["mine_finished"], 0)
        self.assertEqual(result["coverage"], ["res://ships/player/player_ship.gd"])


class TestPlaytestNode(unittest.TestCase):

    def test_passthrough_on_success(self):
        fake = {
            "playtest_results": [{"scenario": "s", "passed": True}],
            "playtest_summary": "1 passed, 0 failed out of 1 scenarios",
            "screenshots": [], "focus_files": [], "balance_issues": [],
            "observed_signals": {"station_spawned": 1},
            "runtime_coverage": ["res://a.gd"],
            "playtest_infra_error": False,
            "agent_status": {"playtest": "ok"},
        }
        with mock.patch("agents.playtest.run", return_value=fake):
            result = playtest_node(_default_state(".", True))
        self.assertEqual(result["observed_signals"], {"station_spawned": 1})
        self.assertFalse(result["playtest_infra_error"])

    def test_exception_degrades_to_skipped(self):
        with mock.patch("agents.playtest.run",
                        side_effect=OSError("godot gone")):
            result = playtest_node(_default_state(".", True))
        self.assertTrue(result["playtest_infra_error"])
        self.assertIn("skipped", result["agent_status"]["playtest"])
        self.assertEqual(result["observed_signals"], {})


class TestRuntimeVerifierNode(unittest.TestCase):

    def test_state_mapping_with_scope(self):
        state = {
            **_default_state(".", True),
            "scope": ["ships/player/player_ship.gd"],
            "findings": [_runtime_finding("mine_finished")],
            "observed_signals": {"mine_started": 2, "mine_finished": 0},
            "runtime_coverage": [],
            "playtest_infra_error": False,
            "meta_findings": [],
        }
        result = runtime_verifier_node(state)
        self.assertEqual(result["findings"][0]["verified"], "confirmed")
        self.assertEqual(result["findings"][0]["final_severity"], "error")
        self.assertEqual(result["meta_findings"][0]["type"], "runtime_coverage")

    def test_no_evidence_is_honest_skip(self):
        state = {
            **_default_state(".", True),
            "scope": ["a.gd"],
            "findings": [_runtime_finding("mine_finished")],
            "observed_signals": {},
            "playtest_infra_error": True,
        }
        result = runtime_verifier_node(state)
        self.assertEqual(result["findings"][0]["verified"], "unverifiable")
        self.assertEqual(result["agent_status"]["runtime_verifier"],
                         "skipped:no_runtime_evidence")
        self.assertEqual(result["meta_findings"], [])


class TestFixVerifyNode(unittest.TestCase):

    @staticmethod
    def _state(**overrides):
        return {
            **_default_state(".", True),
            "auto_fix_approved": True,
            "fix_results": [{"file": "a.gd", "status": "fixed"}],
            "playtest_results": [
                {"scenario": "station_spawns_on_load", "passed": True},
                {"scenario": "player_survives_spawn", "passed": True},
            ],
            "summary": "# Report",
            **overrides,
        }

    def test_regression_detected(self):
        rerun = {
            "playtest_results": [
                {"scenario": "station_spawns_on_load", "passed": False},
                {"scenario": "player_survives_spawn", "passed": True},
            ],
            "playtest_infra_error": False,
        }
        with mock.patch("agents.playtest.run", return_value=rerun):
            result = fix_verify_node(self._state())
        self.assertEqual(result["agent_status"]["fix_verify"], "regression")
        self.assertFalse(result["fix_verify_results"][0]["passed"])
        self.assertIn("station_spawns_on_load",
                      result["fix_verify_results"][0]["message"])
        self.assertIn("Fix Verify", result["summary"])

    def test_no_regression(self):
        rerun = {
            "playtest_results": [
                {"scenario": "station_spawns_on_load", "passed": True},
                {"scenario": "player_survives_spawn", "passed": True},
                {"scenario": "pirates_spawn_on_load", "passed": True},
            ],
            "playtest_infra_error": False,
        }
        with mock.patch("agents.playtest.run", return_value=rerun):
            result = fix_verify_node(self._state())
        self.assertEqual(result["agent_status"]["fix_verify"], "ok")
        self.assertTrue(result["fix_verify_results"][0]["passed"])

    def test_skipped_without_fix(self):
        result = fix_verify_node(self._state(fix_results=[]))
        self.assertEqual(result["agent_status"]["fix_verify"],
                         "skipped:no_fix_applied")

    def test_infra_error_not_a_regression_verdict(self):
        with mock.patch("agents.playtest.run",
                        side_effect=OSError("godot gone")):
            result = fix_verify_node(self._state())
        self.assertEqual(result["agent_status"]["fix_verify"], "infra_error")


if __name__ == "__main__":
    unittest.main()
