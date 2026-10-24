"""tests/test_verifier_extended.py — tests for the three previously-stubbed
assertion verifiers: signal_not_declared, node_not_found, file_not_exists.

Uses a small synthetic index — does NOT require the real game project.

Run:
    python -m pytest tests/test_verifier_extended.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.assertion_verifier import verify_assertion


_INDEX = {
    "scripts": {
        "res://autoload/events.gd": {
            "class_name": None,
            "extends": "Node",
            "signals": ["player_died", "mine_started"],
            "methods": {},
        },
        "res://ships/base_ship.gd": {
            "class_name": "BaseShip",
            "extends": "CharacterBody2D",
            "signals": ["ship_damaged"],
            "methods": {},
        },
        "res://ships/player/player_ship.gd": {
            "class_name": "PlayerShip",
            "extends": "BaseShip",
            "signals": [],
            "methods": {},
        },
    },
    "scenes": {
        "res://ships/player/player_ship.tscn": {
            "root_type": "CharacterBody2D",
            "nodes": {
                "$Gun": {"type": "Node2D", "unique_name": False},
                "%HealthBar": {"type": "ProgressBar", "unique_name": True},
            },
        },
    },
    "autoloads": {"Events": "res://autoload/events.gd"},
    "class_names": {"BaseShip": "res://ships/base_ship.gd",
                    "PlayerShip": "res://ships/player/player_ship.gd"},
    "files": [
        "res://autoload/events.gd",
        "res://ships/player/player_ship.gd",
        "res://ships/player/player_ship.tscn",
        "res://img/laser.png",
    ],
}


class TestSignalNotDeclared(unittest.TestCase):

    def test_declared_in_file_is_refuted(self):
        r = verify_assertion(
            {"type": "signal_not_declared", "signal_name": "player_died",
             "file": "res://autoload/events.gd"}, _INDEX)
        self.assertEqual(r["status"], "refuted")

    def test_declared_in_ancestor_is_refuted(self):
        """PlayerShip extends BaseShip which declares ship_damaged."""
        r = verify_assertion(
            {"type": "signal_not_declared", "signal_name": "ship_damaged",
             "file": "res://ships/player/player_ship.gd"}, _INDEX)
        self.assertEqual(r["status"], "refuted")
        self.assertIn("ancestor", r["details"])

    def test_truly_missing_is_confirmed(self):
        r = verify_assertion(
            {"type": "signal_not_declared", "signal_name": "game_over",
             "file": "res://autoload/events.gd"}, _INDEX)
        self.assertEqual(r["status"], "confirmed")
        self.assertIsNone(r["fix_suggestion"])

    def test_case_typo_gets_fix_suggestion(self):
        r = verify_assertion(
            {"type": "signal_not_declared", "signal_name": "Player_Died",
             "file": "res://autoload/events.gd"}, _INDEX)
        self.assertEqual(r["status"], "confirmed")
        self.assertIn("player_died", r["fix_suggestion"])

    def test_missing_file_field_is_unverifiable(self):
        r = verify_assertion(
            {"type": "signal_not_declared", "signal_name": "x"}, _INDEX)
        self.assertEqual(r["status"], "unverifiable")

    def test_unknown_script_is_unverifiable(self):
        r = verify_assertion(
            {"type": "signal_not_declared", "signal_name": "x",
             "file": "res://nope.gd"}, _INDEX)
        self.assertEqual(r["status"], "unverifiable")

    def test_bare_path_normalised(self):
        r = verify_assertion(
            {"type": "signal_not_declared", "signal_name": "player_died",
             "file": "autoload/events.gd"}, _INDEX)
        self.assertEqual(r["status"], "refuted")


class TestNodeNotFound(unittest.TestCase):

    def test_existing_node_is_refuted(self):
        r = verify_assertion(
            {"type": "node_not_found", "node_path": "$Gun",
             "scene_file": "res://ships/player/player_ship.tscn"}, _INDEX)
        self.assertEqual(r["status"], "refuted")

    def test_unique_name_prefix_fallback(self):
        """%HealthBar found even when asserted with $ prefix."""
        r = verify_assertion(
            {"type": "node_not_found", "node_path": "$HealthBar",
             "scene_file": "res://ships/player/player_ship.tscn"}, _INDEX)
        self.assertEqual(r["status"], "refuted")

    def test_missing_node_is_confirmed(self):
        r = verify_assertion(
            {"type": "node_not_found", "node_path": "$Shield",
             "scene_file": "res://ships/player/player_ship.tscn"}, _INDEX)
        self.assertEqual(r["status"], "confirmed")

    def test_nested_path_leaf_found_reduced_confidence(self):
        r = verify_assertion(
            {"type": "node_not_found", "node_path": "$SomeParent/Gun",
             "scene_file": "res://ships/player/player_ship.tscn"}, _INDEX)
        self.assertEqual(r["status"], "refuted")
        self.assertLessEqual(r["confidence"], 0.9)

    def test_unknown_scene_is_unverifiable(self):
        r = verify_assertion(
            {"type": "node_not_found", "node_path": "$Gun",
             "scene_file": "res://nope.tscn"}, _INDEX)
        self.assertEqual(r["status"], "unverifiable")


class TestFileNotExists(unittest.TestCase):

    def test_existing_file_is_refuted(self):
        r = verify_assertion(
            {"type": "file_not_exists", "path": "res://img/laser.png"}, _INDEX)
        self.assertEqual(r["status"], "refuted")

    def test_missing_file_is_confirmed(self):
        r = verify_assertion(
            {"type": "file_not_exists", "path": "res://img/missile.png"}, _INDEX)
        self.assertEqual(r["status"], "confirmed")
        self.assertIsNone(r["fix_suggestion"])

    def test_same_basename_suggested(self):
        r = verify_assertion(
            {"type": "file_not_exists", "path": "res://vfx/laser.png"}, _INDEX)
        self.assertEqual(r["status"], "confirmed")
        self.assertIn("res://img/laser.png", r["fix_suggestion"])

    def test_missing_path_field_is_unverifiable(self):
        r = verify_assertion({"type": "file_not_exists"}, _INDEX)
        self.assertEqual(r["status"], "unverifiable")

    def test_legacy_index_without_files_partial(self):
        legacy = {k: v for k, v in _INDEX.items() if k != "files"}
        r = verify_assertion(
            {"type": "file_not_exists", "path": "res://img/laser.png"}, legacy)
        self.assertEqual(r["status"], "unverifiable")
        r2 = verify_assertion(
            {"type": "file_not_exists", "path": "res://autoload/events.gd"}, legacy)
        self.assertEqual(r2["status"], "refuted")


class TestDispatchTable(unittest.TestCase):

    def test_no_stub_message_for_implemented_types(self):
        for t, a in [
            ("signal_not_declared", {"signal_name": "x", "file": "res://autoload/events.gd"}),
            ("node_not_found", {"node_path": "$X", "scene_file": "res://ships/player/player_ship.tscn"}),
            ("file_not_exists", {"path": "res://x.gd"}),
        ]:
            r = verify_assertion({"type": t, **a}, _INDEX)
            self.assertNotIn("not yet implemented", r["details"], t)

    def test_unverifiable_type_still_passthrough(self):
        r = verify_assertion({"type": "unverifiable"}, _INDEX)
        self.assertEqual(r["status"], "unverifiable")


if __name__ == "__main__":
    unittest.main()
