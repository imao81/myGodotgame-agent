"""tests/test_verifier_extended.py — tests for ALL six concrete assertion
verifiers with a synthetic index (no dependency on the real game project):

  method_is_not_async, node_type_mismatch, identifier_not_defined  (original 3)
  signal_not_declared, node_not_found, file_not_exists             (added later)

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
            "methods": {
                "fire": {"async": False, "args": [], "line": 10},
                "_on_hit": {"async": True, "args": [], "line": 20},
            },
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


class TestMethodIsNotAsync(unittest.TestCase):

    def test_non_async_method_is_confirmed(self):
        r = verify_assertion(
            {"type": "method_is_not_async", "target_class": "BaseShip",
             "method": "fire"}, _INDEX)
        self.assertEqual(r["status"], "confirmed")

    def test_async_method_is_refuted(self):
        r = verify_assertion(
            {"type": "method_is_not_async", "target_class": "BaseShip",
             "method": "_on_hit"}, _INDEX)
        self.assertEqual(r["status"], "refuted")

    def test_unknown_method_is_unverifiable(self):
        r = verify_assertion(
            {"type": "method_is_not_async", "target_class": "BaseShip",
             "method": "fly"}, _INDEX)
        self.assertEqual(r["status"], "unverifiable")

    def test_file_hint_fallback_resolution(self):
        r = verify_assertion(
            {"type": "method_is_not_async", "target_class": "",
             "method": "fire", "file_hint": "ships/base_ship.gd"}, _INDEX)
        self.assertEqual(r["status"], "confirmed")

    def test_missing_method_field_is_unverifiable(self):
        r = verify_assertion(
            {"type": "method_is_not_async", "target_class": "BaseShip"}, _INDEX)
        self.assertEqual(r["status"], "unverifiable")


class TestNodeTypeMismatch(unittest.TestCase):

    _SCENE = "res://ships/player/player_ship.tscn"

    def test_actual_type_matches_is_confirmed(self):
        r = verify_assertion(
            {"type": "node_type_mismatch", "node_path": "$Gun",
             "expected_type": "CollisionShape3D", "actual_type": "Node2D",
             "scene_file": self._SCENE}, _INDEX)
        self.assertEqual(r["status"], "confirmed")

    def test_expected_type_matches_is_refuted(self):
        r = verify_assertion(
            {"type": "node_type_mismatch", "node_path": "$Gun",
             "expected_type": "Node2D", "actual_type": "CollisionShape3D",
             "scene_file": self._SCENE}, _INDEX)
        self.assertEqual(r["status"], "refuted")

    def test_missing_node_is_confirmed_with_suggestion(self):
        r = verify_assertion(
            {"type": "node_type_mismatch", "node_path": "$Gun2",
             "expected_type": "Node2D", "actual_type": "Node3D",
             "scene_file": self._SCENE}, _INDEX)
        self.assertEqual(r["status"], "confirmed")

    def test_unknown_scene_is_unverifiable(self):
        r = verify_assertion(
            {"type": "node_type_mismatch", "node_path": "$Gun",
             "expected_type": "A", "actual_type": "B",
             "scene_file": "res://nope.tscn"}, _INDEX)
        self.assertEqual(r["status"], "unverifiable")


class TestIdentifierNotDefined(unittest.TestCase):

    def test_defined_class_name_is_refuted(self):
        r = verify_assertion(
            {"type": "identifier_not_defined", "identifier": "BaseShip"}, _INDEX)
        self.assertEqual(r["status"], "refuted")

    def test_registered_autoload_is_refuted(self):
        r = verify_assertion(
            {"type": "identifier_not_defined", "identifier": "Events"}, _INDEX)
        self.assertEqual(r["status"], "refuted")

    def test_case_typo_is_confirmed_with_fix(self):
        r = verify_assertion(
            {"type": "identifier_not_defined", "identifier": "baseship"}, _INDEX)
        self.assertEqual(r["status"], "confirmed")
        self.assertIn("BaseShip", r["fix_suggestion"])

    def test_truly_unknown_is_confirmed(self):
        r = verify_assertion(
            {"type": "identifier_not_defined", "identifier": "Ghost"}, _INDEX)
        self.assertEqual(r["status"], "confirmed")
        self.assertIsNone(r["fix_suggestion"])

    def test_llm_suggestion_validated_against_index(self):
        r = verify_assertion(
            {"type": "identifier_not_defined", "identifier": "Ship",
             "suggested_match": "BaseShip"}, _INDEX)
        self.assertIn("BaseShip", r["fix_suggestion"])


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
