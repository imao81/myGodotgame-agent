"""tests/test_index_builder.py — parsing correctness for the regex index builder.

Builds a tiny synthetic Godot project in a temp dir and asserts the extracted
index.  Everything downstream (verifiers, memory, prompt context) trusts this
index — before this file, nothing asserted the parsing itself.

Run:
    python -m pytest tests/test_index_builder.py -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.index_builder import build_project_index


_PROJECT_GODOT = """\
config_version=5

[application]
config/features=PackedStringArray("4.6")

[autoload]
Events="*res://autoload/events.gd"
"""

_EVENTS_GD = """\
extends Node

signal damaged(target, damage, shooter)
signal player_died
"""

_GUN_GD = """\
class_name Gun
extends Node2D

const Events = preload("res://autoload/events.gd")

@onready var sprite = $Sprite

async func fire() -> void:
	pass

func reload() -> void:
	await get_tree().process_frame
"""

_GUN_TSCN = """\
[gd_scene load_steps=2 format=3]

[ext_resource type="Script" path="res://ships/gun.gd" id="1"]

[node name="Gun" type="Node2D"]
script = ExtResource("1")

[node name="Sprite" type="Sprite2D" parent="."]

[node name="Bar" type="ProgressBar" parent="."]
unique_name_in_owner = true
"""


class TestBuildProjectIndex(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        (root / "project.godot").write_text(_PROJECT_GODOT, encoding="utf-8")
        (root / "autoload").mkdir()
        (root / "autoload" / "events.gd").write_text(_EVENTS_GD, encoding="utf-8")
        (root / "ships").mkdir()
        (root / "ships" / "gun.gd").write_text(_GUN_GD, encoding="utf-8")
        (root / "ships" / "gun.tscn").write_text(_GUN_TSCN, encoding="utf-8")
        cls.index = build_project_index(root)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    # ── project.godot ─────────────────────────────────────────────────────────

    def test_godot_version(self):
        self.assertEqual(self.index["godot_version"], "4.6")

    def test_autoloads(self):
        self.assertEqual(self.index["autoloads"],
                         {"Events": "res://autoload/events.gd"})

    # ── .gd parsing ───────────────────────────────────────────────────────────

    def test_class_names(self):
        self.assertEqual(self.index["class_names"].get("Gun"), "res://ships/gun.gd")

    def test_signals_extracted(self):
        events = self.index["scripts"]["res://autoload/events.gd"]
        self.assertEqual(sorted(events["signals"]), ["damaged", "player_died"])

    def test_signal_signatures_with_params(self):
        sigs = self.index["signal_signatures"]["res://autoload/events.gd"]
        self.assertEqual(sigs["damaged"], ["target", "damage", "shooter"])
        self.assertNotIn("player_died", sigs)  # no params → not recorded

    def test_async_detection(self):
        gun = self.index["scripts"]["res://ships/gun.gd"]
        self.assertTrue(gun["methods"]["fire"]["async"])
        self.assertFalse(gun["methods"]["reload"]["async"])
        # reload has await in body → file-level flag
        self.assertTrue(gun["has_async_methods"])

    def test_preloads_and_onready(self):
        gun = self.index["scripts"]["res://ships/gun.gd"]
        self.assertEqual(gun["preloads"], ["res://autoload/events.gd"])
        self.assertEqual(gun["onready_vars"], ["sprite"])

    # ── .tscn parsing ─────────────────────────────────────────────────────────

    def test_scene_root_and_script(self):
        scene = self.index["scenes"]["res://ships/gun.tscn"]
        self.assertEqual(scene["root_type"], "Node2D")
        self.assertEqual(scene["root_class"], "Gun")
        self.assertEqual(scene["script_path"], "res://ships/gun.gd")

    def test_scene_nodes_dollar_and_percent(self):
        nodes = self.index["scenes"]["res://ships/gun.tscn"]["nodes"]
        self.assertEqual(nodes["$Sprite"]["type"], "Sprite2D")
        self.assertEqual(nodes["%Bar"]["type"], "ProgressBar")
        self.assertTrue(nodes["%Bar"]["unique_name"])

    # ── file listing ──────────────────────────────────────────────────────────

    def test_files_listing(self):
        files = self.index["files"]
        self.assertIn("res://ships/gun.gd", files)
        self.assertIn("res://ships/gun.tscn", files)
        self.assertIn("res://project.godot", files)


if __name__ == "__main__":
    unittest.main()
