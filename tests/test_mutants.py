"""tests/test_mutants.py — mutation operators + mutant-set machinery.

Pure text transforms, no LLM, no Godot.

Run:
    python -m pytest tests/test_mutants.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.mutant_eval import _entity_field, build_mutant_project
from eval.mutators import apply_mutator

SAMPLE = """extends Node2D

const AsteroidScene := preload("res://world/docking_point/asteroid.tscn")

func spawn(asteroid: Node) -> void:
	Events.asteroid_spawned.emit(asteroid)
	var g := Gun.new()
	add_child(g)
"""


class TestMutators(unittest.TestCase):

    def test_rename_emitted_signal(self):
        out = apply_mutator("rename_emitted_signal", SAMPLE, "a.gd", "m001")
        text, rec = out
        self.assertIn("Events.asteroid_spawned_broken.emit(", text)
        self.assertEqual(rec.expected_assertion, "signal_not_declared")
        self.assertEqual(rec.expected_entity, "asteroid_spawned_broken")
        self.assertEqual(rec.line, 6)

    def test_dead_guard_emission(self):
        text, rec = apply_mutator("dead_guard_emission", SAMPLE, "a.gd", "m002")
        self.assertIn("if false:", text)
        self.assertIn("\t\tEvents.asteroid_spawned.emit(", text)
        self.assertEqual(rec.expected_assertion, "runtime_signal_not_emitted")
        self.assertEqual(rec.expected_entity, "asteroid_spawned")

    def test_break_preload_path(self):
        text, rec = apply_mutator("break_preload_path", SAMPLE, "a.gd", "m003")
        self.assertIn('preload("res://world/docking_point/asteroid_missing.tscn")',
                      text)
        self.assertEqual(rec.expected_assertion, "file_not_exists")
        self.assertEqual(rec.line, 3)

    def test_rename_class_reference(self):
        text, rec = apply_mutator("rename_class_reference", SAMPLE, "a.gd",
                                  "m004", known_classes={"Gun"})
        self.assertIn("GunMissing.new()", text)
        self.assertEqual(rec.expected_assertion, "identifier_not_defined")

    def test_rename_class_reference_unknown_class_skipped(self):
        # Gun not in the index → no mutation (we only corrupt known classes)
        self.assertIsNone(
            apply_mutator("rename_class_reference", SAMPLE, "a.gd", "m005",
                          known_classes=set()))

    def test_no_site_returns_none(self):
        self.assertIsNone(
            apply_mutator("rename_emitted_signal", "extends Node\n", "a.gd",
                          "m006"))

    def test_mutated_gdscript_stays_parseable_shape(self):
        """Mutants change lines, not structure: tabs/indents survive."""
        for op in ("rename_emitted_signal", "dead_guard_emission",
                   "break_preload_path"):
            text, _ = apply_mutator(op, SAMPLE, "a.gd", "mX")
            self.assertIn("func spawn", text)
            self.assertTrue(text.endswith("\n"))


class TestEntityField(unittest.TestCase):

    def test_mapping(self):
        rec = {"expected_assertion": "signal_not_declared",
               "expected_entity": "x"}
        self.assertEqual(_entity_field(rec), {"signal_name": "x"})
        rec["expected_assertion"] = "unverifiable"
        self.assertEqual(_entity_field(rec), {})


class TestBuildMutantProject(unittest.TestCase):

    def test_build_injects_and_records(self):
        with tempfile.TemporaryDirectory() as src, \
             tempfile.TemporaryDirectory() as dst:
            src_p = Path(src)
            (src_p / "project.godot").write_text("[application]\n",
                                                 encoding="utf-8")
            (src_p / "a.gd").write_text(SAMPLE, encoding="utf-8")
            (src_p / "b.gd").write_text("extends Node\n", encoding="utf-8")

            import eval.mutant_eval as me
            with mock.patch.object(me, "_MUTANT_SET",
                                   Path(dst) / "mutant_set.jsonl"):
                records = build_mutant_project(src_p, Path(dst) / "work",
                                               seed=1, kimi_dir=Path(dst))

            self.assertEqual(len(records), 1)  # only a.gd has a site
            self.assertEqual(records[0]["file"], "a.gd")
            # source untouched, copy mutated
            self.assertEqual((src_p / "a.gd").read_text(), SAMPLE)
            mutated = (Path(dst) / "work" / "a.gd").read_text()
            self.assertNotEqual(mutated, SAMPLE)
            # ground truth persisted
            saved = [json.loads(l) for l in
                     (Path(dst) / "mutant_set.jsonl").read_text().splitlines()]
            self.assertEqual(len(saved), 1)


class TestIndexCacheIsolation(unittest.TestCase):
    """Regression: a cached index must never be served for a different project."""

    def test_cache_rejects_other_project(self):
        import review
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            proj_a = d / "proj_a"
            proj_b = d / "proj_b"
            for p, body in ((proj_a, SAMPLE), (proj_b, "extends Node\n")):
                p.mkdir()
                (p / "project.godot").write_text("[application]\n",
                                                 encoding="utf-8")
                (p / "x.gd").write_text(body, encoding="utf-8")
            kimi = d / ".kimi"

            idx_a, _ = review.load_or_build_index(proj_a, force_rebuild=True,
                                                  kimi_dir=kimi)
            self.assertTrue(idx_a.get("scripts"))

            # B's mtimes are older than A's snapshot — without the project
            # path check, B would be served A's index.
            idx_b, _ = review.load_or_build_index(proj_b, kimi_dir=kimi)
            self.assertTrue(idx_b.get("scripts"))
            # same filename on purpose: the difference must show in CONTENT
            self.assertNotEqual(idx_a["scripts"]["res://x.gd"],
                                idx_b["scripts"]["res://x.gd"])
            self.assertEqual(idx_b.get("project_root"), str(proj_b.resolve()))


if __name__ == "__main__":
    unittest.main()
