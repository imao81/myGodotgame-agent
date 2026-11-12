"""tests/test_tool_selection.py — tool-selection eval harness (zero API)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.tool_selection_eval import (
    _extract_tool_name,
    live_manifest,
    load_cases,
    load_manifest,
    run_eval,
)


class TestFixture(unittest.TestCase):
    def test_cases_cover_all_tools(self):
        cases = load_cases()
        expects = {c["expect"] for c in cases}
        manifest_names = {t["name"] for t in load_manifest("v0")}
        self.assertEqual(expects - {None}, manifest_names)
        self.assertGreaterEqual(len(cases), 20)
        self.assertIn(None, expects)  # negative cases exist

    def test_v0_manifest_shape(self):
        tools = load_manifest("v0")
        self.assertEqual(len(tools), 6)
        for t in tools:
            self.assertIn("name", t)
            self.assertIn("description", t)
            self.assertTrue(t["description"].strip())


class TestLiveManifest(unittest.TestCase):
    def test_live_manifest_matches_fixture_names(self):
        tools = live_manifest()
        self.assertEqual({t["name"] for t in tools},
                         {t["name"] for t in load_manifest("v0")})
        for t in tools:
            self.assertTrue(t["description"].strip())


class TestParse(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(_extract_tool_name('{"tool": "lookup_symbol"}'),
                         "lookup_symbol")

    def test_null(self):
        self.assertIsNone(_extract_tool_name('{"tool": null}'))

    def test_fenced(self):
        self.assertEqual(_extract_tool_name(
            'Sure!\n```json\n{"tool": "health"}\n```'), "health")

    def test_chatter_around(self):
        self.assertEqual(_extract_tool_name(
            'I would pick {"tool": "ask_codebase"} here.'), "ask_codebase")

    def test_garbage_returns_none(self):
        self.assertIsNone(_extract_tool_name("I have no idea"))


class TestRunEval(unittest.TestCase):
    def test_accuracy_and_invalid_handling(self):
        manifest = [{"name": "a", "description": "da"},
                    {"name": "b", "description": "db"}]
        cases = [{"request": "r1", "expect": "a"},
                 {"request": "r2", "expect": "b"},
                 {"request": "r3", "expect": None},
                 {"request": "r4", "expect": "a"}]
        picks = iter(["a", "b", None, "INVALID:zzz"])
        fake_client = mock.Mock()
        with mock.patch("eval.tool_selection_eval.select_tool",
                        side_effect=lambda c, m, r: next(picks)):
            result = run_eval(manifest, cases, fake_client)
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["accuracy"], 0.75)
        self.assertFalse(result["rows"][3]["ok"])


if __name__ == "__main__":
    unittest.main()
