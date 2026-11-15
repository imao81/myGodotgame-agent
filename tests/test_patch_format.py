"""tests/test_patch_format.py — surgical edit-block parsing/apply/boundary."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.patch_format import (
    apply_blocks,
    boundary_report,
    diff_ratio,
    parse_blocks,
)

FILE = "extends Node2D\nvar speed = 100\nvar name_ = \"ship\"\n"

BLOCK = (
    "<<<<<<< ORIGINAL\n"
    "var speed = 100\n"
    "=======\n"
    "var speed = 200\n"
    ">>>>>>> FIXED\n"
)


class TestParse(unittest.TestCase):
    def test_single_block(self):
        r = parse_blocks(BLOCK)
        self.assertEqual(len(r.blocks), 1)
        self.assertEqual(r.blocks[0].original, "var speed = 100")
        self.assertEqual(r.blocks[0].fixed, "var speed = 200")
        self.assertEqual(r.errors, [])

    def test_chatter_around_block(self):
        r = parse_blocks("Here's the fix:\n" + BLOCK + "\nDone.")
        self.assertEqual(len(r.blocks), 1)
        self.assertIn("Here's the fix:", r.leftover)

    def test_multiple_blocks(self):
        r = parse_blocks(BLOCK + "\n" + BLOCK.replace("speed", "name_"))
        self.assertEqual(len(r.blocks), 2)

    def test_malformed_flagged(self):
        r = parse_blocks("<<<<<<< ORIGINAL\nfoo\n")
        self.assertTrue(r.errors)

    def test_no_blocks(self):
        r = parse_blocks("extends Node2D\nvar speed = 200\n")
        self.assertEqual(r.blocks, [])
        self.assertIn("speed = 200", r.leftover)


class TestApply(unittest.TestCase):
    def test_apply_ok(self):
        r = apply_blocks(FILE, parse_blocks(BLOCK).blocks)
        self.assertEqual(r.new_text,
                         "extends Node2D\nvar speed = 200\nvar name_ = \"ship\"\n")
        self.assertEqual(r.errors, [])

    def test_not_found_rejected(self):
        blocks = parse_blocks(BLOCK.replace("var speed = 100", "var speed = 999")).blocks
        r = apply_blocks(FILE, blocks)
        self.assertIsNone(r.new_text)
        self.assertIn("not found", r.errors[0])

    def test_ambiguous_rejected(self):
        blocks = parse_blocks(BLOCK.replace("var speed = 100", "var ")).blocks
        r = apply_blocks(FILE, blocks)  # "var " appears twice
        self.assertIsNone(r.new_text)
        self.assertIn("ambiguous", r.errors[0])

    def test_all_or_nothing(self):
        blocks = parse_blocks(
            BLOCK + BLOCK.replace("var speed = 100", "does_not_exist")
        ).blocks
        r = apply_blocks(FILE, blocks)
        self.assertIsNone(r.new_text)  # first block valid, second fails → nothing


class TestBoundary(unittest.TestCase):
    def test_edit_block_mode_ok(self):
        r = parse_blocks(BLOCK)
        report = boundary_report(FILE, BLOCK, r.blocks, r.leftover)
        self.assertEqual(report["mode"], "edit_blocks")
        self.assertTrue(report["boundary_ok"])

    def test_edit_block_mode_apply_failure_not_ok(self):
        bad = BLOCK.replace("var speed = 100", "var speed = 999")
        r = parse_blocks(bad)
        report = boundary_report(FILE, bad, r.blocks, r.leftover)
        self.assertFalse(report["boundary_ok"])

    def test_whole_file_small_diff_ok(self):
        fixed = FILE.replace("speed = 100", "speed = 200")
        report = boundary_report(FILE, fixed, [], fixed)
        self.assertEqual(report["mode"], "whole_file")
        self.assertTrue(report["boundary_ok"])
        self.assertLess(report["diff_ratio"], 0.4)

    def test_whole_file_large_diff_rejected(self):
        fixed = "completely\ndifferent\nfile\nwith\nmany\nlines\n"
        report = boundary_report(FILE, fixed, [], fixed)
        self.assertFalse(report["boundary_ok"])
        self.assertGreaterEqual(report["diff_ratio"], 0.4)

    def test_diff_ratio_identical(self):
        self.assertEqual(diff_ratio(FILE, FILE), 0.0)


if __name__ == "__main__":
    unittest.main()
