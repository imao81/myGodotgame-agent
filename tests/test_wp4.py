"""tests/test_wp4.py — WP4 report generator tests.

Validates:
  - enrich_findings: schema, matrix, field presence
  - generate_json_report: structure, all-findings-included (drops present)
  - generate_markdown_report: sections, --show-refuted flag
  - write_reports: files created, parseable

All expectations derived from the live index; nothing hardcoded.

Run:
    python -m unittest tests.test_wp4 -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.index_builder import build_project_index
from tools.parser import sanitize_llm_output
from tools.report_generator import (
    enrich_findings,
    generate_json_report,
    generate_markdown_report,
    write_reports,
)

_PROJECT_ROOT = Path(__file__).parent.parent.parent / "godot-2d-space-game" / "project"


def _load_index() -> dict:
    if not _PROJECT_ROOT.is_dir():
        return {}
    return build_project_index(str(_PROJECT_ROOT))


_INDEX: dict = _load_index()


def _skip_no_game(test: unittest.TestCase) -> None:
    if not _INDEX:
        test.skipTest(f"Game project not found at {_PROJECT_ROOT}")


def _make(severity: str, assertion: dict, message: str = "msg",
          line: int = 1, category: str = "test", file: str = "") -> dict:
    raw = {
        "line": line, "severity": severity, "category": category,
        "message": message, "assertion": assertion,
    }
    if file:
        raw["file"] = file
    return sanitize_llm_output(raw)


# Reusable assertion fixtures
_CONFIRMED_ASSERT = {   # Gun.fire — not async → confirmed
    "type": "method_is_not_async",
    "target_class": "Gun",
    "method": "fire",
}
_REFUTED_ASSERT = {     # Gun IS defined → refuted
    "type": "identifier_not_defined",
    "identifier": "Gun",
}
_UNVERIFIABLE_ASSERT = {  # stub → unverifiable
    "type": "signal_not_declared",
    "signal_name": "x",
    "file": "",
}


# ── enrich_findings ────────────────────────────────────────────────────────────

class TestEnrichFindings(unittest.TestCase):

    def setUp(self):
        _skip_no_game(self)

    # Schema: verified is a string, not bool/None
    def test_verified_field_is_string(self):
        f = _make("error", _CONFIRMED_ASSERT)
        enriched = enrich_findings([f], _INDEX)
        self.assertIsInstance(enriched[0]["verified"], str)
        self.assertIn(enriched[0]["verified"],
                      {"confirmed", "refuted", "unverifiable"})

    # Matrix: error + confirmed → final_severity=error, dormant_bug=False
    def test_matrix_error_confirmed(self):
        f = _make("error", _CONFIRMED_ASSERT)
        r = enrich_findings([f], _INDEX)[0]
        self.assertEqual(r["verified"], "confirmed")
        self.assertEqual(r["final_severity"], "error")
        self.assertFalse(r["dormant_bug"])
        self.assertIsNone(r["trigger_condition"])

    # Matrix: error + refuted → final_severity=drop (NOT excluded from list)
    def test_matrix_error_refuted_is_drop_not_excluded(self):
        f = _make("error", _REFUTED_ASSERT)
        enriched = enrich_findings([f], _INDEX)
        self.assertEqual(len(enriched), 1,
                         "refuted finding must remain in list with final_severity=drop")
        r = enriched[0]
        self.assertEqual(r["verified"], "refuted")
        self.assertEqual(r["final_severity"], "drop")
        self.assertFalse(r["dormant_bug"])

    # Matrix: error + unverifiable → final_severity=warning, dormant_bug=True
    def test_matrix_error_unverifiable_is_dormant(self):
        f = _make("error", _UNVERIFIABLE_ASSERT,
                  message="Triggers when signal fires with wrong args.")
        r = enrich_findings([f], _INDEX)[0]
        self.assertEqual(r["verified"], "unverifiable")
        self.assertEqual(r["final_severity"], "warning")
        self.assertTrue(r["dormant_bug"])
        self.assertIsNotNone(r["trigger_condition"])
        self.assertIn("Triggers", r["trigger_condition"])

    # Matrix: warning + any → warning, dormant_bug=False
    def test_matrix_warning_any(self):
        for assert_ in (_CONFIRMED_ASSERT, _REFUTED_ASSERT, _UNVERIFIABLE_ASSERT):
            with self.subTest(assertion_type=assert_["type"]):
                f = _make("warning", assert_)
                r = enrich_findings([f], _INDEX)[0]
                self.assertEqual(r["final_severity"], "warning")
                self.assertFalse(r["dormant_bug"])

    # Matrix: info + any → info
    def test_matrix_info_any(self):
        f = _make("info", _UNVERIFIABLE_ASSERT)
        r = enrich_findings([f], _INDEX)[0]
        self.assertEqual(r["final_severity"], "info")
        self.assertFalse(r["dormant_bug"])

    # All required WP4 output fields must be present
    def test_required_fields_present(self):
        required = {"verified", "final_severity", "dormant_bug", "trigger_condition"}
        f = _make("warning", _UNVERIFIABLE_ASSERT)
        r = enrich_findings([f], _INDEX)[0]
        for key in required:
            self.assertIn(key, r, f"missing field '{key}'")

    # Original LLM severity is preserved unchanged
    def test_llm_severity_preserved(self):
        f = _make("error", _CONFIRMED_ASSERT)
        r = enrich_findings([f], _INDEX)[0]
        self.assertEqual(r["severity"], "error")

    # trigger_condition is None for non-dormant findings
    def test_trigger_condition_none_when_not_dormant(self):
        for sev in ("error", "warning", "info"):
            assert_ = _CONFIRMED_ASSERT if sev != "warning" else _UNVERIFIABLE_ASSERT
            f = _make(sev, assert_)
            r = enrich_findings([f], _INDEX)[0]
            if not r["dormant_bug"]:
                self.assertIsNone(r["trigger_condition"],
                                  f"trigger_condition should be None for sev={sev}")

    # Batch: mixed findings all enriched
    def test_batch_preserves_order(self):
        findings = [
            _make("error", _CONFIRMED_ASSERT, line=1),
            _make("error", _REFUTED_ASSERT, line=2),
            _make("error", _UNVERIFIABLE_ASSERT, line=3),
        ]
        enriched = enrich_findings(findings, _INDEX)
        self.assertEqual(len(enriched), 3)
        self.assertEqual([r["line"] for r in enriched], [1, 2, 3])
        self.assertEqual(enriched[0]["final_severity"], "error")
        self.assertEqual(enriched[1]["final_severity"], "drop")
        self.assertEqual(enriched[2]["final_severity"], "warning")

    # Empty input → empty output
    def test_empty_input(self):
        self.assertEqual(enrich_findings([], _INDEX), [])


# ── generate_json_report ───────────────────────────────────────────────────────

class TestGenerateJsonReport(unittest.TestCase):

    def setUp(self):
        _skip_no_game(self)

    def _make_enriched(self):
        findings = [
            _make("error", _CONFIRMED_ASSERT, line=1),     # error
            _make("error", _REFUTED_ASSERT, line=2),       # drop
            _make("error", _UNVERIFIABLE_ASSERT, line=3),  # warning + dormant
            _make("warning", _UNVERIFIABLE_ASSERT, line=4), # warning
            _make("info", _UNVERIFIABLE_ASSERT, line=5),   # info
        ]
        return enrich_findings(findings, _INDEX)

    def test_report_structure(self):
        enriched = self._make_enriched()
        report = generate_json_report(enriched)
        for key in ("generated_at", "metadata", "summary", "findings"):
            self.assertIn(key, report)

    def test_json_includes_all_findings_including_drops(self):
        enriched = self._make_enriched()
        report = generate_json_report(enriched)
        self.assertEqual(len(report["findings"]), 5,
                         "JSON must include all findings (drops included)")
        drop_in_json = any(f["final_severity"] == "drop"
                           for f in report["findings"])
        self.assertTrue(drop_in_json,
                        "drop findings must appear in JSON with final_severity='drop'")

    def test_summary_counts(self):
        enriched = self._make_enriched()
        report = generate_json_report(enriched)
        s = report["summary"]
        self.assertEqual(s["total"], 5)
        self.assertEqual(s["verified_errors"], 1)
        self.assertGreaterEqual(s["dormant_bugs"], 1)
        self.assertGreaterEqual(s["refuted_drops"], 1)

    def test_json_is_serialisable(self):
        enriched = self._make_enriched()
        report = generate_json_report(enriched)
        # Should not raise
        json_str = json.dumps(report, indent=2)
        roundtrip = json.loads(json_str)
        self.assertEqual(len(roundtrip["findings"]), 5)

    def test_metadata_passthrough(self):
        enriched = self._make_enriched()
        meta = {"project_path": "/test", "godot_version": "4.6"}
        report = generate_json_report(enriched, metadata=meta)
        self.assertEqual(report["metadata"]["project_path"], "/test")


# ── generate_markdown_report ───────────────────────────────────────────────────

class TestGenerateMarkdownReport(unittest.TestCase):

    def setUp(self):
        _skip_no_game(self)

    def _enriched(self):
        findings = [
            _make("error", _CONFIRMED_ASSERT, message="Crash on fire.", line=10),
            _make("error", _REFUTED_ASSERT, message="Gun undefined.", line=20),
            _make("error", _UNVERIFIABLE_ASSERT,
                  message="Signal mismatch. Triggers on event.", line=30),
            _make("warning", _UNVERIFIABLE_ASSERT, message="Code smell.", line=40),
        ]
        return enrich_findings(findings, _INDEX)

    def test_verified_errors_section_present(self):
        md = generate_markdown_report(self._enriched())
        self.assertIn("Verified Errors", md)

    def test_dormant_bugs_section_present(self):
        md = generate_markdown_report(self._enriched())
        self.assertIn("Dormant Bugs", md)

    def test_warnings_section_present(self):
        md = generate_markdown_report(self._enriched())
        self.assertIn("Warnings", md)

    def test_refuted_not_in_default_report(self):
        md = generate_markdown_report(self._enriched(), show_refuted=False)
        self.assertNotIn("Refuted Findings", md,
                         "Refuted section must be hidden by default")

    def test_refuted_shown_with_flag(self):
        md = generate_markdown_report(self._enriched(), show_refuted=True)
        self.assertIn("Refuted Findings", md)
        self.assertIn("REFUTED", md)

    def test_drop_finding_not_in_main_sections(self):
        """A refuted finding (final_severity=drop) must not leak into main sections."""
        enriched = self._enriched()
        drop_msg = next(f["message"] for f in enriched if f["final_severity"] == "drop")
        md = generate_markdown_report(enriched, show_refuted=False)
        self.assertNotIn(drop_msg, md,
                         "Drop finding message must not appear in main report")

    def test_trigger_condition_appears_for_dormant(self):
        md = generate_markdown_report(self._enriched())
        self.assertIn("Trigger condition", md)

    def test_show_refuted_contains_developer_note(self):
        md = generate_markdown_report(self._enriched(), show_refuted=True)
        self.assertIn("Developer section", md)

    def test_summary_line_in_header(self):
        md = generate_markdown_report(self._enriched())
        self.assertIn("Summary", md)
        self.assertIn("verified errors", md)

    def test_metadata_in_header(self):
        enriched = self._enriched()
        md = generate_markdown_report(enriched, metadata={"project_path": "/myproject"})
        self.assertIn("/myproject", md)


# ── write_reports ──────────────────────────────────────────────────────────────

class TestWriteReports(unittest.TestCase):

    def setUp(self):
        _skip_no_game(self)

    def test_files_created(self):
        findings = [_make("warning", _UNVERIFIABLE_ASSERT)]
        enriched = enrich_findings(findings, _INDEX)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            json_p, md_p = write_reports(enriched, out)
            self.assertTrue(json_p.exists())
            self.assertTrue(md_p.exists())
            # JSON must be parseable
            data = json.loads(json_p.read_text())
            self.assertIn("findings", data)
            # Markdown must be non-empty
            self.assertGreater(len(md_p.read_text(encoding="utf-8")), 0)

    def test_show_refuted_flag_propagated(self):
        findings = [_make("error", _REFUTED_ASSERT, message="Gun undefined.")]
        enriched = enrich_findings(findings, _INDEX)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            _, md_p = write_reports(enriched, out, show_refuted=True)
            self.assertIn("Refuted Findings", md_p.read_text(encoding="utf-8"))

    def test_output_dir_created_if_not_exists(self):
        findings = [_make("info", _UNVERIFIABLE_ASSERT)]
        enriched = enrich_findings(findings, _INDEX)
        with tempfile.TemporaryDirectory() as tmp:
            nested = Path(tmp) / "deep" / "nested"
            write_reports(enriched, nested)
            self.assertTrue(nested.exists())

    def test_json_drop_findings_in_file(self):
        findings = [_make("error", _REFUTED_ASSERT)]
        enriched = enrich_findings(findings, _INDEX)
        with tempfile.TemporaryDirectory() as tmp:
            json_p, _ = write_reports(enriched, Path(tmp))
            data = json.loads(json_p.read_text())
            self.assertEqual(data["findings"][0]["final_severity"], "drop")


if __name__ == "__main__":
    unittest.main()
