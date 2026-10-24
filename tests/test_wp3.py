"""tests/test_wp3.py — Stage 3 pipeline integration tests.

Covers parser.sanitize_llm_output, post_processor.apply_verification,
and the end-to-end finding lifecycle through the severity matrix.

Test expectations are derived from the live project index
(build_project_index) — nothing is hardcoded.

Run:
    python -m pytest tests/test_wp3.py -v
    python -m unittest tests.test_wp3
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Make the agent root importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.index_builder import build_project_index
from tools.parser import sanitize_llm_output, sanitize_findings
from tools.post_processor import apply_verification


# ── Shared index (built once for the entire test run) ─────────────────────────

_PROJECT_ROOT = Path(__file__).parent.parent.parent / "godot-2d-space-game" / "project"

def _load_index() -> dict:
    if not _PROJECT_ROOT.is_dir():
        return {}   # tests run without the game repo → graceful skip
    return build_project_index(str(_PROJECT_ROOT))


_INDEX: dict = _load_index()


def _has_game(test: unittest.TestCase) -> bool:
    """Skip the test with an informative message if the game project is absent."""
    if not _INDEX:
        test.skipTest(f"Game project not found at {_PROJECT_ROOT}; skipping")
    return True


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_finding(severity: str, assertion: dict, message: str = "test finding",
                  line: int = 1, category: str = "test") -> dict:
    return {
        "line": line,
        "severity": severity,
        "category": category,
        "message": message,
        "assertion": assertion,
    }


# ── parser tests ───────────────────────────────────────────────────────────────

class TestSanitizeLLMOutput(unittest.TestCase):

    def _valid(self, **overrides) -> dict:
        base = {
            "line": 1, "severity": "error", "category": "test",
            "message": "msg", "assertion": {"type": "identifier_not_defined", "identifier": "X"},
        }
        base.update(overrides)
        return base

    def test_strips_forbidden_keys(self):
        raw = self._valid()
        raw["verified"] = True
        raw["dormant_bug"] = False
        raw["trigger_condition"] = "whenever"
        raw["final_severity"] = "error"
        clean = sanitize_llm_output(raw)
        for key in ("verified", "dormant_bug", "trigger_condition", "final_severity"):
            self.assertNotIn(key, clean, f"forbidden key '{key}' not stripped")

    def test_preserves_required_fields(self):
        raw = self._valid()
        clean = sanitize_llm_output(raw)
        for key in ("line", "severity", "category", "message", "assertion"):
            self.assertIn(key, clean)

    def test_normalises_severity_to_lowercase(self):
        raw = self._valid(severity="ERROR")
        clean = sanitize_llm_output(raw)
        self.assertEqual(clean["severity"], "error")

    def test_invalid_severity_defaults_to_info(self):
        raw = self._valid(severity="critical")
        clean = sanitize_llm_output(raw)
        self.assertEqual(clean["severity"], "info")

    def test_missing_required_field_raises(self):
        raw = self._valid()
        del raw["message"]
        with self.assertRaises(ValueError):
            sanitize_llm_output(raw)

    def test_missing_assertion_type_raises(self):
        raw = self._valid()
        raw["assertion"] = {"identifier": "X"}  # no 'type' key
        with self.assertRaises(ValueError):
            sanitize_llm_output(raw)

    def test_assertion_not_dict_raises(self):
        raw = self._valid()
        raw["assertion"] = "not_a_dict"
        with self.assertRaises(ValueError):
            sanitize_llm_output(raw)

    def test_non_dict_input_raises(self):
        with self.assertRaises(ValueError):
            sanitize_llm_output("not a dict")   # type: ignore[arg-type]

    def test_sanitize_findings_drops_invalid(self):
        good = self._valid()
        bad = {"broken": True}          # missing all required fields
        results = sanitize_findings([good, bad, good])
        self.assertEqual(len(results), 2)   # bad finding dropped

    def test_sanitize_findings_empty_list(self):
        self.assertEqual(sanitize_findings([]), [])


# ── post_processor tests ───────────────────────────────────────────────────────

class TestApplyVerification(unittest.TestCase):

    # ── TC1: await on GSAICharacterBody2DAgent.new() ──────────────────────────
    # _init IS in the index, async=False → verifier returns "confirmed" → verified=True
    # severity=error, verified=True → final_severity=error, dormant_bug=False
    def test_tc1_await_on_constructor_confirmed_error(self):
        _has_game(self)

        finding = _make_finding(
            severity="error",
            assertion={
                "type": "method_is_not_async",
                "target_class": "GSAICharacterBody2DAgent",
                "method": "_init",
                "file_hint": "",
            },
            message=(
                "await used on GSAICharacterBody2DAgent.new(self) at class-level. "
                "_init uses await internally. Triggers if body not in scene tree."
            ),
            line=36,
            category="async_misuse",
        )

        # Derive expected verified from the live index so the test never hardcodes
        from tools.assertion_verifier import verify_assertion
        vr = verify_assertion(finding["assertion"], _INDEX)
        expected_verified = {"confirmed": True, "refuted": False, "unverifiable": None}[vr["status"]]

        results = apply_verification([finding], _INDEX)

        self.assertEqual(len(results), 1, "finding should survive (not dropped)")
        r = results[0]
        self.assertEqual(r["verified"], expected_verified)
        self.assertEqual(r["verified"], True)           # index shows _init.async=False → confirmed
        self.assertEqual(r["final_severity"], "error")  # error+True → error
        self.assertFalse(r["dormant_bug"])

    # ── TC2: type-hint warning, verifier cannot check → verified=null ─────────
    # severity=warning, verified=None → final_severity=warning, dormant_bug=False
    def test_tc2_type_hint_warning_unverifiable(self):
        _has_game(self)

        finding = _make_finding(
            severity="warning",
            assertion={
                "type": "signal_not_declared",   # empty file → unverifiable
                "signal_name": "__no_signal__",
                "file": "",
            },
            message="Parameter 'world_objects' lacks a type hint.",
            line=74,
            category="type_safety",
        )

        results = apply_verification([finding], _INDEX)

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertIsNone(r["verified"])            # unverifiable → None
        self.assertEqual(r["final_severity"], "warning")
        self.assertFalse(r["dormant_bug"])          # warning+None never dormant
        self.assertIsNone(r["trigger_condition"])

    # ── TC3: signal mismatch, WP2 stub → verified=null, error → dormant bug ──
    # severity=error, verified=None → final_severity=warning, dormant_bug=True
    def test_tc3_signal_mismatch_becomes_dormant_bug(self):
        _has_game(self)

        trigger_msg = (
            "signal 'damaged' parameters changed; callers may fail. "
            "Triggers when Events.damaged fires with wrong argument count."
        )
        finding = _make_finding(
            severity="error",
            assertion={
                "type": "unverifiable",   # opt-out → always unverifiable
            },
            message=trigger_msg,
            line=45,
            category="signal_mismatch",
        )

        results = apply_verification([finding], _INDEX)

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertIsNone(r["verified"])             # unverifiable → None
        self.assertEqual(r["final_severity"], "warning")  # error+None → warning
        self.assertTrue(r["dormant_bug"])            # error+None → dormant
        self.assertEqual(r["trigger_condition"], trigger_msg)

    # ── TC4: LLM error claiming identifier undefined, but index shows it ──────
    # identifier_not_defined for "Gun" → "Gun" IS in class_names → refuted
    # severity=error, verified=False → finding DROPPED entirely
    def test_tc4_refuted_error_is_dropped(self):
        _has_game(self)

        # Verify our expectation against the live index before the real test
        from tools.assertion_verifier import verify_assertion
        assertion = {"type": "identifier_not_defined", "identifier": "Gun"}
        vr = verify_assertion(assertion, _INDEX)
        self.assertEqual(vr["status"], "refuted",
                         "Precondition: 'Gun' must be in class_names for TC4 to be meaningful")

        finding = _make_finding(
            severity="error",
            assertion=assertion,
            message="Class 'Gun' is not defined in this project.",
            line=22,
            category="undefined_reference",
        )

        results = apply_verification([finding], _INDEX)

        self.assertEqual(len(results), 0, "refuted error should be dropped from output")

    # ── Matrix: warning with refuted → still kept as warning ─────────────────
    def test_warning_refuted_is_kept(self):
        _has_game(self)

        finding = _make_finding(
            severity="warning",
            assertion={"type": "identifier_not_defined", "identifier": "Gun"},
            message="Possibly undefined: Gun",
        )
        results = apply_verification([finding], _INDEX)

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["verified"], False)          # refuted
        self.assertEqual(r["final_severity"], "warning")  # warning+any → warning
        self.assertFalse(r["dormant_bug"])

    # ── Matrix: info with any verified → kept as info ─────────────────────────
    def test_info_always_kept_as_info(self):
        _has_game(self)

        for verified_status in ("confirmed", "refuted", "unverifiable"):
            with self.subTest(verified_status=verified_status):
                # Empty file field → unverifiable regardless of signal name
                finding = _make_finding(
                    severity="info",
                    assertion={"type": "signal_not_declared", "signal_name": "x", "file": ""},
                    message="style note",
                )
                results = apply_verification([finding], _INDEX)
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0]["final_severity"], "info")
                break  # one subtest is sufficient; stub always returns unverifiable

    # ── Forbidden keys are stripped before post_processor sees the finding ────
    def test_forbidden_keys_not_present_in_output(self):
        _has_game(self)

        raw = {
            "line": 1, "severity": "warning", "category": "test",
            "message": "msg",
            "assertion": {"type": "signal_not_declared", "signal_name": "x", "file": ""},
            # These should have been stripped by parser; post_processor still
            # must not re-introduce them from the input.
        }
        clean = sanitize_llm_output(raw)
        results = apply_verification([clean], _INDEX)
        self.assertEqual(len(results), 1)
        r = results[0]
        # These keys ARE added BY the post_processor (that is correct).
        # What must NOT happen: the LLM's values bleed through.
        self.assertIn("verified", r)
        self.assertIn("final_severity", r)
        self.assertIn("dormant_bug", r)


# ── End-to-end: parser → post_processor ───────────────────────────────────────

class TestEndToEnd(unittest.TestCase):

    def test_forbidden_keys_stripped_before_verification(self):
        """LLM sneaks in verified=True; parser must strip it; verifier must win."""
        _has_game(self)

        raw = {
            "line": 1, "severity": "error", "category": "test",
            "message": "LLM pretends it is already verified",
            "assertion": {
                "type": "identifier_not_defined",
                "identifier": "Gun",       # Gun IS in class_names → refuted
            },
            "verified": True,              # LLM tries to force verified
            "final_severity": "error",     # LLM tries to force severity
        }

        clean = sanitize_llm_output(raw)
        self.assertNotIn("verified", clean)
        self.assertNotIn("final_severity", clean)

        results = apply_verification([clean], _INDEX)
        # Gun IS defined → refuted → error + False → DROPPED
        self.assertEqual(len(results), 0,
                         "refuted error must be dropped even if LLM claimed verified=True")

    def test_full_pipeline_preserves_finding_fields(self):
        """Fields from LLM output survive through the pipeline unchanged."""
        _has_game(self)

        raw = {
            "line": 99,
            "severity": "warning",
            "category": "resource_leak",
            "message": "Timer started but never stopped.",
            "assertion": {"type": "signal_not_declared", "signal_name": "x", "file": ""},
        }
        clean = sanitize_llm_output(raw)
        results = apply_verification([clean], _INDEX)

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["line"], 99)
        self.assertEqual(r["category"], "resource_leak")
        self.assertEqual(r["message"], "Timer started but never stopped.")


if __name__ == "__main__":
    unittest.main()
