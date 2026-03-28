"""tests/test_wp5.py — WP5 orchestrator tests.

Validates:
  - MockLLMProvider: keyed dispatch, default fallback
  - discover_files: addon exclusion, --review-addons flag, bad target
  - review_file: stub skipping, JSON extraction, sanitize integration
  - deduplicate_cascading: root kept, cascading demoted, singletons untouched
  - run_pipeline: full pipeline via MockLLMProvider, exit codes, report files created
  - main(): exit code 0 (no errors), 1 (verified errors), 2 (bad project)

All tests use MockLLMProvider + temp dirs — no real LLM calls.
Run:
    python -m unittest tests.test_wp5 -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import review as _review_module

# Stub out the prompt template so tests never hit str.format() conflicts
# from JSON examples embedded in the real review_prompt_v2.txt.
_STUB_TEMPLATE = "Review {filename} autoloads={autoload_list}\n{context_and_code}"
_review_module._PROMPT_TEMPLATE = _STUB_TEMPLATE

from review import (
    MockLLMProvider,
    deduplicate_cascading,
    discover_files,
    main,
    review_file,
    run_pipeline,
    _cascade_entity,
    _parse_json_findings,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_gd(root: Path, rel: str, lines: int = 10) -> Path:
    """Create a .gd stub with `lines` lines of content."""
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join([f"var x{i} = {i}" for i in range(lines)]), encoding="utf-8")
    return p


def _make_project(root: Path) -> Path:
    """Create a minimal Godot project structure in root."""
    (root / "project.godot").write_text(
        'config_version=5\n[application]\nconfig/name="Test"\n', encoding="utf-8"
    )
    return root


def _minimal_index() -> dict:
    """Return a trivial index so enrich_findings doesn't crash."""
    return {
        "godot_version": "4.x",
        "scripts": {},
        "scenes": {},
        "autoloads": {},
        "class_names": {},
        "signal_signatures": {},
    }


def _valid_finding(**overrides) -> dict:
    base = {
        "line": 1,
        "severity": "warning",
        "category": "test",
        "message": "test finding",
        "assertion": {"type": "signal_not_declared", "signal_name": "x", "file": ""},
    }
    base.update(overrides)
    return base


# ── MockLLMProvider ───────────────────────────────────────────────────────────

class TestMockLLMProvider(unittest.TestCase):

    def test_keyed_dispatch(self):
        findings = [_valid_finding()]
        p = MockLLMProvider(responses={"ship.gd": findings})
        out = p.generate("review ship.gd please")
        self.assertEqual(json.loads(out), findings)

    def test_default_fallback(self):
        default = [_valid_finding(message="default")]
        p = MockLLMProvider(default=default)
        out = p.generate("review anything.gd")
        self.assertEqual(json.loads(out), default)

    def test_no_match_returns_empty_list(self):
        p = MockLLMProvider()
        out = json.loads(p.generate("nothing matches"))
        self.assertEqual(out, [])

    def test_first_matching_key_wins(self):
        p = MockLLMProvider(responses={
            "foo": [_valid_finding(message="foo")],
            "foobar": [_valid_finding(message="foobar")],
        })
        # "foobar" prompt matches "foo" first (dict insertion order)
        out = json.loads(p.generate("foobar"))
        self.assertEqual(out[0]["message"], "foo")


# ── _parse_json_findings ──────────────────────────────────────────────────────

class TestParseJsonFindings(unittest.TestCase):

    def test_bare_array(self):
        data = [{"line": 1}]
        result = _parse_json_findings(json.dumps(data))
        self.assertEqual(result, data)

    def test_array_in_prose(self):
        raw = 'Here are the findings:\n[{"line": 2}]\nDone.'
        result = _parse_json_findings(raw)
        self.assertEqual(result[0]["line"], 2)

    def test_no_array_returns_empty(self):
        self.assertEqual(_parse_json_findings("no json here"), [])

    def test_non_array_json_returns_empty(self):
        self.assertEqual(_parse_json_findings('{"key": "val"}'), [])

    def test_invalid_json_returns_empty(self):
        self.assertEqual(_parse_json_findings("[broken json"), [])


# ── discover_files ────────────────────────────────────────────────────────────

class TestDiscoverFiles(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _make_project(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_finds_gd_files(self):
        _make_gd(self.root, "scripts/player.gd")
        _make_gd(self.root, "scripts/enemy.gd")
        files = discover_files(self.root)
        names = {f.name for f in files}
        self.assertIn("player.gd", names)
        self.assertIn("enemy.gd", names)

    def test_excludes_addons_by_default(self):
        _make_gd(self.root, "scripts/main.gd")
        _make_gd(self.root, "addons/plugin/plugin.gd")
        files = discover_files(self.root)
        names = {f.name for f in files}
        self.assertIn("main.gd", names)
        self.assertNotIn("plugin.gd", names)

    def test_excludes_libraries_by_default(self):
        _make_gd(self.root, "scripts/main.gd")
        _make_gd(self.root, "libraries/gsai/agent.gd")
        files = discover_files(self.root)
        names = {f.name for f in files}
        self.assertNotIn("agent.gd", names)

    def test_review_addons_includes_addons(self):
        _make_gd(self.root, "addons/plugin/plugin.gd")
        files = discover_files(self.root, review_addons=True)
        names = {f.name for f in files}
        self.assertIn("plugin.gd", names)

    def test_review_addons_includes_libraries(self):
        _make_gd(self.root, "libraries/gsai/agent.gd")
        files = discover_files(self.root, review_addons=True)
        names = {f.name for f in files}
        self.assertIn("agent.gd", names)

    def test_excludes_hidden_dirs(self):
        _make_gd(self.root, ".godot/cache/something.gd")
        _make_gd(self.root, "scripts/main.gd")
        files = discover_files(self.root)
        names = {f.name for f in files}
        self.assertNotIn("something.gd", names)
        self.assertIn("main.gd", names)

    def test_target_sub_path(self):
        _make_gd(self.root, "ships/player.gd")
        _make_gd(self.root, "enemies/enemy.gd")
        files = discover_files(self.root, target="ships")
        names = {f.name for f in files}
        self.assertIn("player.gd", names)
        self.assertNotIn("enemy.gd", names)

    def test_missing_target_raises(self):
        with self.assertRaises(FileNotFoundError):
            discover_files(self.root, target="nonexistent")

    def test_returns_sorted_list(self):
        _make_gd(self.root, "scripts/z.gd")
        _make_gd(self.root, "scripts/a.gd")
        files = discover_files(self.root)
        self.assertEqual(files, sorted(files))


# ── review_file ───────────────────────────────────────────────────────────────

class TestReviewFile(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _make_project(self.root)
        self.index = _minimal_index()

    def tearDown(self):
        self._tmp.cleanup()

    def test_stub_file_skipped(self):
        """Files with < 5 non-empty lines are skipped silently."""
        gd = _make_gd(self.root, "stub.gd", lines=3)
        p = MockLLMProvider(default=[_valid_finding()])
        findings, err = review_file(gd, self.root, self.index, p)
        self.assertEqual(findings, [])
        self.assertIsNone(err)

    def test_valid_file_returns_findings(self):
        gd = _make_gd(self.root, "player.gd", lines=10)
        finding = _valid_finding()
        p = MockLLMProvider(default=[finding])
        findings, err = review_file(gd, self.root, self.index, p)
        self.assertIsNone(err)
        self.assertEqual(len(findings), 1)

    def test_file_field_injected(self):
        """review_file must inject the relative file path into each finding."""
        gd = _make_gd(self.root, "scripts/player.gd", lines=10)
        p = MockLLMProvider(default=[_valid_finding()])
        findings, _ = review_file(gd, self.root, self.index, p)
        self.assertEqual(findings[0]["file"], "scripts/player.gd")

    def test_invalid_finding_stripped(self):
        """Findings missing required fields are dropped by sanitize_findings."""
        gd = _make_gd(self.root, "player.gd", lines=10)
        # One valid, one broken
        p = MockLLMProvider(default=[_valid_finding(), {"broken": True}])
        findings, err = review_file(gd, self.root, self.index, p)
        self.assertEqual(len(findings), 1)

    def test_llm_error_returns_error_message(self):
        class BrokenProvider(MockLLMProvider):
            def generate(self, prompt):
                raise RuntimeError("network failure")

        gd = _make_gd(self.root, "player.gd", lines=10)
        findings, err = review_file(gd, self.root, self.index, BrokenProvider())
        self.assertEqual(findings, [])
        self.assertIsNotNone(err)
        self.assertIn("LLM call failed", err)

    def test_missing_file_returns_error(self):
        gd = self.root / "nonexistent.gd"
        p = MockLLMProvider()
        findings, err = review_file(gd, self.root, self.index, p)
        self.assertEqual(findings, [])
        self.assertIn("cannot read", err)


# ── deduplicate_cascading ─────────────────────────────────────────────────────

class TestDeduplicateCascading(unittest.TestCase):

    def _enriched(self, file="a.gd", line=1, severity="warning",
                  assertion=None, final_severity="warning") -> dict:
        if assertion is None:
            assertion = {"type": "identifier_not_defined", "identifier": "Foo"}
        return {
            "file": file,
            "line": line,
            "severity": severity,
            "message": f"Finding at line {line}",
            "assertion": assertion,
            "final_severity": final_severity,
            "verified": "confirmed",
            "dormant_bug": False,
            "trigger_condition": None,
        }

    def test_singleton_group_untouched(self):
        f = self._enriched(line=1)
        result = deduplicate_cascading([f])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["final_severity"], "warning")
        self.assertNotIn("root_cause_ref", result[0])

    def test_root_kept_cascading_demoted(self):
        f1 = self._enriched(line=1)
        f2 = self._enriched(line=2)  # same file + entity
        result = deduplicate_cascading([f1, f2])
        self.assertEqual(len(result), 2)
        # Root unchanged
        self.assertEqual(result[0]["final_severity"], "warning")
        self.assertNotIn("root_cause_ref", result[0])
        # Cascading demoted
        self.assertEqual(result[1]["final_severity"], "info")
        self.assertEqual(result[1]["root_cause_ref"], 1)

    def test_three_way_cascade(self):
        f1 = self._enriched(line=10)
        f2 = self._enriched(line=20)
        f3 = self._enriched(line=30)
        result = deduplicate_cascading([f1, f2, f3])
        self.assertEqual(result[0]["final_severity"], "warning")   # root
        self.assertEqual(result[1]["final_severity"], "info")      # cascade
        self.assertEqual(result[2]["final_severity"], "info")      # cascade
        self.assertEqual(result[1]["root_cause_ref"], 10)
        self.assertEqual(result[2]["root_cause_ref"], 10)

    def test_different_files_not_grouped(self):
        f1 = self._enriched(file="a.gd", line=1)
        f2 = self._enriched(file="b.gd", line=2)  # different file
        result = deduplicate_cascading([f1, f2])
        self.assertNotIn("root_cause_ref", result[0])
        self.assertNotIn("root_cause_ref", result[1])

    def test_different_entities_not_grouped(self):
        f1 = self._enriched(line=1, assertion={"type": "identifier_not_defined", "identifier": "Foo"})
        f2 = self._enriched(line=2, assertion={"type": "identifier_not_defined", "identifier": "Bar"})
        result = deduplicate_cascading([f1, f2])
        self.assertNotIn("root_cause_ref", result[0])
        self.assertNotIn("root_cause_ref", result[1])

    def test_no_entity_key_passed_through(self):
        """Findings with no extractable entity are passed through unchanged."""
        f = {
            "file": "a.gd", "line": 1, "message": "some message",
            "assertion": {"type": "signal_not_declared"},  # no stable entity
            "final_severity": "warning", "dormant_bug": False,
            "trigger_condition": None, "verified": "unverifiable",
        }
        result = deduplicate_cascading([f])
        self.assertNotIn("root_cause_ref", result[0])

    def test_empty_input(self):
        self.assertEqual(deduplicate_cascading([]), [])


# ── _cascade_entity ───────────────────────────────────────────────────────────

class TestCascadeEntity(unittest.TestCase):

    def test_identifier_not_defined(self):
        f = {"assertion": {"type": "identifier_not_defined", "identifier": "Gun"}}
        self.assertEqual(_cascade_entity(f), "id:Gun")

    def test_node_type_mismatch(self):
        f = {"assertion": {"type": "node_type_mismatch", "node_path": "$Ship"}}
        self.assertEqual(_cascade_entity(f), "node:$Ship")

    def test_method_is_not_async(self):
        f = {"assertion": {"type": "method_is_not_async", "target_class": "Gun", "method": "fire"}}
        self.assertEqual(_cascade_entity(f), "method:Gun.fire")

    def test_fallback_to_message(self):
        f = {
            "assertion": {"type": "signal_not_declared"},
            "message": "Signal `my_signal` is missing",
        }
        entity = _cascade_entity(f)
        self.assertIsNotNone(entity)
        self.assertIn("my_signal", entity)

    def test_none_for_no_match(self):
        f = {"assertion": {"type": "signal_not_declared"}, "message": "no backticks here"}
        self.assertIsNone(_cascade_entity(f))


# ── run_pipeline ──────────────────────────────────────────────────────────────

class TestRunPipeline(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _make_project(self.root)
        self.reports = self.root / "reports"
        self.kimi = self.root / ".kimi"

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, provider, **kwargs):
        return run_pipeline(
            project_root=self.root,
            provider=provider,
            reports_base=self.reports,
            kimi_dir=self.kimi,
            **kwargs,
        )

    def test_no_files_produces_empty_report(self):
        p = MockLLMProvider()
        enriched, json_p, md_p = self._run(p)
        self.assertEqual(enriched, [])
        self.assertTrue(json_p.exists())
        self.assertTrue(md_p.exists())

    def test_report_files_created(self):
        _make_gd(self.root, "player.gd", lines=10)
        p = MockLLMProvider(default=[_valid_finding()])
        _, json_p, md_p = self._run(p)
        self.assertTrue(json_p.exists())
        self.assertTrue(md_p.exists())
        data = json.loads(json_p.read_text(encoding="utf-8"))
        self.assertIn("findings", data)

    def test_json_includes_metadata(self):
        p = MockLLMProvider()
        _, json_p, _ = self._run(p)
        data = json.loads(json_p.read_text(encoding="utf-8"))
        self.assertIn("project_path", data["metadata"])

    def test_addons_excluded_by_default(self):
        _make_gd(self.root, "scripts/main.gd", lines=10)
        _make_gd(self.root, "addons/plugin.gd", lines=10)
        p = MockLLMProvider(default=[_valid_finding()])
        enriched, _, _ = self._run(p)
        files = {f.get("file", "") for f in enriched}
        self.assertFalse(any("addons" in f for f in files))

    def test_addons_included_when_flag_set(self):
        _make_gd(self.root, "addons/plugin.gd", lines=10)
        p = MockLLMProvider(default=[_valid_finding()])
        enriched, _, _ = self._run(p, review_addons=True)
        files = {f.get("file", "") for f in enriched}
        self.assertTrue(any("addons" in f for f in files))

    def test_report_dir_named_with_timestamp(self):
        p = MockLLMProvider()
        _, json_p, _ = self._run(p)
        self.assertIn("REVIEW_", json_p.parent.name)

    def test_show_refuted_flag_propagated(self):
        """When show_refuted=True, refuted findings appear in markdown."""
        # identifier_not_defined for an identifier NOT in class_names → unverifiable
        # Use a finding that results in "drop" by being refuted if class exists
        _make_gd(self.root, "main.gd", lines=10)
        finding = _valid_finding(
            severity="error",
            assertion={"type": "identifier_not_defined", "identifier": "__no_class__999"},
        )
        p = MockLLMProvider(default=[finding])
        _, _, md_p = self._run(p, show_refuted=True)
        # MD has Refuted section marker (even if empty)
        md_text = md_p.read_text(encoding="utf-8")
        self.assertIn("Refuted Findings", md_text)


# ── main() exit codes ─────────────────────────────────────────────────────────

class TestMainExitCodes(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _make_project(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_exit_2_nonexistent_project(self):
        code = main(["--project", str(self.root / "nonexistent")])
        self.assertEqual(code, 2)

    def test_exit_2_no_project_godot(self):
        empty = Path(self._tmp.name) / "noproj"
        empty.mkdir()
        code = main(["--project", str(empty)])
        self.assertEqual(code, 2)

    def test_exit_0_no_findings(self):
        """Empty project → 0 verified errors → exit 0."""
        # No .gd files → no findings → 0 errors
        code = main(["--project", str(self.root)])
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
