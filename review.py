#!/usr/bin/env python3
"""review.py — Propose-Verify GDScript code review pipeline.

Pipeline stages
---------------
  1. build_index     — scan project, write project_index.json to disk (mtime-cached)
  2. resolve_scope   — discover .gd files to review
  3. code_review_loop — per file: LLM review → sanitize → verify inline (keeps refuted)
  4. cross_correlator — detect refuted-assertion patterns → index_gap meta_findings
  5. filter_refuted  — separate drops from main findings
  6. findings_dedup  — cascading dedup on non-refuted findings
  7. report_merger   — write verified_report.json + verified_report.md

Exit codes
----------
  0   No verified errors found.
  1   One or more verified errors found.
  2   Pipeline failure (project not found, index build failed, etc.).

Usage
-----
  python review.py --project /path/to/godot/project
  python review.py --project ./project --target ships/enemies --show-refuted
  python review.py --project ./project --rebuild-index --review-addons

Options
-------
  --project PATH        (required) Godot project root (contains project.godot)
  --target PATH         Relative sub-path to scan (default: .)
  --rebuild-index       Force WP1 index rebuild (ignore cache)
  --show-refuted        Include refuted findings in Markdown (always in JSON)
  --review-addons       Also review addons/ and libraries/ source (default: excluded)
                        NOTE: addons are always indexed for verification regardless.
  --anchor FINDINGS_JSON
                        A/B test mode: inject static analysis findings from a JSON file
                        as per-file hints in the LLM prompt.  JSON must be a list of
                        {file, line, message} objects with project-relative posix paths.
                        Compare results against a default (--no-anchor) run to measure
                        whether anchoring helps or hurts finding quality.
  --no-anchor           (Default) Independent review with anti-anchoring instruction.
  --apply-config REPORT_JSON
                        Read config suggestions from a previous verified_report.json
                        and apply approved patches (non-graph operation, no LLM calls).
"""

from __future__ import annotations

import abc
import argparse
import json
import logging
import re
import signal
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ── Bootstrap path so sibling modules are importable regardless of CWD ────────
_AGENT_ROOT = Path(__file__).parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

# ── Auto-load .env if present ─────────────────────────────────────────────────
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(_AGENT_ROOT / ".env")
except ImportError:
    pass

from tools.assertion_verifier import verify_assertion
from tools.index_builder import build_project_index, write_index
from tools.parser import sanitize_findings
from tools.report_generator import write_reports

_log = logging.getLogger("review")

# ── Directories always excluded from code review ──────────────────────────────
_ALWAYS_EXCLUDE: frozenset[str] = frozenset({".godot", ".kimi", ".git", ".venv", "__pycache__"})
# Additional dirs excluded unless --review-addons is set
_ADDON_DIRS: frozenset[str] = frozenset({"addons", "libraries"})

# Minimum source lines to bother reviewing (stub files add noise)
_MIN_LINES = 5

# Max tokens for LLM review response
_MAX_TOKENS = 1536

# ── LLMProvider interface ──────────────────────────────────────────────────────

class LLMProvider(abc.ABC):
    """Dependency-injectable LLM interface for testability."""

    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        """Send prompt, return raw text response."""


class RealLLMProvider(LLMProvider):
    """Production adapter wrapping agents.llm_client.LLMClient."""

    def __init__(self, model: str | None = None, provider: str | None = None) -> None:
        from agents.llm_client import LLMClient
        self._client = LLMClient(provider=provider, model=model)

    @property
    def last_usage(self) -> dict | None:
        """Token usage of the most recent generate() call (None after failure)."""
        return self._client.last_usage

    def generate(self, prompt: str) -> str:
        return self._client.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=_MAX_TOKENS,
        ).strip()


class MockLLMProvider(LLMProvider):
    """Test double.  Returns pre-canned JSON keyed by substring in the prompt."""

    def __init__(self, responses: dict[str, list[dict]] | None = None,
                 default: list[dict] | None = None) -> None:
        self._responses = responses or {}
        self._default = default or []

    def generate(self, prompt: str) -> str:
        for key, findings in self._responses.items():
            if key in prompt:
                return json.dumps(findings)
        return json.dumps(self._default)


# ── Stage 1: Index loading with mtime-based caching ──────────────────────────

def _max_source_mtime(project_root: Path) -> float:
    mt = 0.0
    for ext in ("*.gd", "*.tscn"):
        for f in project_root.rglob(ext):
            try:
                mt = max(mt, f.stat().st_mtime)
            except OSError:
                pass
    return mt


def load_or_build_index(
    project_root: Path,
    force_rebuild: bool = False,
    kimi_dir: Path | None = None,
) -> tuple[dict, Path]:
    """Load cached WP1 index or rebuild it. Returns (index_dict, cache_path).

    Cache location: <kimi_dir>/project_index.json (default: <agent_root>/.kimi/).
    Cache is valid when its stored source_mtime_snapshot >= newest .gd/.tscn mtime.
    The returned path is the on-disk canonical location — downstream stages may
    reload the index from disk rather than carrying the full dict in state.
    """
    if kimi_dir is None:
        kimi_dir = _AGENT_ROOT / ".kimi"
    cache_path = kimi_dir / "project_index.json"
    src_mtime = _max_source_mtime(project_root)

    if not force_rebuild and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            # Accept cache if the stored snapshot covers the current source mtime
            if cached.get("source_mtime_snapshot", 0.0) >= src_mtime:
                _log.info("[build_index] loaded from cache (%d scripts)", len(cached.get("scripts", {})))
                return cached, cache_path
        except (OSError, json.JSONDecodeError):
            pass

    _log.info("[build_index] building fresh index for %s …", project_root)
    index = build_project_index(project_root)
    index["source_mtime_snapshot"] = src_mtime   # store for next-run cache check
    write_index(index, cache_path)
    _log.info(
        "[build_index] built in %.2fs — %d scripts, %d scenes, %d classes → %s",
        index.get("build_seconds", 0),
        len(index.get("scripts", {})),
        len(index.get("scenes", {})),
        len(index.get("class_names", {})),
        cache_path,
    )
    return index, cache_path


# ── Stage 2: File discovery ───────────────────────────────────────────────────

def discover_files(
    project_root: Path,
    target: str = ".",
    review_addons: bool = False,
) -> list[Path]:
    """Return sorted list of .gd files to review.

    Excludes: .godot/, .kimi/, .git/, .venv/ always.
    Excludes: addons/, libraries/ unless review_addons=True.
    """
    target_path = (project_root / target).resolve()
    if not target_path.is_dir():
        raise FileNotFoundError(f"target directory not found: {target_path}")

    excluded = _ALWAYS_EXCLUDE | (frozenset() if review_addons else _ADDON_DIRS)

    def _ok(path: Path) -> bool:
        return not any(part in excluded for part in path.parts)

    return sorted(f for f in target_path.rglob("*.gd") if _ok(f))


# ── Stage 3a: Context builder from WP1 index ─────────────────────────────────

def _build_file_context(res_path: str, code: str, index: dict) -> str:
    """Build a GDScript comment block describing cross-file context (from WP1 index)."""
    lines = [f"# === Project context for: {res_path} ==="]

    # Find the scene attached to this script
    for scene_res, scene_data in index.get("scenes", {}).items():
        if scene_data.get("script_path") == res_path:
            lines.append(f"# Attached to scene: {scene_res.split('/')[-1]}")
            nodes = scene_data.get("nodes", {})
            if nodes:
                lines.append("# Scene node tree ($NodeName: NodeType):")
                for node_key, node_data in nodes.items():
                    ntype = node_data.get("type") or "(inherited)"
                    lines.append(f"#   {node_key}: {ntype}")
            break

    # class_names referenced in this file
    all_classes = index.get("class_names", {})
    referenced = sorted(
        cls for cls in all_classes
        if cls in code and all_classes[cls] != res_path
    )
    if referenced:
        lines.append("# Available class_names (defined elsewhere, no preload needed):")
        for cls in referenced:
            src = all_classes[cls].replace("res://", "")
            lines.append(f"#   {cls}  →  {src}")

    # Autoloads referenced in this file (autoloads is now a dict)
    autoloads: dict = index.get("autoloads", {})
    used_autoloads = [name for name in autoloads if name in code]
    if used_autoloads:
        lines.append(f"# Autoload singletons used: {', '.join(used_autoloads)}")

    # Signals declared in this file
    script_entry = index.get("scripts", {}).get(res_path, {})
    signals = script_entry.get("signals", [])
    if signals:
        lines.append(f"# Signals declared here: {', '.join(signals)}")

    if len(lines) <= 1:
        return ""
    lines.append("# === END project context ===")
    return "\n".join(lines)


# ── Stage 3b: Prompt building ─────────────────────────────────────────────────

_PROMPT_TEMPLATE: str | None = None
_GDSCRIPT_REFERENCE: str | None = None
_GODOT3_WARNED: bool = False

# Severity matrix — mirrors tools/report_generator.py for per-file inline enrichment
_SEVERITY_MATRIX: dict[tuple[str, str], tuple[str, bool]] = {
    ("error",   "confirmed"):    ("error",   False),
    ("error",   "refuted"):      ("drop",    False),
    ("error",   "unverifiable"): ("warning", True),
    ("warning", "confirmed"):    ("warning", False),
    ("warning", "refuted"):      ("warning", False),
    ("warning", "unverifiable"): ("warning", False),
    ("info",    "confirmed"):    ("info",    False),
    ("info",    "refuted"):      ("info",    False),
    ("info",    "unverifiable"): ("info",    False),
}

# Minimum number of files sharing the same refuted assertion to emit an index_gap meta_finding
_REFUTED_PATTERN_MIN_FILES = 3


def _load_prompt_template() -> str:
    global _PROMPT_TEMPLATE
    if _PROMPT_TEMPLATE is None:
        tpl_path = _AGENT_ROOT / "prompts" / "review_prompt_v2.txt"
        if not tpl_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {tpl_path}")
        _PROMPT_TEMPLATE = tpl_path.read_text(encoding="utf-8")
    return _PROMPT_TEMPLATE


def _load_gdscript_reference() -> str:
    global _GDSCRIPT_REFERENCE
    if _GDSCRIPT_REFERENCE is None:
        ref_path = _AGENT_ROOT / "prompts" / "gdscript4_reference.md"
        if ref_path.exists():
            _GDSCRIPT_REFERENCE = ref_path.read_text(encoding="utf-8")
        else:
            _GDSCRIPT_REFERENCE = ""
    return _GDSCRIPT_REFERENCE


_ANTI_ANCHOR_INSTRUCTION = (
    "You are reviewing this code independently. You have NO prior information about "
    "which lines may have issues. Do not assume any particular line is more likely to "
    "have bugs than another. Analyze the entire file with equal attention.\n\n"
)


def build_prompt(
    rel_posix: str,
    code: str,
    index: dict,
    anchor_findings: list[dict] | None = None,
    memory_hints: list[dict] | None = None,
) -> str:
    """Build the formatted LLM review prompt for one file.

    Parameters
    ----------
    anchor_findings:
        When None (default): the anti-anchoring instruction is injected so the LLM
        reviews the file without any bias toward pre-flagged lines.
        When a list (may be empty): the anti-anchoring instruction is omitted and the
        findings are injected as static-analysis hints for A/B comparison.
    memory_hints:
        Verified false-positive patterns from ReviewMemory (past runs).  Injected
        as their own block, orthogonal to anchor mode.
    """
    template = _load_prompt_template()
    res_path = f"res://{rel_posix}"

    autoloads: dict = index.get("autoloads", {})
    autoload_list = ", ".join(autoloads) if autoloads else "(none)"

    context_block = _build_file_context(res_path, code, index)
    if context_block:
        context_and_code = f"{context_block}\n=== CODE TO REVIEW BELOW ===\n{code}"
    else:
        context_and_code = code

    # Prepend GDScript 4.x language reference with enforcement instruction
    global _GODOT3_WARNED
    godot_version: str = index.get("godot_version", "4.x")
    is_godot3 = godot_version.startswith("3")

    if is_godot3:
        if not _GODOT3_WARNED:
            _log.warning(
                "[review] Godot 3.x detected (version: %s) — "
                "GDScript 4 reference not applicable, LLM review may have reduced accuracy",
                godot_version,
            )
            _GODOT3_WARNED = True
        reference = ""
    else:
        reference = _load_gdscript_reference().replace("{godot_version}", godot_version)

    version_note = f"This project uses Godot {godot_version}.\n\n"
    if reference:
        preamble = (
            version_note
            + reference
            + "\n\n> The GDScript 4.x reference above describes how the language "
            "ACTUALLY works at runtime. If your analysis contradicts any statement "
            "in the reference, your analysis is WRONG — revise it before outputting.\n\n"
            "==============================================================================\n\n"
        )
    else:
        preamble = version_note

    # Anti-anchoring vs. anchor mode
    if anchor_findings is None:
        # Default: independent review, no hints about pre-flagged lines
        bias_block = _ANTI_ANCHOR_INSTRUCTION
    elif anchor_findings:
        lines = "\n".join(
            f"  - line {f.get('line', '?')}: {f.get('message', '(no message)')}"
            for f in anchor_findings
        )
        bias_block = (
            "The following static analysis issues were found in this file:\n"
            + lines
            + "\nYou may use these as hints but do not assume they are correct.\n\n"
        )
    else:
        bias_block = "No static analysis issues were pre-detected for this file.\n\n"

    # Experience memory: verifier-backed false-positive patterns from past runs
    if memory_hints:
        mem_lines = "\n".join(
            f"  - [{m.get('assertion_type', '')}] {m.get('summary', '')} "
            f"(verifier: {m.get('evidence', {}).get('verifier_status', '')}; "
            f"seen {m.get('hits', 1)}x)"
            for m in memory_hints
        )
        memory_block = (
            "KNOWN FALSE-POSITIVE PATTERNS (refuted by the static verifier in past runs):\n"
            + mem_lines
            + "\nThese were REFUTED before — do not report them again unless the "
              "current code clearly differs. Judge the code independently.\n\n"
        )
    else:
        memory_block = ""

    return (
        preamble
        + bias_block
        + memory_block
        + template
        .replace("{filename}", rel_posix)
        .replace("{autoload_list}", autoload_list)
        .replace("{context_and_code}", context_and_code)
    )


# ── Stage 3c: LLM call + JSON extraction ─────────────────────────────────────

def _parse_json_findings(raw: str) -> list[dict]:
    """Extract the JSON array from raw LLM text output."""
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return []
    try:
        result = json.loads(m.group())
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        return []


def _enrich_one_finding(finding: dict, index: dict) -> dict:
    """Verify one sanitized finding and apply the severity matrix inline.

    Mirrors tools/report_generator.enrich_findings but operates on a single
    finding so verification happens inside the per-file review loop.

    Refuted findings (final_severity="drop") are KEPT in the returned dict —
    the cross_correlator needs them for refuted-pattern detection.
    """
    assertion: dict = finding.get("assertion", {})
    llm_severity: str = finding.get("severity", "info")
    try:
        vr = verify_assertion(assertion, index)
    except Exception as exc:  # noqa: BLE001
        _log.debug("[review] verifier raised for %r: %s", assertion.get("type"), exc)
        vr = {"status": "unverifiable", "confidence": 0.0,
              "details": f"verifier error: {exc}", "fix_suggestion": None}

    vr_status: str = vr["status"]
    final_severity, dormant_bug = _SEVERITY_MATRIX.get(
        (llm_severity, vr_status), ("info", False)
    )
    return {
        **finding,
        "verified":         vr_status,
        "final_severity":   final_severity,
        "dormant_bug":      dormant_bug,
        "trigger_condition": finding.get("message") if dormant_bug else None,
        "_verifier": {
            "confidence":    vr.get("confidence"),
            "details":       vr.get("details"),
            "fix_suggestion": vr.get("fix_suggestion"),
        },
    }


def review_file(
    gd_path: Path,
    project_root: Path,
    index: dict,
    provider: LLMProvider,
    anchor_findings: list[dict] | None = None,
    memory_hints: list[dict] | None = None,
) -> tuple[list[dict], str | None]:
    """Review one file.  Returns (sanitized_findings, error_message_or_None).

    anchor_findings: None = anti-anchor mode (default); list = anchor hint mode.
    memory_hints: verified false-positive patterns from past runs (ReviewMemory).
    """
    rel_posix = gd_path.relative_to(project_root).as_posix()
    try:
        code = gd_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return [], f"cannot read file: {e}"

    if len(code.strip().splitlines()) < _MIN_LINES:
        return [], None   # stub — skip silently

    try:
        prompt = build_prompt(rel_posix, code, index, anchor_findings=anchor_findings,
                              memory_hints=memory_hints)
        raw = provider.generate(prompt)
    except Exception as e:
        return [], f"LLM call failed: {e}"

    raw_findings = _parse_json_findings(raw)
    for f in raw_findings:
        f["file"] = rel_posix
        f.setdefault("check_method", "llm")

    # sanitize → verify + apply severity matrix per-file (keeps refuted as "drop")
    clean = sanitize_findings(raw_findings)
    enriched = [_enrich_one_finding(f, index) for f in clean]
    return enriched, None


# ── Stage 4: Cascading error deduplication ────────────────────────────────────

def _cascade_entity(finding: dict) -> str | None:
    """Extract a stable entity key for cascading grouping, or None."""
    a = finding.get("assertion", {})
    t = a.get("type", "")
    if t == "identifier_not_defined":
        ident = a.get("identifier", "")
        return f"id:{ident}" if ident else None
    if t == "node_type_mismatch":
        node = a.get("node_path", "")
        return f"node:{node}" if node else None
    if t == "method_is_not_async":
        cls = a.get("target_class", "")
        meth = a.get("method", "")
        return f"method:{cls}.{meth}" if cls and meth else None
    # Fallback: first backtick/quote identifier in message
    msg = finding.get("message", "")
    m = re.search(r"[`'\"]([\w.]+)[`'\"]", msg)
    return f"msg:{m.group(1)}" if m else None


def deduplicate_cascading(enriched: list[dict]) -> list[dict]:
    """Demote subsequent findings that share (file, assertion_type, entity).

    Root (first per group): unchanged.
    Cascading (rest): final_severity → "info", root_cause_ref → root line.
    Only non-dropped groups of 2+ are affected.
    """
    # First pass: count how many times each group key appears
    group_counts: dict[tuple, int] = defaultdict(int)
    group_root_line: dict[tuple, int | None] = {}

    for f in enriched:
        entity = _cascade_entity(f)
        if entity is None:
            continue
        key = (f.get("file", ""), entity)
        group_counts[key] += 1
        if key not in group_root_line:
            group_root_line[key] = f.get("line")

    # Second pass: demote
    seen_keys: set[tuple] = set()
    result: list[dict] = []

    for f in enriched:
        entity = _cascade_entity(f)
        if entity is None:
            result.append(f)
            continue

        key = (f.get("file", ""), entity)

        # Only demote in groups of 2+
        if group_counts[key] < 2:
            result.append(f)
            continue

        if key not in seen_keys:
            seen_keys.add(key)
            result.append(f)           # root — keep as-is
        else:
            # Cascading: demote to info, reference the root
            result.append({
                **f,
                "final_severity": "info",
                "root_cause_ref": group_root_line[key],
            })

    return result


# ── Stage 4: Cross_correlator — refuted pattern detection ─────────────────────

def _detect_refuted_patterns(enriched: list[dict]) -> list[dict]:
    """Detect systematic refuted-assertion patterns across files.

    Groups refuted findings (final_severity="drop") by (assertion.type, entity).
    When ≥ _REFUTED_PATTERN_MIN_FILES files share the same refuted assertion,
    emits a meta_finding with category "index_gap" — a signal that the project
    index may be missing a class/method, or that context injection is failing.

    Returns a list of meta_findings (may be empty).  These are included in the
    report as info-level findings so they are visible without --show-refuted.
    """
    refuted = [f for f in enriched if f.get("final_severity") == "drop"]
    if not refuted:
        return []

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for f in refuted:
        a = f.get("assertion", {})
        atype = a.get("type", "")
        if atype == "identifier_not_defined":
            entity = a.get("identifier", "")
        elif atype == "method_is_not_async":
            entity = f"{a.get('target_class', '')}.{a.get('method', '')}"
        elif atype == "node_type_mismatch":
            entity = a.get("node_path", "")
        else:
            entity = ""
        if entity:
            groups[(atype, entity)].append(f)

    meta: list[dict] = []
    for (atype, entity), findings in groups.items():
        files = sorted({f.get("file", "") for f in findings if f.get("file")})
        if len(files) < _REFUTED_PATTERN_MIN_FILES:
            continue
        extra = f" (+{len(files) - 5} more)" if len(files) > 5 else ""
        meta.append({
            "file":             "",
            "line":             None,
            "severity":         "info",
            "category":         "index_gap",
            "message": (
                f"LLM flagged '{entity}' ({atype}) in {len(files)} files, "
                f"all refuted by verifier. Possible project_index gap or context "
                f"injection issue. Affected: {', '.join(files[:5])}{extra}"
            ),
            "assertion":        {"type": "unverifiable"},
            "verified":         "unverifiable",
            "final_severity":   "info",
            "dormant_bug":      False,
            "trigger_condition": None,
            "_verifier":        {"confidence": 0.0, "details": "meta finding from cross_correlator",
                                 "fix_suggestion": "Re-index the project (--rebuild-index) or check "
                                                   "that the relevant class/method is in the index."},
            "_is_meta":         True,
            "_affected_files":  files,
            "check_method":     "meta",
        })

    return meta


# ── Main pipeline orchestrator ────────────────────────────────────────────────

def run_pipeline(
    project_root: Path,
    target: str = ".",
    rebuild_index: bool = False,
    review_addons: bool = False,
    show_refuted: bool = False,
    provider: LLMProvider | None = None,
    reports_base: Path | None = None,
    kimi_dir: Path | None = None,
    anchor_findings_path: Path | None = None,
    use_memory: bool = True,
) -> tuple[list[dict], Path, Path]:
    """Run the full WP1→WP4 pipeline.

    Parameters
    ----------
    anchor_findings_path:
        Path to a JSON file containing static analysis findings (list of dicts with
        at least "file" and "message" keys, project-relative posix paths).  When
        provided each file's findings are injected into the LLM prompt as hints.
        When None (default) the anti-anchoring instruction is used instead.
    use_memory:
        When True (default), retrieve verifier-backed false-positive patterns from
        ReviewMemory before each file review and curate new outcomes afterwards.
        Pass False (or --no-memory) for an A/B baseline run.

    Returns
    -------
    (all_enriched_findings, json_report_path, md_report_path)
    """
    if provider is None:
        provider = RealLLMProvider()
    if reports_base is None:
        reports_base = _AGENT_ROOT / "reports"
    if kimi_dir is None:
        kimi_dir = _AGENT_ROOT / ".kimi"

    memory = None
    if use_memory:
        from agents.review_memory import ReviewMemory
        memory = ReviewMemory(kimi_dir / "memory")

    # ── Load anchor findings (grouped by file) ────────────────────────────────
    # anchor_by_file: None → anti-anchor mode; {} or populated → anchor mode
    anchor_by_file: dict[str, list[dict]] | None = None
    if anchor_findings_path is not None:
        try:
            raw_anchor = json.loads(anchor_findings_path.read_text(encoding="utf-8"))
            anchor_by_file = {}
            for entry in raw_anchor:
                key = entry.get("file", "")
                anchor_by_file.setdefault(key, []).append(entry)
            _log.info(
                "[anchor] loaded %d static findings from %s covering %d file(s)",
                len(raw_anchor), anchor_findings_path, len(anchor_by_file),
            )
        except Exception as e:
            _log.warning("[anchor] could not load anchor findings from %s: %s — "
                         "falling back to anti-anchor mode", anchor_findings_path, e)
            anchor_by_file = None

    # ── Stage 1: build_index (writes project_index.json to disk) ─────────────
    index, index_path = load_or_build_index(project_root, force_rebuild=rebuild_index, kimi_dir=kimi_dir)
    _log.info("[build_index] project_index.json at %s", index_path)

    # ── Stage 2: resolve_scope (file discovery) ───────────────────────────────
    gd_files = discover_files(project_root, target=target, review_addons=review_addons)
    _log.info("[resolve_scope] %d .gd files in scope", len(gd_files))

    # ── Stage 3: code_review_loop — per-file LLM review + sanitize + verify ──
    # review_file() returns enriched findings (with final_severity set).
    # Refuted findings (final_severity="drop") are KEPT for the cross_correlator.
    all_enriched: list[dict] = []
    errors: list[str] = []
    interrupted = False
    i = 0

    def _handle_interrupt(sig, frame):
        nonlocal interrupted
        _log.warning("[review] interrupted — saving partial results …")
        interrupted = True

    old_handler = signal.signal(signal.SIGINT, _handle_interrupt)

    try:
        for i, gd in enumerate(gd_files, 1):
            if interrupted:
                break
            rel = gd.relative_to(project_root).as_posix()
            _log.info("[%d/%d] reviewing %s", i, len(gd_files), rel)

            file_anchor = anchor_by_file.get(rel, []) if anchor_by_file is not None else None
            mem_hints = memory.retrieve(rel, index) if memory is not None else None
            enriched_file, err = review_file(gd, project_root, index, provider,
                                             anchor_findings=file_anchor,
                                             memory_hints=mem_hints)
            if err:
                _log.warning("[review] %s — %s", rel, err)
                errors.append(f"{rel}: {err}")
            all_enriched.extend(enriched_file)

    finally:
        signal.signal(signal.SIGINT, old_handler)

    # ── Stage 4: cross_correlator — detect refuted patterns → meta_findings ──
    meta_findings = _detect_refuted_patterns(all_enriched)
    if meta_findings:
        _log.info("[cross_correlator] %d index_gap meta finding(s) detected", len(meta_findings))

    # ── Stage 5: filter_refuted — separate drops from main findings ───────────
    refuted_drops = [f for f in all_enriched if f.get("final_severity") == "drop"]
    non_refuted   = [f for f in all_enriched if f.get("final_severity") != "drop"]

    # ── Memory write: curate verifier-backed outcomes into ReviewMemory ───────
    memory_stats = None
    if memory is not None:
        ts_run = datetime.now().strftime("%Y%m%d_%H%M%S")
        memory_stats = memory.curate(all_enriched, run_id=ts_run)
        _log.info("[memory] curated: %d added, %d updated, %d skipped (total %d)",
                  memory_stats["added"], memory_stats["updated"],
                  memory_stats["skipped"], memory_stats["total"])

    # ── Stage 6: findings_dedup — cascading dedup on non-refuted only ─────────
    non_refuted_deduped = deduplicate_cascading(non_refuted)

    # ── Stage 7: report_merger — combine all findings for the report ──────────
    # deduped non-refuted first (main content), then drops (for --show-refuted /
    # eval JSON), then meta_findings (info-level index-gap diagnostics).
    report_findings = non_refuted_deduped + refuted_drops + meta_findings

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_partial" if interrupted else ""
    out_dir = reports_base / f"REVIEW_{ts}{suffix}"

    meta = {
        "project_path":        str(project_root),
        "target":              target,
        "godot_version":       index.get("godot_version", "unknown"),
        "index_path":          str(index_path),
        "timestamp":           datetime.now(timezone.utc).isoformat(),
        "files_reviewed":      i if not interrupted else i - 1,
        "files_total":         len(gd_files),
        "pipeline_errors":     errors,
        "interrupted":         interrupted,
        "meta_findings_count": len(meta_findings),
    }
    if memory_stats is not None:
        meta["memory"] = memory_stats

    json_p, md_p = write_reports(
        report_findings, out_dir,
        show_refuted=show_refuted,
        metadata=meta,
    )

    counts = _summarise(report_findings)
    _log.info(
        "[done] %d findings — %d errors, %d dormant, %d warnings, %d dropped, %d meta | %s",
        len(report_findings),
        counts["verified_errors"], counts["dormant_bugs"],
        counts["warnings"], counts["refuted_drops"],
        len(meta_findings),
        "PARTIAL" if interrupted else "complete",
    )
    return report_findings, json_p, md_p


def _summarise(enriched: list[dict]) -> dict:
    return {
        "verified_errors": sum(1 for f in enriched if f["final_severity"] == "error"),
        "dormant_bugs":    sum(1 for f in enriched if f.get("dormant_bug")),
        "warnings":        sum(1 for f in enriched
                               if f["final_severity"] == "warning" and not f.get("dormant_bug")),
        "infos":           sum(1 for f in enriched if f["final_severity"] == "info"),
        "refuted_drops":   sum(1 for f in enriched if f["final_severity"] == "drop"),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="review.py",
        description="Propose-Verify GDScript code review (WP1→WP4).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit codes:
  0   No verified errors found.
  1   One or more verified errors found.
  2   Pipeline failure (project not found, index error, etc.).

Examples:
  python review.py --project ./project
  python review.py --project ./project --target ships/enemies --show-refuted
  python review.py --project ./project --rebuild-index --review-addons
""",
    )
    p.add_argument("--project", required=True, metavar="PATH",
                   help="Godot project root (contains project.godot)")
    p.add_argument("--target", default=".", metavar="PATH",
                   help="Relative sub-path to scan (default: .)")
    p.add_argument("--rebuild-index", action="store_true",
                   help="Force WP1 index rebuild (ignore cache)")
    p.add_argument("--show-refuted", action="store_true",
                   help="Include refuted findings in Markdown report (always in JSON)")
    p.add_argument("--review-addons", action="store_true",
                   help="Review addons/ and libraries/ source files "
                        "(addons are always indexed for verification)")
    anchor_group = p.add_mutually_exclusive_group()
    anchor_group.add_argument(
        "--anchor", metavar="FINDINGS_JSON",
        help="Anchor mode (A/B testing): path to a JSON file of static analysis findings "
             "(list of {file, line, message} objects with project-relative posix paths). "
             "Each file's findings are injected into the LLM prompt as hints. "
             "Compare output against the default --no-anchor run to measure anchoring effect.",
    )
    anchor_group.add_argument(
        "--no-anchor", action="store_true", default=True,
        help="(Default) Independent review: inject anti-anchoring instruction so the LLM "
             "analyses every line with equal attention, without prior hints.",
    )
    p.add_argument(
        "--no-memory", action="store_true",
        help="Disable ReviewMemory (no retrieval injection, no curation). "
             "Use for A/B baseline runs measuring the memory effect.",
    )
    p.add_argument(
        "--apply-config", metavar="REPORT_JSON",
        help="Non-graph operation: read config_suggestions from a previous "
             "verified_report.json and apply approved patches without re-running the "
             "full pipeline. No LLM calls are made.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s  %(message)s",
        stream=sys.stderr,
    )

    parser = _build_argparser()
    args = parser.parse_args(argv)

    # ── --apply-config: standalone config-patch operation (no pipeline) ─────────
    if args.apply_config:
        report_path = Path(args.apply_config)
        if not report_path.is_file():
            print(f"ERROR: report file not found: {report_path}", file=sys.stderr)
            return 2
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            suggestions = report.get("findings", [])
            config_sug = [f for f in suggestions if f.get("category") == "config_suggestion"]
            if not config_sug:
                print("No config suggestions found in report.", file=sys.stderr)
                return 0
            print(f"Found {len(config_sug)} config suggestion(s) in {report_path}",
                  file=sys.stderr)
            print("(--apply-config patch application not yet implemented; "
                  "review suggestions in the report and apply manually.)", file=sys.stderr)
            return 0
        except Exception as exc:
            print(f"ERROR reading report: {exc}", file=sys.stderr)
            return 2

    project_root = Path(args.project).resolve()
    if not project_root.is_dir():
        print(f"ERROR: project path not found: {project_root}", file=sys.stderr)
        return 2
    if not (project_root / "project.godot").exists():
        print(f"ERROR: no project.godot in {project_root}", file=sys.stderr)
        return 2

    # Ensure reports/ exists and is in .gitignore
    reports_base = _AGENT_ROOT / "reports"
    reports_base.mkdir(exist_ok=True)
    gitignore = _AGENT_ROOT / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("reports/\n.kimi/\n", encoding="utf-8")
    else:
        content = gitignore.read_text(encoding="utf-8")
        lines_needed = []
        if "reports/" not in content:
            lines_needed.append("reports/")
        if ".kimi/" not in content:
            lines_needed.append(".kimi/")
        if lines_needed:
            gitignore.write_text(
                content.rstrip("\n") + "\n" + "\n".join(lines_needed) + "\n",
                encoding="utf-8",
            )

    anchor_path: Path | None = None
    if args.anchor:
        anchor_path = Path(args.anchor)
        if not anchor_path.is_file():
            print(f"ERROR: anchor findings file not found: {anchor_path}", file=sys.stderr)
            return 2
        _log.info("[anchor] A/B mode enabled — static hints from %s", anchor_path)
    else:
        _log.info("[anchor] anti-anchor mode — independent review (use --anchor to A/B test)")

    try:
        enriched, json_p, md_p = run_pipeline(
            project_root=project_root,
            target=args.target,
            rebuild_index=args.rebuild_index,
            review_addons=args.review_addons,
            show_refuted=args.show_refuted,
            anchor_findings_path=anchor_path,
            use_memory=not args.no_memory,
        )
    except Exception as exc:
        _log.exception("[review] pipeline failed: %s", exc)
        return 2

    counts = _summarise(enriched)
    print(f"\nReport: {md_p}", file=sys.stderr)
    print(f"JSON:   {json_p}", file=sys.stderr)
    print(
        f"\n{counts['verified_errors']} verified error(s) | "
        f"{counts['dormant_bugs']} dormant bug(s) | "
        f"{counts['warnings']} warning(s) | "
        f"{counts['refuted_drops']} refuted (dropped)",
        file=sys.stderr,
    )

    return 1 if counts["verified_errors"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
