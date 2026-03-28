"""tools/report_generator.py — WP4: Verification pipeline + report generation.

Sits between the WP3 parser output and the final artefacts.

Pipeline
--------
  sanitized findings (WP3)
       │
       ▼  enrich_findings()
  enriched findings  ← verified, final_severity, dormant_bug, trigger_condition
       │
       ├──▶ generate_json_report()   → dict  (all findings, drops included)
       └──▶ generate_markdown_report() → str  (human-readable, drops hidden by default)

Schema delta vs WP3 post_processor
-----------------------------------
  verified        str   "confirmed" | "refuted" | "unverifiable"  (WP2 raw status)
  final_severity  str   "error" | "warning" | "info" | "drop"
  dormant_bug     bool
  trigger_condition str | None

Severity matrix (same logic as WP3, different encoding)
--------------------------------------------------------
  LLM severity | verified      | final_severity | dormant_bug
  -------------|---------------|----------------|------------
  error        | confirmed     | error          | False
  error        | refuted       | drop           | False
  error        | unverifiable  | warning        | True
  warning      | any           | warning        | False
  info         | any           | info           | False

Public API
----------
    enrich_findings(findings, index) -> list[dict]
    generate_json_report(enriched)   -> dict
    generate_markdown_report(enriched, show_refuted=False, metadata=None) -> str
    write_reports(enriched, output_dir, show_refuted=False, metadata=None) -> tuple[Path, Path]
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from tools.assertion_verifier import verify_assertion

_log = logging.getLogger(__name__)

# ── Severity matrix ────────────────────────────────────────────────────────────

_MATRIX: dict[tuple[str, str], tuple[str, bool]] = {
    # (llm_severity, vr_status) -> (final_severity, dormant_bug)
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


def _apply_matrix(llm_severity: str, vr_status: str) -> tuple[str, bool]:
    """Return (final_severity, dormant_bug) from the matrix.

    Falls back to info/False for unexpected input so the pipeline never crashes.
    """
    key = (llm_severity, vr_status)
    return _MATRIX.get(key, ("info", False))


# ── Core enrichment ────────────────────────────────────────────────────────────

def enrich_findings(findings: list[dict], index: dict) -> list[dict]:
    """Verify every finding's assertion and add WP4 schema fields.

    Unlike WP3's apply_verification, refuted errors are NOT dropped —
    they remain in the output list with final_severity="drop" so the
    JSON report can include them for eval / prompt-tuning purposes.

    Parameters
    ----------
    findings:
        Sanitized findings from tools.parser.sanitize_findings.
    index:
        Project index from tools.index_builder.build_project_index.

    Returns
    -------
    list[dict] — all input findings, each annotated with:
        verified, final_severity, dormant_bug, trigger_condition
    """
    results: list[dict] = []

    for finding in findings:
        assertion: dict = finding.get("assertion", {})
        llm_severity: str = finding.get("severity", "info")

        try:
            vr = verify_assertion(assertion, index)
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "[report_generator] verifier raised for %r: %s",
                assertion.get("type"), exc,
            )
            vr = {
                "status": "unverifiable",
                "confidence": 0.0,
                "details": f"verifier error: {exc}",
                "fix_suggestion": None,
            }

        vr_status: str = vr["status"]
        final_severity, dormant_bug = _apply_matrix(llm_severity, vr_status)

        enriched = {
            **finding,
            "verified": vr_status,           # raw WP2 string, not bool
            "final_severity": final_severity,
            "dormant_bug": dormant_bug,
            "trigger_condition": finding.get("message") if dormant_bug else None,
            # Carry verifier detail for debugging
            "_verifier": {
                "confidence": vr.get("confidence"),
                "details": vr.get("details"),
                "fix_suggestion": vr.get("fix_suggestion"),
            },
        }
        results.append(enriched)

    return results


# ── JSON report ────────────────────────────────────────────────────────────────

def generate_json_report(enriched: list[dict], metadata: dict | None = None) -> dict:
    """Build the full JSON report dict.

    Includes ALL findings — drops are present with final_severity="drop".
    This is intentional: the JSON is used for eval and prompt tuning.
    """
    counts = _count_sections(enriched)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
        "summary": {
            "total": len(enriched),
            **counts,
        },
        "findings": enriched,
    }


# ── Markdown report ────────────────────────────────────────────────────────────

def _count_sections(enriched: list[dict]) -> dict[str, int]:
    errors   = sum(1 for f in enriched if f["final_severity"] == "error")
    dormant  = sum(1 for f in enriched if f["dormant_bug"])
    warnings = sum(1 for f in enriched
                   if f["final_severity"] == "warning" and not f["dormant_bug"])
    infos    = sum(1 for f in enriched if f["final_severity"] == "info")
    drops    = sum(1 for f in enriched if f["final_severity"] == "drop")
    return {
        "verified_errors": errors,
        "dormant_bugs": dormant,
        "warnings": warnings,
        "infos": infos,
        "refuted_drops": drops,
    }


def _fmt_finding(finding: dict, label: str) -> str:
    """Format one finding as a markdown block."""
    file_  = finding.get("file", "")
    line   = finding.get("line")
    cat    = finding.get("category", finding.get("rule", ""))
    msg    = finding.get("message", "")
    fix    = finding.get("_verifier", {}).get("fix_suggestion")
    loc    = f"{file_}:{line}" if file_ and line is not None else (file_ or f"line {line}" if line else "")

    lines = [f"### `[{label}]` {loc} — {cat}" if loc else f"### `[{label}]` {cat}"]
    lines.append(msg)

    tc = finding.get("trigger_condition")
    if tc:
        lines.append(f"\n> **Trigger condition:** {tc}")

    if fix:
        lines.append(f"\n> **Fix:** {fix}")

    return "\n".join(lines)


def generate_markdown_report(
    enriched: list[dict],
    show_refuted: bool = False,
    metadata: dict | None = None,
) -> str:
    """Render a human-readable markdown report.

    Parameters
    ----------
    enriched:
        Output of enrich_findings().
    show_refuted:
        When True, append a "Refuted Findings" section at the bottom.
        Intended for system developers debugging the LLM, not end users.
    metadata:
        Optional dict of run context (project path, godot_version, etc.)
        written into the report header.
    """
    meta = metadata or {}
    project = meta.get("project_path", "")
    godot_v = meta.get("godot_version", "")
    now     = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    counts = _count_sections(enriched)
    errors   = [f for f in enriched if f["final_severity"] == "error"]
    dormant  = [f for f in enriched if f["dormant_bug"]]
    warnings = [f for f in enriched if f["final_severity"] == "warning" and not f["dormant_bug"]]
    infos    = [f for f in enriched if f["final_severity"] == "info"]
    drops    = [f for f in enriched if f["final_severity"] == "drop"]

    parts: list[str] = []

    # ── Header ──────────────────────────────────────────────────────────────
    header_lines = ["# Code Review Report — Propose-Verify Pipeline"]
    if project:
        header_lines.append(f"**Project:** {project}")
    if godot_v:
        header_lines.append(f"**Godot:** {godot_v}")
    header_lines.append(f"**Generated:** {now}")
    header_lines.append("")

    total = len(enriched)
    header_lines.append(
        f"**Summary:** {total} findings — "
        f"{counts['verified_errors']} verified errors, "
        f"{counts['dormant_bugs']} dormant bugs, "
        f"{counts['warnings']} warnings, "
        f"{counts['infos']} info, "
        f"{counts['refuted_drops']} refuted (dropped)"
    )
    parts.append("\n".join(header_lines))

    # ── Verified Errors ──────────────────────────────────────────────────────
    parts.append(f"\n---\n\n## ✗ Verified Errors ({counts['verified_errors']})")
    if errors:
        parts.append(
            "\nThese errors were **confirmed** by static analysis of the project index."
        )
        for f in errors:
            parts.append("\n" + _fmt_finding(f, "ERROR"))
    else:
        parts.append("\n_No verified errors._")

    # ── Dormant Bugs ─────────────────────────────────────────────────────────
    parts.append(f"\n---\n\n## ⚑ Dormant Bugs ({counts['dormant_bugs']})")
    if dormant:
        parts.append(
            "\nPotential errors that **could not be verified statically**. "
            "They may only trigger under specific runtime conditions."
        )
        for f in dormant:
            parts.append("\n" + _fmt_finding(f, "DORMANT"))
    else:
        parts.append("\n_No dormant bugs detected._")

    # ── Warnings ─────────────────────────────────────────────────────────────
    parts.append(f"\n---\n\n## ⚠ Warnings ({counts['warnings']})")
    if warnings:
        for f in warnings:
            parts.append("\n" + _fmt_finding(f, "WARNING"))
    else:
        parts.append("\n_No warnings._")

    # ── Info ─────────────────────────────────────────────────────────────────
    if infos:
        parts.append(f"\n---\n\n## ℹ Info ({counts['infos']})")
        for f in infos:
            parts.append("\n" + _fmt_finding(f, "INFO"))

    # ── Refuted (optional) ───────────────────────────────────────────────────
    if show_refuted:
        parts.append(
            f"\n---\n\n## ↩ Refuted Findings ({counts['refuted_drops']})"
            "\n\n> **Developer section** — these findings were contradicted by "
            "static analysis and excluded from the main report. "
            "Use this section to tune the LLM prompt."
        )
        if drops:
            for f in drops:
                vdet = f.get("_verifier", {}).get("details", "")
                parts.append("\n" + _fmt_finding(f, "REFUTED"))
                if vdet:
                    parts.append(f"\n> _Verifier: {vdet}_")
        else:
            parts.append("\n_No refuted findings._")

    return "\n".join(parts) + "\n"


# ── File writer ────────────────────────────────────────────────────────────────

def write_reports(
    enriched: list[dict],
    output_dir: Path,
    show_refuted: bool = False,
    metadata: dict | None = None,
) -> tuple[Path, Path]:
    """Write verified_report.json and verified_report.md into output_dir.

    Returns
    -------
    (json_path, md_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "verified_report.json"
    md_path   = output_dir / "verified_report.md"

    json_data = generate_json_report(enriched, metadata=metadata)
    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")

    md_text = generate_markdown_report(
        enriched, show_refuted=show_refuted, metadata=metadata
    )
    md_path.write_text(md_text, encoding="utf-8")

    _log.info(
        "[report_generator] wrote %d findings → %s, %s",
        len(enriched), json_path, md_path,
    )
    return json_path, md_path
