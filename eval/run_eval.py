#!/usr/bin/env python3
"""Evaluate pipeline quality against a human-labelled golden set.

Usage
-----
    python eval/run_eval.py                          # reads eval/golden_set.jsonl
    python eval/run_eval.py eval/golden_set.jsonl    # explicit path

Input
-----
golden_set.jsonl where each record has:
    human_label   "tp" | "fp" | "partial"   (null = unlabelled, skipped)
    final_severity, dormant_bug, check_method

Output
------
    eval/benchmark.md   — summary table + per-dimension breakdown
    Prints the same summary to stdout.

Metric definitions
------------------
    Precision          TP / (TP + FP)      where partial counts as 0.5 TP
    Dormant accuracy   TP-rate among dormant_bug=true findings
    FP rate            FP / (TP + FP)      by check_method
    Severity calib.    fraction where human-judged severity matches final_severity
                       (requires human_notes to contain "sev:<actual>" token,
                        e.g. "sev:warning" — optional, skipped if absent)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

_EVAL_DIR = Path(__file__).parent
_BENCHMARK_PATH = _EVAL_DIR / "benchmark.md"

# Partial findings count as this fraction of a true positive
_PARTIAL_WEIGHT = 0.5


# ── Loader ─────────────────────────────────────────────────────────────────────

def load_labelled(path: Path) -> list[dict]:
    """Read JSONL, return only labelled (non-null human_label) records."""
    records: list[dict] = []
    skipped = 0
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  WARNING line {lineno}: invalid JSON — {e}")
                continue
            if rec.get("human_label") is None:
                skipped += 1
                continue
            records.append(rec)
    if skipped:
        print(f"  Skipped {skipped} unlabelled records.")
    return records


# ── Metric helpers ─────────────────────────────────────────────────────────────

def _tp_weight(label: str) -> float:
    if label == "tp":
        return 1.0
    if label == "partial":
        return _PARTIAL_WEIGHT
    return 0.0


def _is_fp(label: str) -> bool:
    return label == "fp"


def _precision(records: list[dict]) -> float:
    if not records:
        return float("nan")
    tp = sum(_tp_weight(r["human_label"]) for r in records)
    fp = sum(1 for r in records if _is_fp(r["human_label"]))
    denom = tp + fp
    return tp / denom if denom > 0 else float("nan")


def _fp_rate(records: list[dict]) -> float:
    if not records:
        return float("nan")
    fp = sum(1 for r in records if _is_fp(r["human_label"]))
    return fp / len(records)


def _parse_human_severity(notes: str) -> str | None:
    """Extract sev:<value> token from human_notes, e.g. 'sev:warning'."""
    for token in notes.split():
        if token.startswith("sev:"):
            return token[4:].lower().strip()
    return None


# ── Core evaluation ────────────────────────────────────────────────────────────

def evaluate(records: list[dict]) -> dict:
    """Compute all benchmark metrics. Returns a structured result dict."""
    n = len(records)
    if n == 0:
        return {"error": "No labelled records found."}

    # ── Overall precision ────────────────────────────────────────────────────
    overall_precision = _precision(records)
    overall_fp_rate   = _fp_rate(records)

    # ── Precision by final_severity ──────────────────────────────────────────
    by_severity: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_severity[r.get("final_severity", "unknown")].append(r)

    precision_by_severity: dict[str, float] = {
        sev: _precision(recs)
        for sev, recs in sorted(by_severity.items())
    }
    count_by_severity: dict[str, int] = {
        sev: len(recs) for sev, recs in sorted(by_severity.items())
    }

    # ── Dormant bug accuracy ─────────────────────────────────────────────────
    dormant = [r for r in records if r.get("dormant_bug")]
    dormant_accuracy = _precision(dormant) if dormant else float("nan")
    dormant_fp_rate  = _fp_rate(dormant)   if dormant else float("nan")

    # ── FP rate by check_method ──────────────────────────────────────────────
    by_method: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_method[r.get("check_method", "unknown")].append(r)

    fp_rate_by_method: dict[str, float] = {
        method: _fp_rate(recs)
        for method, recs in sorted(by_method.items())
    }
    precision_by_method: dict[str, float] = {
        method: _precision(recs)
        for method, recs in sorted(by_method.items())
    }
    count_by_method: dict[str, int] = {
        method: len(recs) for method, recs in sorted(by_method.items())
    }

    # ── Severity calibration ─────────────────────────────────────────────────
    calib_total = 0
    calib_match = 0
    for r in records:
        human_sev = _parse_human_severity(r.get("human_notes", ""))
        if human_sev is not None:
            calib_total += 1
            if human_sev == r.get("final_severity", ""):
                calib_match += 1

    severity_calibration = (calib_match / calib_total) if calib_total > 0 else None

    return {
        "n_labelled":              n,
        "overall_precision":       overall_precision,
        "overall_fp_rate":         overall_fp_rate,
        "precision_by_severity":   precision_by_severity,
        "count_by_severity":       count_by_severity,
        "fp_rate_by_severity":     {
            sev: _fp_rate(recs)
            for sev, recs in sorted(by_severity.items())
        },
        "n_dormant":               len(dormant),
        "dormant_accuracy":        dormant_accuracy,
        "dormant_fp_rate":         dormant_fp_rate,
        "fp_rate_by_method":       fp_rate_by_method,
        "precision_by_method":     precision_by_method,
        "count_by_method":         count_by_method,
        "severity_calib_total":    calib_total,
        "severity_calib_match":    calib_match,
        "severity_calibration":    severity_calibration,
    }


# ── Report renderer ────────────────────────────────────────────────────────────

def _pct(v: float | None) -> str:
    if v is None or v != v:   # NaN check
        return "N/A"
    return f"{v * 100:.1f}%"


def render_markdown(results: dict, golden_path: Path) -> str:
    lines: list[str] = []

    lines += [
        "# Pipeline Benchmark Report",
        "",
        f"**Golden set:** `{golden_path}`  ",
        f"**Labelled findings:** {results['n_labelled']}  ",
        f"> `partial` findings count as {_PARTIAL_WEIGHT} TP for precision.",
        "",
        "---",
        "",
    ]

    # ── Table 1: Overall ──────────────────────────────────────────────────────
    lines += [
        "## Overall Quality",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Precision (TP+partial / all) | **{_pct(results['overall_precision'])}** |",
        f"| False-positive rate          | {_pct(results['overall_fp_rate'])} |",
        "",
    ]

    # ── Table 2: Precision by severity ───────────────────────────────────────
    lines += [
        "## Precision by `final_severity`",
        "",
        "| Severity | Count | Precision | FP Rate |",
        "|----------|------:|----------:|--------:|",
    ]
    for sev in sorted(results["precision_by_severity"]):
        n   = results["count_by_severity"].get(sev, 0)
        p   = results["precision_by_severity"][sev]
        fpr = results["fp_rate_by_severity"].get(sev, float("nan"))
        lines.append(f"| `{sev}` | {n} | {_pct(p)} | {_pct(fpr)} |")
    lines.append("")

    # ── Table 3: Dormant bug accuracy ─────────────────────────────────────────
    lines += [
        "## Dormant Bug Accuracy",
        "",
        f"> {results['n_dormant']} findings with `dormant_bug=true` were labelled.",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Dormant precision (are they real defects?) | **{_pct(results['dormant_accuracy'])}** |",
        f"| Dormant FP rate                            | {_pct(results['dormant_fp_rate'])} |",
        "",
    ]

    # ── Table 4: FP rate by check_method ─────────────────────────────────────
    lines += [
        "## False-Positive Rate by `check_method`",
        "",
        "| Method | Count | Precision | FP Rate |",
        "|--------|------:|----------:|--------:|",
    ]
    for method in sorted(results["fp_rate_by_method"]):
        n   = results["count_by_method"].get(method, 0)
        p   = results["precision_by_method"].get(method, float("nan"))
        fpr = results["fp_rate_by_method"][method]
        lines.append(f"| `{method}` | {n} | {_pct(p)} | {_pct(fpr)} |")
    lines.append("")

    # ── Table 5: Severity calibration ─────────────────────────────────────────
    calib = results["severity_calibration"]
    calib_n = results["severity_calib_total"]
    lines += [
        "## Severity Calibration",
        "",
    ]
    if calib is None or calib_n == 0:
        lines += [
            "> No severity calibration data found.",
            "> To enable: add `sev:<value>` token to `human_notes`,",
            "> e.g. `\"human_notes\": \"sev:warning real issue but not a crash\"`.",
            "",
        ]
    else:
        lines += [
            f"> Based on {calib_n} findings with `sev:` annotation in `human_notes`.",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Severity match rate | **{_pct(calib)}** "
            f"({results['severity_calib_match']}/{calib_n}) |",
            "",
        ]

    # ── Interpretation guide ──────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## Interpretation",
        "",
        "| Range | Meaning |",
        "|-------|---------|",
        "| Precision ≥ 80% | Pipeline is production-quality |",
        "| Precision 60–79% | Acceptable for assisted review; prompt tuning advised |",
        "| Precision < 60%  | High noise; prompt or verifier work needed |",
        "| FP rate (static) < FP rate (llm) | Verifier is working correctly |",
        "| Dormant accuracy ≥ 70% | Dormant-bug escalation is well-calibrated |",
        "",
    ]

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    golden_path = Path(args[0]) if args else _EVAL_DIR / "golden_set.jsonl"

    if not golden_path.exists():
        print(f"ERROR: golden set not found: {golden_path}")
        print("Run:  python eval/build_golden_set.py")
        sys.exit(1)

    print(f"Reading: {golden_path}")
    records = load_labelled(golden_path)

    if not records:
        print(
            "No labelled records found.\n"
            "Open eval/golden_set.jsonl and set 'human_label' to "
            "\"tp\", \"fp\", or \"partial\" for each entry."
        )
        sys.exit(1)

    results = evaluate(records)
    if "error" in results:
        print(f"ERROR: {results['error']}")
        sys.exit(1)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  Labelled: {results['n_labelled']} findings")
    print(f"  Overall precision : {_pct(results['overall_precision'])}")
    print(f"  Overall FP rate   : {_pct(results['overall_fp_rate'])}")
    print(f"  Dormant accuracy  : {_pct(results['dormant_accuracy'])}"
          f"  (n={results['n_dormant']})")
    print(f"  Severity calib.   : "
          + (_pct(results['severity_calibration'])
             if results['severity_calibration'] is not None
             else "N/A (add sev: tokens to human_notes)"))
    print(f"\n  FP rate by method:")
    for method, fpr in sorted(results["fp_rate_by_method"].items()):
        n = results["count_by_method"].get(method, 0)
        print(f"    {method:8s}: {_pct(fpr)}  (n={n})")
    print(f"\n  Precision by severity:")
    for sev, p in sorted(results["precision_by_severity"].items()):
        n = results["count_by_severity"].get(sev, 0)
        print(f"    {sev:8s}: {_pct(p)}  (n={n})")
    print(f"{'='*50}")

    # ── Write benchmark.md ────────────────────────────────────────────────────
    md = render_markdown(results, golden_path)
    _BENCHMARK_PATH.write_text(md, encoding="utf-8")
    print(f"\nReport written → {_BENCHMARK_PATH}")


if __name__ == "__main__":
    main()
