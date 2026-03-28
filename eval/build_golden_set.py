#!/usr/bin/env python3
"""Build a stratified golden eval set from a verified_report.json.

Usage
-----
    # Use the most recent REVIEW_* report:
    python eval/build_golden_set.py

    # Point at a specific report:
    python eval/build_golden_set.py reports/REVIEW_20260327_133503/verified_report.json

Output
------
    eval/golden_set.jsonl   — 50 JSONL records ready for human labelling

Each record has:
    id              "f001" … "f050"
    file            relative GDScript path
    line            int or null
    final_severity  "error" | "warning" | "info"
    dormant_bug     bool
    check_method    "static" | "llm"   (derived from assertion.type)
    message         human-readable description
    assertion       verifier sub-object
    human_label     null  ← fill in: "tp" | "fp" | "partial"
    human_notes     ""    ← free-text annotation
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────

_AGENT_ROOT = Path(__file__).parent.parent
_EVAL_DIR = _AGENT_ROOT / "eval"

# Assertion types that are checked with static analysis (not LLM-only)
_STATIC_TYPES: frozenset[str] = frozenset({
    "method_is_not_async",
    "node_type_mismatch",
    "identifier_not_defined",
})

# Stratified sampling targets
_TARGETS = {
    "error":     8,
    "dormant":   8,
    "static":    8,
    "cascading": 5,
}
_TOTAL = 50
_MAX_PER_FILE = 3


# ── Helpers ────────────────────────────────────────────────────────────────────

def _check_method(finding: dict) -> str:
    atype = finding.get("assertion", {}).get("type", "")
    return "static" if atype in _STATIC_TYPES else "llm"


def _find_latest_report() -> Path:
    candidates = sorted(
        (_AGENT_ROOT / "reports").glob("REVIEW_*/verified_report.json")
    )
    if not candidates:
        raise FileNotFoundError(
            "No REVIEW_*/verified_report.json found in reports/. "
            "Run: python review.py --project ./godot-2d-space-game/project"
        )
    return candidates[-1]


def load_findings(report_path: Path) -> list[dict]:
    """Load findings from a verified_report.json, annotating check_method."""
    data = json.loads(report_path.read_text(encoding="utf-8"))
    findings: list[dict] = data.get("findings", [])
    for f in findings:
        f["check_method"] = _check_method(f)
    return findings


# ── Stratified sampler ─────────────────────────────────────────────────────────

def _build_strata(candidates: list[dict]) -> dict[str, list[dict]]:
    return {
        "error":     [f for f in candidates if f.get("final_severity") == "error"],
        "dormant":   [f for f in candidates if f.get("dormant_bug")],
        "static":    [f for f in candidates if f.get("check_method") == "static"],
        "cascading": [f for f in candidates if f.get("root_cause_ref") is not None],
        "warning":   [f for f in candidates if f.get("final_severity") == "warning"],
        "info":      [f for f in candidates if f.get("final_severity") == "info"],
    }


def build_golden_set(
    findings: list[dict],
    seed: int = 42,
) -> tuple[list[dict], dict[str, int], dict[str, int]]:
    """Stratified-sample up to _TOTAL findings.

    Returns
    -------
    (selected_findings, sampled_counts_per_stratum, pool_sizes_per_stratum)
    """
    rng = random.Random(seed)

    # Exclude refuted drops — they have no meaningful human label
    candidates = [f for f in findings if f.get("final_severity") != "drop"]

    selected: list[dict] = []
    selected_obj_ids: set[int] = set()
    file_counts: dict[str, int] = {}

    def _can_add(f: dict) -> bool:
        return (
            id(f) not in selected_obj_ids
            and file_counts.get(f.get("file", ""), 0) < _MAX_PER_FILE
        )

    def _add(f: dict) -> None:
        selected.append(f)
        selected_obj_ids.add(id(f))
        fname = f.get("file", "")
        file_counts[fname] = file_counts.get(fname, 0) + 1

    def _sample_stratum(pool: list[dict], n: int) -> int:
        """Sample up to n eligible items from pool. Returns count added."""
        shuffled = list(pool)
        rng.shuffle(shuffled)
        added = 0
        for f in shuffled:
            if added >= n or len(selected) >= _TOTAL:
                break
            if _can_add(f):
                _add(f)
                added += 1
        return added

    strata = _build_strata(candidates)
    pool_sizes = {name: len(pool) for name, pool in strata.items()}

    # Phase 1: fill mandatory strata in priority order
    sampled: dict[str, int] = {}
    for name, target in _TARGETS.items():
        sampled[name] = _sample_stratum(strata[name], target)

    # Phase 2: fill remainder from warnings then any remaining candidate
    remaining = _TOTAL - len(selected)
    if remaining > 0:
        filler_pool = strata["warning"] + strata["info"] + candidates
        sampled["filler"] = _sample_stratum(filler_pool, remaining)

    return selected, sampled, pool_sizes


# ── Output serialiser ──────────────────────────────────────────────────────────

def _to_jsonl_record(finding: dict, idx: int) -> dict:
    return {
        "id":             f"f{idx:03d}",
        "file":           finding.get("file", ""),
        "line":           finding.get("line"),
        "final_severity": finding.get("final_severity", ""),
        "dormant_bug":    bool(finding.get("dormant_bug")),
        "check_method":   finding.get("check_method", "llm"),
        "message":        finding.get("message", ""),
        "assertion":      finding.get("assertion", {}),
        "human_label":    None,
        "human_notes":    "",
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    if args:
        report_path = Path(args[0])
    else:
        report_path = _find_latest_report()

    print(f"Report : {report_path}")

    findings = load_findings(report_path)
    non_drop = [f for f in findings if f.get("final_severity") != "drop"]
    print(f"Findings (excl. drops): {len(non_drop)}")

    if not non_drop:
        print(
            "\nERROR: No findings to sample. "
            "Run the pipeline first:\n"
            "  python review.py --project ../godot-2d-space-game/project"
        )
        sys.exit(1)

    selected, sampled, pool_sizes = build_golden_set(findings)

    # Write JSONL
    _EVAL_DIR.mkdir(exist_ok=True)
    out_path = _EVAL_DIR / "golden_set.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for i, f in enumerate(selected, 1):
            fh.write(json.dumps(_to_jsonl_record(f, i)) + "\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nWrote {len(selected)} samples → {out_path}")
    print("\n┌─────────────┬────────┬──────────┬────────────┐")
    print("│ Stratum     │ Target │ Sampled  │ Pool size  │")
    print("├─────────────┼────────┼──────────┼────────────┤")
    for name, target in _TARGETS.items():
        got = sampled.get(name, 0)
        pool = pool_sizes.get(name, 0)
        flag = " OK" if got >= target else f" !! (only {got})"
        print(f"│ {name:<11} │ {target:>6} │ {got:>6}{flag:<4} │ {pool:>10} │")
    filler = sampled.get("filler", 0)
    print(f"│ {'filler':<11} │ {'—':>6} │ {filler:>8} │ {'—':>10} │")
    print("├─────────────┼────────┼──────────┼────────────┤")
    print(f"│ {'TOTAL':<11} │ {_TOTAL:>6} │ {len(selected):>8} │ {'—':>10} │")
    print("└─────────────┴────────┴──────────┴────────────┘")

    # File distribution check
    file_counts: dict[str, int] = {}
    for f in selected:
        fname = f.get("file", "")
        file_counts[fname] = file_counts.get(fname, 0) + 1
    over_cap = {fname: n for fname, n in file_counts.items() if n > _MAX_PER_FILE}
    if over_cap:
        print(f"\nWARNING: Files exceeding {_MAX_PER_FILE}-per-file cap: {over_cap}")
    else:
        print(f"\nFile cap ({_MAX_PER_FILE}/file): OK — "
              f"{len(file_counts)} unique files represented")

    print(
        "\nNext step: open eval/golden_set.jsonl and fill in "
        "'human_label' (\"tp\"|\"fp\"|\"partial\") and 'human_notes' for each entry."
    )
    print("Then run:  python eval/run_eval.py")


if __name__ == "__main__":
    main()
