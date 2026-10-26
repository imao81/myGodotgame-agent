#!/usr/bin/env python3
"""eval/routing_eval.py — collect the (file × arm) review matrix.

Reviews every in-scope .gd file of the mutant project with each routing arm
and appends one JSON record per (file, arm) to eval/results/routing_matrix.jsonl.
Re-running skips (file, arm) combinations already recorded (resume-safe).

Matching semantics mirror eval/mutant_eval.py exactly: findings from
review_file() (sanitized + verifier-enriched, refuted KEPT) are matched
against the per-file pseudo-golden via eval.fitness.match_findings.

Usage:
  HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 \
      .venv/Scripts/python.exe eval/routing_eval.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_AGENT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_AGENT_ROOT))

from eval.fitness import match_findings  # noqa: E402
from eval.mutant_eval import _entity_field  # noqa: E402

_MUTANT_SET = _AGENT_ROOT / "eval" / "mutant_set.jsonl"
_WORKDIR = _AGENT_ROOT / ".kimi" / "mutant_project"
_RESULTS = _AGENT_ROOT / "eval" / "results" / "routing_matrix.jsonl"
_TEMPLATE = _AGENT_ROOT / "prompts" / "review_prompt_v2.txt"
_MIN_LINES = 5  # mirrors review._MIN_LINES (stub files are skipped silently)

# Two arms under test (subset of agents.router._ARM_TABLE).
ARMS = [
    {"name": "deepseek-flash", "provider": "deepseek",
     "model": "deepseek-v4-flash", "rel_cost": 0.7},
    {"name": "kimi-standard", "provider": "kimi",
     "model": "kimi-k3", "rel_cost": 2.0},
]

# Historical recall anchors from eval/mutant_report.json runs.
_REFERENCE_RECALL = {"deepseek-flash": 0.435, "kimi-standard": 0.731}

_MAX_ATTEMPTS = 3


def load_mutant_records(path: Path = _MUTANT_SET) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def pseudo_golden_for(records: list[dict], rel: str) -> list[dict]:
    """Per-file pseudo-golden, same shape as mutant_eval.evaluate()."""
    return [
        {"id": r["mutant_id"], "file": r["file"], "line": r["line"],
         "human_label": "tp", "human_notes": r["description"],
         "message": r["description"],
         "assertion": {"type": r["expected_assertion"], **_entity_field(r)}}
        for r in records if r["file"] == rel
    ]


_SKIP_PARTS = {"libraries", "addons", ".godot", ".kimi", ".git", "__pycache__"}


def discover_review_files(workdir: Path) -> list[str]:
    """All reviewable .gd files (production scope: addons/libraries excluded).

    NOTE: review.discover_files() cannot be used here — it matches exclusion
    names against ABSOLUTE path parts, and the mutant project lives under
    `.kimi/`, which is itself an excluded name. Match on relative parts instead
    (same policy as eval/mutant_eval.py).
    """
    return sorted(
        f.relative_to(workdir).as_posix()
        for f in workdir.rglob("*.gd")
        if not _SKIP_PARTS & set(f.relative_to(workdir).parts)
    )


def _load_done(path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("error") is None:
                done.add((rec["file"], rec["arm"]))
    return done


def collect_one(rel: str, arm: dict, workdir: Path, index: dict,
                template: str, golden: list[dict]) -> dict:
    """Review one file with one arm; returns the matrix record."""
    from review import RealLLMProvider, review_file

    gd_path = workdir / rel
    code = gd_path.read_text(encoding="utf-8", errors="replace")
    n_lines = len(code.strip().splitlines())

    record: dict = {
        "file": rel, "arm": arm["name"], "n_lines": n_lines,
        "findings_raw": 0, "findings_verified": [],
        "matched_mutant_ids": [], "fp_count": 0,
        "prompt_tokens": 0, "completion_tokens": 0,
        "latency_s": 0.0, "error": None,
    }
    if n_lines < _MIN_LINES:
        return record  # stub: review_file would skip silently, no API call

    provider = RealLLMProvider(model=arm["model"], provider=arm["provider"],
                               max_tokens=4096)
    last_err: str | None = None
    for _attempt in range(_MAX_ATTEMPTS):
        t0 = time.perf_counter()
        findings, err = review_file(gd_path, workdir, index, provider,
                                    template=template)
        latency = time.perf_counter() - t0
        if err is None:
            usage = provider.last_usage or {}
            match = match_findings(findings, golden)
            matched_ids = sorted({g["id"] for _, g in match["matched"]})
            verified = [f for f in findings if f.get("verified") != "refuted"]
            fp = [f for f in match["novel"] if f.get("verified") != "refuted"]
            record.update({
                "findings_raw": len(findings),
                "findings_verified": [
                    {"line": f.get("line"),
                     "verified": f.get("verified"),
                     "assertion_type": (f.get("assertion") or {}).get("type", ""),
                     "message": (f.get("message") or "")[:160]}
                    for f in verified
                ],
                "matched_mutant_ids": matched_ids,
                "fp_count": len(fp),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "latency_s": round(latency, 2),
            })
            return record
        last_err = err
        time.sleep(2.0)
    record["error"] = last_err
    return record


def summarise(path: Path, n_mutants: int) -> dict:
    """Per-arm recall / tokens / latency from the matrix file. LLM-free."""
    records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
               if l.strip()]
    records = [r for r in records if r.get("error") is None]
    out: dict[str, dict] = {}
    for arm in ARMS:
        sub = [r for r in records if r["arm"] == arm["name"]]
        detected = len({m for r in sub for m in r["matched_mutant_ids"]})
        out[arm["name"]] = {
            "files": len(sub),
            "detected": detected,
            "n_mutants": n_mutants,
            "recall": round(detected / n_mutants, 4) if n_mutants else None,
            "total_tokens": sum(r["prompt_tokens"] + r["completion_tokens"]
                                for r in sub),
            "total_latency_s": round(sum(r["latency_s"] for r in sub), 1),
            "fp_total": sum(r["fp_count"] for r in sub),
        }
    return out


def main() -> None:
    workdir = _WORKDIR.resolve()
    records = load_mutant_records()
    files = discover_review_files(workdir)
    expected = len(files) * len(ARMS)
    print(f"[routing] {len(files)} files × {len(ARMS)} arms = {expected} reviews")

    from review import load_or_build_index
    index, cache = load_or_build_index(workdir, kimi_dir=_AGENT_ROOT / ".kimi")
    print(f"[routing] index: {len(index.get('scripts', {}))} scripts ({cache})")

    template = _TEMPLATE.read_text(encoding="utf-8")
    _RESULTS.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(_RESULTS)
    print(f"[routing] {len(done)} combinations already recorded — resuming")

    t_start = time.perf_counter()
    n_done = len(done)
    for rel in files:
        golden = pseudo_golden_for(records, rel)
        for arm in ARMS:
            if (rel, arm["name"]) in done:
                continue
            rec = collect_one(rel, arm, workdir, index, template, golden)
            with _RESULTS.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_done += 1
            tag = "ERROR" if rec["error"] else \
                f"{len(rec['matched_mutant_ids'])} hit, {rec['fp_count']} fp"
            print(f"[routing] {n_done}/{expected} {arm['name']:15s} {rel}: {tag}",
                  flush=True)

    summary = summarise(_RESULTS, len(records))
    wall = time.perf_counter() - t_start
    print(f"\n[routing] wall time this run: {wall:.0f}s")
    for name, s in summary.items():
        ref = _REFERENCE_RECALL.get(name)
        line = (f"[routing] {name:15s} recall {s['recall'] * 100:.1f}% "
                f"({s['detected']}/{s['n_mutants']}) on {s['files']} files | "
                f"tokens {s['total_tokens']:,} | {s['total_latency_s']}s | "
                f"fp {s['fp_total']}")
        if ref is not None:
            dev = (s["recall"] - ref) * 100
            line += f" | ref {ref * 100:.1f}% (dev {dev:+.1f}pp)"
        print(line)
    missing = expected - len(_load_done(_RESULTS))
    if missing:
        print(f"[routing] WARNING: {missing} combinations still missing "
              f"(API errors) — re-run to retry")


if __name__ == "__main__":
    main()
