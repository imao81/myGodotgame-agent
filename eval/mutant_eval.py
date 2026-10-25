#!/usr/bin/env python3
"""eval/mutant_eval.py — build a mutated project copy and measure recall.

Flow:
  1. --build:  copy the game project, inject one mutant per file across as
     many files as possible, write eval/mutant_set.jsonl (ground truth)
  2. (default) replay review on every mutant file in the mutated copy and
     score detection: recall = detected / injected
  3. --report: eval/mutant_report.md + .json

Mutants live in a COPY of the project — the game repo is never touched.

Usage:
  python eval/mutant_eval.py                       # build + evaluate (flash)
  python eval/mutant_eval.py --build-only
  python eval/mutant_eval.py --model kimi-k3 --provider kimi
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

_AGENT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_AGENT_ROOT))

from eval.mutators import MUTATORS, apply_mutator  # noqa: E402

_MUTANT_SET = _AGENT_ROOT / "eval" / "mutant_set.jsonl"
_WORKDIR = _AGENT_ROOT / ".kimi" / "mutant_project"

_SKIP_PARTS = {"libraries", "addons", ".godot", ".kimi"}


def build_mutant_project(project_path: Path, workdir: Path,
                         seed: int = 42,
                         kimi_dir: Path | None = None) -> list[dict]:
    """Copy the project and inject one mutant per file. Returns records."""
    import random

    if workdir.exists():
        shutil.rmtree(workdir)
    shutil.copytree(
        project_path, workdir,
        ignore=shutil.ignore_patterns(".godot", ".git", "*.import", ".kimi"),
    )

    known_classes = set()
    try:
        from review import load_or_build_index
        index, _ = load_or_build_index(project_path,
                                       kimi_dir=kimi_dir or (_AGENT_ROOT / ".kimi"))
        known_classes = set(index.get("class_names", {}))
    except Exception:
        pass

    gd_files = sorted(
        f for f in workdir.rglob("*.gd")
        if not _SKIP_PARTS & set(f.relative_to(workdir).parts)
        and f.name != "test_controller.gd"
    )

    rng = random.Random(seed)
    operators = list(MUTATORS)
    rng.shuffle(gd_files)

    records: list[dict] = []
    for gd in gd_files:
        rel = gd.relative_to(workdir).as_posix()
        text = gd.read_text(encoding="utf-8", errors="replace")
        # shuffle operator priority per file so common patterns (Events.emit
        # sites are everywhere) don't starve the rarer operators
        ops = list(operators)
        rng.shuffle(ops)
        for op in ops:
            mid = f"m{len(records) + 1:03d}"
            result = apply_mutator(op, text, rel, mid, known_classes)
            if result is None:
                continue
            new_text, record = result
            gd.write_text(new_text, encoding="utf-8")
            records.append(record.to_dict())
            break  # one mutant per file

    _MUTANT_SET.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8")
    print(f"[mutant] injected {len(records)} mutants across "
          f"{len({r['file'] for r in records})} files → {workdir}")
    by_op: dict[str, int] = {}
    for r in records:
        by_op[r["operator"]] = by_op.get(r["operator"], 0) + 1
    print(f"[mutant] operators: {by_op}")
    return records


def evaluate(records: list[dict], workdir: Path, model: str,
             provider_name: str) -> dict:
    """Replay review on mutant files; recall = detected / injected."""
    from eval.fitness import match_findings
    from review import RealLLMProvider, load_or_build_index, review_file

    template = (_AGENT_ROOT / "prompts" / "review_prompt_v2.txt") \
        .read_text(encoding="utf-8")
    index, _ = load_or_build_index(workdir, kimi_dir=_AGENT_ROOT / ".kimi")
    provider = RealLLMProvider(model=model, provider=provider_name)

    # Mutants are true bugs by construction → pseudo-golden with label "tp"
    pseudo_golden = [
        {"id": r["mutant_id"], "file": r["file"], "line": r["line"],
         "human_label": "tp", "human_notes": r["description"],
         "message": r["description"],
         "assertion": {"type": r["expected_assertion"],
                       **_entity_field(r)}}
        for r in records
    ]

    produced: list[dict] = []
    errors: list[str] = []
    files = sorted({r["file"] for r in records})
    for i, rel in enumerate(files, 1):
        findings, err = review_file(workdir / rel, workdir, index, provider,
                                    template=template)
        if err:
            errors.append(f"{rel}: {err}")
        produced.extend(findings)
        print(f"[mutant] {i}/{len(files)} {rel}: {len(findings)} finding(s)")

    match = match_findings(produced, pseudo_golden)
    detected_ids = {g["id"] for _, g in match["matched"]}
    missed = [r for r in records if r["mutant_id"] not in detected_ids]

    by_op: dict[str, dict] = {}
    for r in records:
        d = by_op.setdefault(r["operator"], {"total": 0, "detected": 0})
        d["total"] += 1
        d["detected"] += int(r["mutant_id"] in detected_ids)

    recall = len(detected_ids) / len(records) if records else float("nan")
    return {
        "model": model, "provider": provider_name,
        "n_mutants": len(records), "detected": len(detected_ids),
        "recall": round(recall, 4), "errors": errors,
        "missed": [{k: m[k] for k in ("mutant_id", "file", "operator",
                                      "description")} for m in missed],
        "by_operator": by_op,
    }


def _entity_field(record: dict) -> dict:
    """Map expected_assertion → the assertion field carrying the entity."""
    field = {
        "signal_not_declared": "signal_name",
        "runtime_signal_not_emitted": "signal_name",
        "identifier_not_defined": "identifier",
        "file_not_exists": "path",
    }.get(record["expected_assertion"])
    return {field: record["expected_entity"]} if field else {}


def _write_report(result: dict) -> None:
    lines = [
        "# Mutant Recall Report", "",
        f"Model: `{result['provider']}/{result['model']}`", "",
        f"**Recall: {result['recall'] * 100:.1f}% "
        f"({result['detected']}/{result['n_mutants']} injected bugs detected)**",
        "", "| Operator | Detected | Total |", "|----------|---------:|-----:|",
    ]
    for op, d in sorted(result["by_operator"].items()):
        lines.append(f"| {op} | {d['detected']} | {d['total']} |")
    if result["missed"]:
        lines += ["", "## Missed mutants", ""]
        for m in result["missed"]:
            lines.append(f"- `{m['mutant_id']}` {m['file']} "
                         f"[{m['operator']}] — {m['description']}")
    md = "\n".join(lines) + "\n"
    (_AGENT_ROOT / "eval" / "mutant_report.md").write_text(md, encoding="utf-8")
    (_AGENT_ROOT / "eval" / "mutant_report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mutation-based recall eval")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_path = (_AGENT_ROOT.parent / "godot-2d-space-game" / "project").resolve()

    records = build_mutant_project(project_path, _WORKDIR, seed=args.seed)
    if args.build_only:
        return

    t0 = time.perf_counter()
    result = evaluate(records, _WORKDIR, args.model, args.provider)
    result["wall_seconds"] = round(time.perf_counter() - t0, 1)
    print(f"[mutant] recall {result['recall'] * 100:.1f}% "
          f"({result['detected']}/{result['n_mutants']}) "
          f"[{result['wall_seconds']}s]")
    _write_report(result)
    print("[mutant] report → eval/mutant_report.md")


if __name__ == "__main__":
    main()
