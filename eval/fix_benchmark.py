"""Fix benchmark: the mutation set, run in reverse.

Each injected bug is a fix task with a built-in ground truth: the mutant
record's expected_assertion must no longer reproduce after the fix (checked
by the deterministic assertion verifier inside the sandbox gates).

    .venv/Scripts/python.exe eval/fix_benchmark.py --arm flash --limit 4   # smoke
    .venv/Scripts/python.exe eval/fix_benchmark.py --arm k3                # full

Metrics: fix_rate, mean rounds, boundary-violation rate, token cost.
Breakpoint resume: completed mutant_ids in the output JSONL are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from dotenv import load_dotenv
load_dotenv(_AGENT_ROOT / ".env")

_EVAL_DIR = Path(__file__).resolve().parent
_MUTANT_SET = _EVAL_DIR / "mutant_set.jsonl"
_MUTANT_PROJECT = _AGENT_ROOT / ".kimi" / "mutant_project"
_OUT = _EVAL_DIR / "results" / "fix_benchmark.jsonl"

_ENTITY_FIELD = {
    "signal_not_declared": "signal_name",
    "runtime_signal_not_emitted": "signal_name",
    "identifier_not_defined": "identifier",
    "file_not_exists": "path",
}


def load_mutants(path: Path = _MUTANT_SET) -> list[dict]:
    return [json.loads(l) for l in
            path.read_text(encoding="utf-8").splitlines() if l.strip()]


def finding_for_mutant(record: dict) -> dict:
    """Synthesize the finding a perfect reviewer would have filed."""
    assertion = {"type": record["expected_assertion"]}
    key = _ENTITY_FIELD.get(record["expected_assertion"])
    if key:
        assertion[key] = record["expected_entity"]
    assertion["file"] = "res://" + record["file"]
    return {
        "file": record["file"],
        "line": record["line"],
        "rule": record["operator"],
        "message": record["description"],
        "assertion": assertion,
    }


def evaluate_outcome(outcome: dict) -> bool:
    """A fix counts only if the bug no longer reproduces inside clean gates."""
    gates = outcome.get("gate_report") or {}
    return (
        outcome.get("status") == "fixed"
        and gates.get("ok") is True
        # None means the gate never ran — unverified, must not count
        and gates.get("assertions_resolved") is True
        and (outcome.get("boundary") or {}).get("boundary_ok") is True
    )


def run_benchmark(mutants: list[dict], project_root: Path, client,
                  fix_fn, out_path: Path = _OUT) -> dict:
    """fix_fn injectable for tests. Breakpoint resume via out_path."""
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["mutant_id"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fh:
        for record in mutants:
            if record["mutant_id"] in done:
                continue
            finding = finding_for_mutant(record)
            t0 = time.perf_counter()
            outcome = fix_fn(record["file"], [finding], project_root,
                             client=client)
            row = {
                "mutant_id": record["mutant_id"],
                "operator": record["operator"],
                "status": outcome["status"],
                "fixed": evaluate_outcome(outcome),
                "rounds_used": outcome.get("rounds_used"),
                "boundary_mode": (outcome.get("boundary") or {}).get("mode"),
                "usage": outcome.get("usage"),
                "elapsed_s": round(time.perf_counter() - t0, 1),
                "error": outcome.get("error"),
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{row['mutant_id']}] {row['operator']}: "
                  f"{'FIXED' if row['fixed'] else row['status']} "
                  f"({row['rounds_used']} rounds, {row['elapsed_s']}s)")

    return summarize(out_path)


def summarize(out_path: Path = _OUT) -> dict:
    rows = [json.loads(l) for l in
            out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        return {"n": 0}
    fixed = [r for r in rows if r["fixed"]]
    prompt = sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in rows)
    completion = sum((r.get("usage") or {}).get("completion_tokens", 0)
                     for r in rows)
    return {
        "n": len(rows),
        "fix_rate": round(len(fixed) / len(rows), 4),
        "mean_rounds": round(sum(r["rounds_used"] or 0 for r in rows)
                             / len(rows), 2),
        "whole_file_fallbacks": sum(
            1 for r in rows if r.get("boundary_mode") == "whole_file"),
        "total_tokens": prompt + completion,
        "by_operator": {
            op: {
                "n": len([r for r in rows if r["operator"] == op]),
                "fixed": len([r for r in rows
                              if r["operator"] == op and r["fixed"]]),
            }
            for op in sorted({r["operator"] for r in rows})
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="flash", choices=["flash", "k3"])
    parser.add_argument("--limit", type=int, default=0, help="first N mutants only")
    parser.add_argument("--out", default=str(_OUT))
    args = parser.parse_args()

    from agents.fix_agent import fix_file
    from agents.llm_client import LLMClient
    client = (LLMClient(provider="deepseek", model="deepseek-v4-flash",
                        enable_fallback=True)
              if args.arm == "flash"
              else LLMClient(provider="kimi", model="kimi-k3",
                             enable_fallback=True))

    mutants = load_mutants()
    if args.limit:
        mutants = mutants[: args.limit]
    if not _MUTANT_PROJECT.is_dir():
        print(f"mutant project missing: {_MUTANT_PROJECT}", file=sys.stderr)
        sys.exit(2)

    summary = run_benchmark(mutants, _MUTANT_PROJECT, client, fix_file,
                            Path(args.out))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
