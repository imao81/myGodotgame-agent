"""agents/gepa.py — offline prompt evolution (GEPA-style reflective mutation).

Evolves prompts/review_prompt_v2.txt against the human-labelled golden set.

Loop per iteration:
  1. pick the best-scoring candidate so far as parent
  2. reflect: the mutator LLM reads the parent template + textual fitness
     feedback (missed true bugs, reproduced false positives with the human
     notes explaining WHY they were wrong) and writes a full improved template
  3. gate: candidate must keep the {filename}/{autoload_list}/{context_and_code}
     placeholders — a template that can't be filled is discarded for free
  4. score on a random minibatch of golden-covered files with the cheap eval
     tier (default: deepseek-v4-flash)

Final: top candidates are re-scored on the FULL golden-covered file set;
the winner is written to --output and a report to eval/gepa_report.md.

Simplification vs. the paper (arXiv 2507.19457): GEPA tracks a Pareto front
over per-instance scores; with only 33 files we use mean fitness over
minibatches plus a full-set runoff — documented here so nobody quotes the
paper's guarantees verbatim for this implementation.

Cost control: fitness replays are LLM calls. ~12s/file means a 12-file
minibatch ≈ 2.5 min on the eval tier. --iterations 8 --minibatch 12 is a
sane first run (~20 min, eval tier only; mutator uses the stronger model).

Usage:
  python -m agents.gepa --iterations 8 --minibatch 12 --seed 42
  python -m agents.gepa --dry-run            # validate setup, zero LLM calls
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

_AGENT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_AGENT_ROOT))

REQUIRED_PLACEHOLDERS = ("{filename}", "{autoload_list}", "{context_and_code}")

_MUTATION_PROMPT = """\
You are a prompt engineer improving a GDScript (Godot 4) code-review prompt.

The prompt's output is replayed against a human-labelled golden set. Below is
the current template and its fitness feedback: true bugs it keeps missing, and
false positives it keeps producing — with the human reviewers' notes on why
those were wrong.

Improve the template so it finds the missed true bugs WITHOUT reproducing the
known false positives. Keep the output JSON schema and the assertion-type
system intact — downstream verifiers depend on them. Make targeted changes
(rules, examples, emphasis), not a rewrite.

HARD REQUIREMENT: the template MUST keep these placeholders verbatim:
{filename}  {autoload_list}  {context_and_code}

Output ONLY the complete new template. No commentary, no markdown fences.

=== CURRENT TEMPLATE ===
"""


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\n?", "", stripped)
        stripped = re.sub(r"\n?```\s*$", "", stripped)
    return stripped


def template_is_valid(template: str) -> list[str]:
    """Return list of missing placeholders ([] = valid)."""
    return [p for p in REQUIRED_PLACEHOLDERS if p not in template]


def mutate_template(base_template: str, feedback: str, provider) -> str | None:
    """One reflective mutation. Returns None when the mutant is unusable."""
    # Note: plain concatenation, NOT .replace() — the mutation prompt itself
    # mentions the placeholder tokens, so .replace("{filename}", ...) would
    # corrupt the instruction block.
    prompt = (
        _MUTATION_PROMPT
        + base_template
        + "\n=== END TEMPLATE ===\n\n=== FITNESS FEEDBACK ===\n" + feedback
        + "\n=== END FEEDBACK ===\n"
    )
    mutant = _strip_fences(provider.generate(prompt))
    if not mutant or len(mutant) < 200:
        return None
    if template_is_valid(mutant):
        return None
    return mutant


def run_evolution(
    iterations: int = 8,
    minibatch_size: int = 12,
    seed: int = 42,
    eval_model: str = "deepseek-v4-flash",
    eval_provider_name: str = "deepseek",
    mutator_model: str = "kimi-k2.7-code",
    mutator_provider_name: str = "kimi",
    output: Path | None = None,
    dry_run: bool = False,
) -> dict:
    """Run the evolution loop. Returns the final report dict."""
    from eval.fitness import (
        feedback_text, load_golden, match_findings, replay_with_prompt,
        score_match,
    )

    base_template = (_AGENT_ROOT / "prompts" / "review_prompt_v2.txt") \
        .read_text(encoding="utf-8")
    golden_path = _AGENT_ROOT / "eval" / "golden_set.jsonl"
    golden = load_golden(golden_path)
    files = sorted({g["file"] for g in golden})
    report_path = _AGENT_ROOT / "eval" / "gepa_report.md"

    print(f"[gepa] golden: {len(golden)} labelled findings over {len(files)} files")
    print(f"[gepa] eval tier: {eval_provider_name}/{eval_model} — "
          f"mutator: {mutator_provider_name}/{mutator_model}")

    if dry_run:
        missing = template_is_valid(base_template)
        print(f"[gepa] dry-run OK — base template placeholders "
              f"{'all present' if not missing else f'MISSING: {missing}'}")
        return {"dry_run": True, "files": len(files), "golden": len(golden)}

    # Providers + index (loaded once). Generous token budgets: thinking tiers
    # (DeepSeek V4) spend tokens on reasoning, and the mutator must reproduce
    # a ~18k-char template — the default 1536 review budget truncates both.
    from review import RealLLMProvider, load_or_build_index
    project_path = _AGENT_ROOT.parent / "godot-2d-space-game" / "project"
    index, _ = load_or_build_index(project_path, kimi_dir=_AGENT_ROOT / ".kimi")
    eval_provider = RealLLMProvider(model=eval_model, provider=eval_provider_name,
                                    max_tokens=8192)
    mutator_provider = RealLLMProvider(model=mutator_model,
                                       provider=mutator_provider_name,
                                       max_tokens=16384)

    rng = random.Random(seed)
    candidates: list[dict] = [{
        "id": 0, "parent": None, "template": base_template,
        "evals": [], "mean_fitness": None,
    }]

    def evaluate(cand: dict, file_subset: list[str]) -> dict:
        produced, errors = replay_with_prompt(
            cand["template"], file_subset, project_path, index, eval_provider)
        scoped_golden = [g for g in golden if g["file"] in set(file_subset)]
        match = match_findings(produced, scoped_golden)
        scored = score_match(match, scoped_golden)
        scored["n_files"] = len(file_subset)
        scored["errors"] = len(errors)
        cand["evals"].append(scored)
        cand["mean_fitness"] = round(
            sum(e["fitness"] for e in cand["evals"]) / len(cand["evals"]), 4)
        cand["_last_match"] = match  # transient, for feedback text
        return scored

    # ── Baseline: score the un-evolved prompt on the full set ───────────────
    print("[gepa] baseline evaluation (full set)…")
    t0 = time.perf_counter()
    base_scored = evaluate(candidates[0], files)
    print(f"[gepa] baseline fitness {base_scored['fitness']:+.4f} "
          f"(recall {base_scored['recall_true']:.2f}, "
          f"fp_repro {base_scored['fp_repro_rate']:.2f}, "
          f"novel {base_scored['novel_count']}) [{time.perf_counter()-t0:.0f}s]")

    # ── Evolution loop ───────────────────────────────────────────────────────
    for it in range(1, iterations + 1):
        parent = max(candidates,
                     key=lambda c: c["mean_fitness"]
                     if c["mean_fitness"] is not None else float("-inf"))
        fb = feedback_text(parent["evals"][-1], parent["_last_match"])
        print(f"[gepa] iter {it}/{iterations}: mutating candidate "
              f"{parent['id']} (fitness {parent['mean_fitness']:+.4f})…")

        mutant = mutate_template(parent["template"], fb, mutator_provider)
        if mutant is None:
            print("[gepa]   mutant invalid (placeholders lost) — discarded")
            continue

        cand = {"id": len(candidates), "parent": parent["id"],
                "template": mutant, "evals": [], "mean_fitness": None}
        minibatch = rng.sample(files, min(minibatch_size, len(files)))
        scored = evaluate(cand, minibatch)
        print(f"[gepa]   candidate {cand['id']}: fitness {scored['fitness']:+.4f} "
              f"on {len(minibatch)}-file minibatch")
        candidates.append(cand)

    # ── Runoff: full-set re-score of the top candidates ──────────────────────
    ranked = sorted((c for c in candidates if c["mean_fitness"] is not None),
                    key=lambda c: c["mean_fitness"], reverse=True)
    finalists = ranked[:3] if len(ranked) > 1 else ranked
    print(f"[gepa] runoff: re-scoring top {len(finalists)} on all {len(files)} files…")
    for cand in finalists:
        if cand["id"] != 0:  # baseline already has a full-set eval
            scored = evaluate(cand, files)
            print(f"[gepa]   candidate {cand['id']}: full-set fitness "
                  f"{scored['fitness']:+.4f}")

    winner = max(finalists,
                 key=lambda c: c["evals"][-1]["fitness"]
                 if c["id"] != 0 else c["evals"][0]["fitness"])

    out_path = output or (_AGENT_ROOT / "prompts" / "review_prompt_evolved.txt")
    if winner["id"] != 0:
        out_path.write_text(winner["template"], encoding="utf-8")
    print(f"[gepa] winner: candidate {winner['id']} "
          f"(parent chain: {_lineage(candidates, winner['id'])})")

    report = {
        "winner_id": winner["id"],
        "baseline_fitness": candidates[0]["evals"][0],
        "winner_fitness": winner["evals"][-1],
        "candidates": [
            {"id": c["id"], "parent": c["parent"],
             "mean_fitness": c["mean_fitness"], "n_evals": len(c["evals"])}
            for c in candidates
        ],
        "output": str(out_path) if winner["id"] != 0 else "(kept base prompt)",
    }
    _write_report(report_path, report)
    print(f"[gepa] report → {report_path}")
    return report


def _lineage(candidates: list[dict], cid: int) -> str:
    by_id = {c["id"]: c for c in candidates}
    chain = [cid]
    while by_id[cid]["parent"] is not None:
        cid = by_id[cid]["parent"]
        chain.append(cid)
    return " ← ".join(map(str, chain))


def _write_report(path: Path, report: dict) -> None:
    lines = [
        "# GEPA Prompt Evolution Report", "",
        f"Winner: candidate {report['winner_id']}",
        f"Evolved prompt: `{report['output']}`", "",
        "## Fitness (full golden set)", "",
        "| Candidate | Fitness | Recall | FP repro | Novel |",
        "|-----------|--------:|-------:|---------:|------:|",
    ]
    base = report["baseline_fitness"]
    lines.append(f"| 0 (baseline) | {base['fitness']:+.4f} | "
                 f"{base['recall_true']:.2f} | {base['fp_repro_rate']:.2f} | "
                 f"{base['novel_count']} |")
    win = report["winner_fitness"]
    lines.append(f"| {report['winner_id']} (winner) | {win['fitness']:+.4f} | "
                 f"{win['recall_true']:.2f} | {win['fp_repro_rate']:.2f} | "
                 f"{win['novel_count']} |")
    lines += ["", "## All candidates", "",
              "| ID | Parent | Mean fitness | Evals |",
              "|----|--------|-------------:|------:|"]
    for c in report["candidates"]:
        mf = f"{c['mean_fitness']:+.4f}" if c["mean_fitness"] is not None else "—"
        parent = str(c["parent"]) if c["parent"] is not None else "—"
        lines.append(f"| {c['id']} | {parent} | {mf} | {c['n_evals']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GEPA prompt evolution")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--minibatch", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-model", default="deepseek-v4-flash")
    parser.add_argument("--eval-provider", default="deepseek")
    parser.add_argument("--mutator-model", default="kimi-k2.7-code")
    parser.add_argument("--mutator-provider", default="kimi")
    parser.add_argument("--output", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_evolution(
        iterations=args.iterations,
        minibatch_size=args.minibatch,
        seed=args.seed,
        eval_model=args.eval_model,
        eval_provider_name=args.eval_provider,
        mutator_model=args.mutator_model,
        mutator_provider_name=args.mutator_provider,
        output=Path(args.output) if args.output else None,
        dry_run=args.dry_run,
    )
