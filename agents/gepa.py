"""agents/gepa.py — offline prompt evolution, faithful GEPA skeleton.

Implements the core of Algorithm 1/2 from "GEPA: Reflective Prompt Evolution
Can Outperform Reinforcement Learning" (arXiv 2507.19457), adapted to
code review:

  - Dataset split: golden-covered files → D_pareto (fixed validation subset,
    instance-level score matrix S) and D_feedback (minibatches for reflection)
  - Reflective mutation with TRACES: the mutator sees the parent candidate's
    own findings annotated with verifier verdicts and human golden notes —
    not aggregate metrics (trace_feedback in eval/fitness.py)
  - Acceptance gate: a mutant enters the pool ONLY if its minibatch mean
    beats the parent's on the same files — rejected mutants never spend a
    full validation pass
  - Pareto-based parent selection (Algorithm 2): per-instance winners, union,
    remove dominated, sample proportional to win count. Greedy best-mean
    selection is explicitly avoided (paper: it collapses to local optima)
  - Per-(candidate, file) replay caching: parents are re-scored on a
    minibatch without paying for files they've already seen
  - Final selection: best average on D_pareto (paper's return rule)

Domain mapping: task instance = one file; metric μ = verifier + human golden
labels (verifiable reward); trajectory = per-file findings with verifier
verdicts.

Usage:
  python -m agents.gepa --iterations 6 --minibatch 8 --pareto-size 12
  python -m agents.gepa --dry-run           # validate setup, zero LLM calls
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
the current template and its execution traces: what it found per file, how the
static verifier judged each finding, which findings match human-confirmed
bugs, which reproduce human-rejected false positives (with the reviewers'
notes on WHY they were wrong), and which confirmed bugs it missed entirely.

Improve the template so it finds the missed true bugs WITHOUT reproducing the
known false positives or unverifiable noise. Keep the output JSON schema and
the assertion-type system intact — downstream verifiers depend on them.
Make targeted changes (rules, examples, emphasis), not a rewrite.

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
        + "\n=== END TEMPLATE ===\n\n=== EXECUTION TRACES ===\n" + feedback
        + "\n=== END TRACES ===\n"
    )
    mutant = _strip_fences(provider.generate(prompt))
    if not mutant or len(mutant) < 200:
        return None
    if template_is_valid(mutant):
        return None
    return mutant


# ── Pareto-based candidate selection (paper Algorithm 2) ──────────────────────

def pareto_sample(candidates: list[dict], pareto_files: list[str],
                  rng: random.Random) -> dict:
    """Sample a parent from the per-instance Pareto frontier.

    1. For each instance (file), winners = candidates with the max score.
    2. Frontier = union of all winners; a candidate wins ties.
    3. Remove dominated candidates (win-set is a strict subset of another's).
    4. Sample proportional to win count — candidates that uniquely own
       instances keep a proportionally high selection probability.
    """
    win_sets: dict[int, set[str]] = {c["id"]: set() for c in candidates}
    for f in pareto_files:
        best = max(c["scores"].get(f, float("-inf")) for c in candidates)
        for c in candidates:
            if c["scores"].get(f, float("-inf")) == best:
                win_sets[c["id"]].add(f)

    frontier = {cid: ws for cid, ws in win_sets.items() if ws}
    # Domination: win-set strictly covered by another candidate's
    dominated = {
        cid for cid, ws in frontier.items()
        if any(oid != cid and ws < ows for oid, ows in frontier.items())
    }
    frontier = {cid: ws for cid, ws in frontier.items() if cid not in dominated}

    if not frontier:  # no scores yet — shouldn't happen after baseline
        return rng.choice(candidates)
    pool = [c for c in candidates if c["id"] in frontier]
    weights = [len(frontier[c["id"]]) for c in pool]
    return rng.choices(pool, weights=weights, k=1)[0]


# ── Evolution loop ─────────────────────────────────────────────────────────────

def run_evolution(
    iterations: int = 6,
    minibatch_size: int = 8,
    pareto_size: int = 12,
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
        load_golden, replay_with_prompt, score_files, trace_feedback,
    )

    base_template = (_AGENT_ROOT / "prompts" / "review_prompt_v2.txt") \
        .read_text(encoding="utf-8")
    golden = load_golden(_AGENT_ROOT / "eval" / "golden_set.jsonl")
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
    shuffled = list(files)
    rng.shuffle(shuffled)
    pareto_files = sorted(shuffled[:min(pareto_size, len(shuffled))])
    feedback_files = sorted(shuffled[min(pareto_size, len(shuffled)):]) \
        or pareto_files  # tiny sets: fall back to pareto as feedback source
    print(f"[gepa] D_pareto: {len(pareto_files)} files — "
          f"D_feedback: {len(feedback_files)} files")

    # Candidate: {id, parent, template, scores: {file: float},
    #             findings: {file: [finding]}}  — scores/findings cached per file
    candidates: list[dict] = [{
        "id": 0, "parent": None, "template": base_template,
        "scores": {}, "findings": {},
    }]
    rejected: list[dict] = []

    def eval_on_files(cand: dict, file_subset: list[str]) -> dict[str, float]:
        """Per-instance scores, replaying only files not already cached."""
        todo = [f for f in file_subset if f not in cand["scores"]]
        if todo:
            produced, errors = replay_with_prompt(
                cand["template"], todo, project_path, index, eval_provider)
            for f in todo:
                cand["findings"][f] = [p for p in produced if p.get("file") == f]
            if errors:
                print(f"[gepa]   ({len(errors)} replay error(s))")
            scored = score_files(produced,
                                 [g for g in golden if g["file"] in set(todo)])
            cand["scores"].update(scored)
        return {f: cand["scores"][f] for f in file_subset}

    def mean_on(cand: dict, file_subset: list[str]) -> float:
        vals = [cand["scores"][f] for f in file_subset if f in cand["scores"]]
        return sum(vals) / len(vals) if vals else float("-inf")

    # ── Baseline on D_pareto ──────────────────────────────────────────────────
    print("[gepa] baseline evaluation on D_pareto…")
    t0 = time.perf_counter()
    eval_on_files(candidates[0], pareto_files)
    print(f"[gepa] baseline mean on D_pareto: "
          f"{mean_on(candidates[0], pareto_files):+.4f} "
          f"[{time.perf_counter() - t0:.0f}s]")

    # ── Budget loop (paper Algorithm 1) ───────────────────────────────────────
    for it in range(1, iterations + 1):
        parent = pareto_sample(candidates, pareto_files, rng)
        mb = rng.sample(feedback_files, min(minibatch_size, len(feedback_files)))

        # Gather parent scores + traces on the minibatch (mostly cache hits)
        eval_on_files(parent, mb)
        parent_mean = mean_on(parent, mb)
        parent_findings = [f for fpath in mb
                           for f in parent["findings"].get(fpath, [])]
        fb = trace_feedback(parent_findings,
                            [g for g in golden if g["file"] in set(mb)])
        print(f"[gepa] iter {it}/{iterations}: parent=candidate {parent['id']} "
              f"(minibatch mean {parent_mean:+.4f})")

        mutant = mutate_template(parent["template"], fb, mutator_provider)
        if mutant is None:
            print("[gepa]   mutant invalid (placeholders lost) — discarded")
            continue

        cand = {"id": len(candidates) + len(rejected), "parent": parent["id"],
                "template": mutant, "scores": {}, "findings": {}}
        eval_on_files(cand, mb)
        mutant_mean = mean_on(cand, mb)

        # Acceptance gate: minibatch must improve, else discard cheaply
        if mutant_mean <= parent_mean:
            print(f"[gepa]   candidate {cand['id']}: {mutant_mean:+.4f} "
                  f"≤ parent {parent_mean:+.4f} — REJECTED at gate")
            rejected.append({"id": cand["id"], "parent": parent["id"],
                             "minibatch_mean": mutant_mean})
            continue

        # Accepted: full D_pareto evaluation
        eval_on_files(cand, pareto_files)
        cand["pareto_mean"] = mean_on(cand, pareto_files)
        candidates.append(cand)
        print(f"[gepa]   candidate {cand['id']}: {mutant_mean:+.4f} "
              f"> parent — ACCEPTED; D_pareto mean {cand['pareto_mean']:+.4f}")

    # ── Final selection: best average on D_pareto (paper's return rule) ───────
    for c in candidates:
        c["pareto_mean"] = mean_on(c, pareto_files)
    winner = max(candidates, key=lambda c: c["pareto_mean"])

    out_path = output or (_AGENT_ROOT / "prompts" / "review_prompt_evolved.txt")
    adopted = winner["id"] != 0
    if adopted:
        out_path.write_text(winner["template"], encoding="utf-8")
    base_mean = candidates[0]["pareto_mean"]
    print(f"[gepa] winner: candidate {winner['id']} "
          f"(D_pareto mean {winner['pareto_mean']:+.4f} vs baseline "
          f"{base_mean:+.4f}) — "
          f"{'written to ' + str(out_path) if adopted else 'baseline kept'}")

    report = {
        "winner_id": winner["id"],
        "adopted": adopted,
        "baseline_pareto_mean": base_mean,
        "winner_pareto_mean": winner["pareto_mean"],
        "pareto_files": pareto_files,
        "candidates": [
            {"id": c["id"], "parent": c["parent"],
             "pareto_mean": round(c["pareto_mean"], 4),
             "files_scored": len(c["scores"])}
            for c in candidates
        ],
        "rejected": rejected,
        "output": str(out_path) if adopted else "(kept base prompt)",
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
        "Faithful skeleton: per-instance score matrix, acceptance gate,",
        "Pareto-frontier parent sampling (arXiv 2507.19457 Algorithm 1/2).", "",
        f"Winner: candidate {report['winner_id']}",
        f"Evolved prompt: `{report['output']}`",
        f"D_pareto files: {len(report['pareto_files'])}", "",
        "## Candidates (mean score on D_pareto)", "",
        "| ID | Parent | D_pareto mean | Files scored |",
        "|----|--------|--------------:|-------------:|",
    ]
    for c in report["candidates"]:
        parent = str(c["parent"]) if c["parent"] is not None else "—"
        tag = " (baseline)" if c["id"] == 0 else ""
        lines.append(f"| {c['id']}{tag} | {parent} | "
                     f"{c['pareto_mean']:+.4f} | {c['files_scored']} |")
    if report["rejected"]:
        lines += ["", "## Rejected at the acceptance gate", "",
                  "| ID | Parent | Minibatch mean |", "|----|--------|---------------:|"]
        for r in report["rejected"]:
            lines.append(f"| {r['id']} | {r['parent']} | "
                         f"{r['minibatch_mean']:+.4f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GEPA prompt evolution")
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--minibatch", type=int, default=8)
    parser.add_argument("--pareto-size", type=int, default=12)
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
        pareto_size=args.pareto_size,
        seed=args.seed,
        eval_model=args.eval_model,
        eval_provider_name=args.eval_provider,
        mutator_model=args.mutator_model,
        mutator_provider_name=args.mutator_provider,
        output=Path(args.output) if args.output else None,
        dry_run=args.dry_run,
    )
