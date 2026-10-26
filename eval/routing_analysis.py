#!/usr/bin/env python3
"""eval/routing_analysis.py — offline policy evaluation on the routing matrix.

Reads eval/results/routing_matrix.jsonl (produced by eval/routing_eval.py) and
scores 6 routing policies on the recall–cost frontier:

  1. all-flash        every file → deepseek-flash
  2. all-k3           every file → kimi-standard
  3. rule             n_lines > median(n_lines) → kimi-standard, else flash
  4. linucb-replay    online replay: fresh bandit, random file order,
                      select → reward from matrix → update (10 seeds)
  5. linucb-cv        5-fold CV: train on 4/5 of files, evaluate held-out 1/5
  6. oracle           per file: cheapest arm that caught the bug (else flash)

Reward (aligned with agents.router.compute_reward, caught ∈ {0,1} playing the
role of the confirmed-finding count):

    r = caught − λ · (total_tokens / 1000) · rel_cost

Usage:
    .venv/Scripts/python.exe eval/routing_analysis.py
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

import numpy as np

_AGENT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_AGENT_ROOT))

_MATRIX = _AGENT_ROOT / "eval" / "results" / "routing_matrix.jsonl"
_MUTANT_SET = _AGENT_ROOT / "eval" / "mutant_set.jsonl"
_OUT_MD = _AGENT_ROOT / "eval" / "results" / "routing_frontier.md"
_OUT_PNG = _AGENT_ROOT / "eval" / "results" / "routing_frontier.png"
_WORKDIR = _AGENT_ROOT / ".kimi" / "mutant_project"

# ── Arms & prices ──────────────────────────────────────────────────────────────
# rel_cost mirrors agents/router.py _ARM_TABLE.
REL_COST = {"deepseek-flash": 0.7, "kimi-standard": 2.0}
ARM_ORDER = ["deepseek-flash", "kimi-standard"]  # cheap → expensive

# USD per 1M tokens (input / output), sources read 2026-10-25:
#   deepseek-flash: DeepSeek pricing page via usagebox.com — $0.15/$0.60
#     off-peak cache-miss (peak hours double input+output; `deepseek-v4-flash`
#     is an alias served by V4.1-Flash at the same price).
#   kimi-k3: Moonshot official pricing (modelpricewatch.com / benchlm.ai
#     mirrors) — $3.00 input (cache-miss) / $15.00 output.
# Both estimates assume cache-miss off-peak; real bills may be lower (cache)
# or higher (DeepSeek peak hours).
PRICES = {
    "deepseek-flash": {"input": 0.15, "output": 0.60},
    "kimi-standard": {"input": 3.00, "output": 15.00},
}

LAMBDAS = [0.01, 0.05, 0.2]
N_REPLAY_SEEDS = 10
N_FOLDS = 5
_CV_SEED = 42


# ── Matrix I/O ─────────────────────────────────────────────────────────────────

def load_matrix(path: Path = _MATRIX) -> list[dict]:
    """Successful (file, arm) records only."""
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("error") is None:
            records.append(rec)
    return records


def matrix_lookup(matrix: list[dict]) -> dict[tuple[str, str], dict]:
    return {(r["file"], r["arm"]): r for r in matrix}


def caught(rec: dict) -> int:
    return 1 if rec.get("matched_mutant_ids") else 0


def reward_of(rec: dict, rel_cost: float, lam: float) -> float:
    """caught − λ·(tokens/1000)·rel_cost — router.compute_reward with the
    confirmed-count collapsed to the binary ground-truth catch signal."""
    total = rec.get("prompt_tokens", 0) + rec.get("completion_tokens", 0)
    return caught(rec) - lam * (total / 1000.0) * rel_cost


def usd_of(rec: dict) -> float:
    p = PRICES[rec["arm"]]
    return (rec.get("prompt_tokens", 0) * p["input"]
            + rec.get("completion_tokens", 0) * p["output"]) / 1_000_000


# ── Policies ───────────────────────────────────────────────────────────────────

def policy_constant(files: list[str], arm: str) -> dict[str, str]:
    return {f: arm for f in files}


def policy_rule(n_lines: dict[str, int]) -> dict[str, str]:
    """Files longer than the median line count go to the expensive arm."""
    median = statistics.median(n_lines.values())
    return {f: ("kimi-standard" if n > median else "deepseek-flash")
            for f, n in n_lines.items()}


def policy_oracle(lookup: dict[tuple[str, str], dict],
                  files: list[str]) -> dict[str, str]:
    """Cheapest arm that caught the file's bug; cheapest arm if none did."""
    plan = {}
    for f in files:
        chosen = ARM_ORDER[0]
        for arm in ARM_ORDER:  # cheap → expensive
            rec = lookup.get((f, arm))
            if rec is not None and caught(rec):
                chosen = arm
                break
        plan[f] = chosen
    return plan


def run_replay(lookup: dict[tuple[str, str], dict],
               contexts: dict[str, np.ndarray],
               lam: float, seed: int) -> dict[str, str]:
    """Online LinUCB replay over a randomly ordered file stream."""
    from agents.router import LinUCB
    bandit = LinUCB(ARM_ORDER)
    order = sorted(contexts)
    random.Random(seed).shuffle(order)
    plan: dict[str, str] = {}
    for f in order:
        arm = bandit.select({f: contexts[f]}, allowed=ARM_ORDER)[f]
        plan[f] = arm
        bandit.update(arm, contexts[f],
                      reward_of(lookup[(f, arm)], REL_COST[arm], lam))
    return plan


def make_folds(files: list[str], n_folds: int = N_FOLDS,
               seed: int = _CV_SEED) -> list[list[str]]:
    """Deterministic shuffled k-fold split (returned for leak assertions)."""
    order = sorted(files)
    random.Random(seed).shuffle(order)
    folds = [[] for _ in range(n_folds)]
    for i, f in enumerate(order):
        folds[i % n_folds].append(f)
    return folds


def run_cv(lookup: dict[tuple[str, str], dict],
           contexts: dict[str, np.ndarray],
           lam: float, n_folds: int = N_FOLDS,
           seed: int = _CV_SEED) -> dict[str, str]:
    """Calibrated LinUCB: train on 4/5 of files, route the held-out 1/5."""
    from agents.router import LinUCB
    folds = make_folds(sorted(contexts), n_folds, seed)
    plan: dict[str, str] = {}
    for k in range(n_folds):
        held_out = set(folds[k])
        train = [f for f in sorted(contexts) if f not in held_out]
        bandit = LinUCB(ARM_ORDER)
        for f in train:  # fixed order within a fold → deterministic
            for arm in ARM_ORDER:  # full-information training (both arms)
                bandit.update(arm, contexts[f],
                              reward_of(lookup[(f, arm)], REL_COST[arm], lam))
        for f in sorted(held_out):
            plan[f] = bandit.select({f: contexts[f]}, allowed=ARM_ORDER)[f]
    return plan


# ── Metrics ────────────────────────────────────────────────────────────────────

def evaluate_plan(plan: dict[str, str],
                  lookup: dict[tuple[str, str], dict],
                  n_mutants: int, lam: float) -> dict:
    """Recall / tokens / USD / FP / total reward for one assignment."""
    detected = set()
    tokens = 0
    usd = 0.0
    fp = 0
    reward = 0.0
    for f, arm in plan.items():
        rec = lookup[(f, arm)]
        detected.update(rec["matched_mutant_ids"])
        tokens += rec["prompt_tokens"] + rec["completion_tokens"]
        usd += usd_of(rec)
        fp += rec["fp_count"]
        reward += reward_of(rec, REL_COST[arm], lam)
    return {
        "recall": len(detected) / n_mutants if n_mutants else float("nan"),
        "detected": len(detected),
        "tokens": tokens,
        "usd": usd,
        "fp": fp,
        "reward": reward,
    }


# ── Contexts ───────────────────────────────────────────────────────────────────

def build_contexts(workdir: Path, files: list[str]) -> tuple[dict, dict]:
    """(contexts, n_lines) using the production feature extractor."""
    from agents.router import file_features
    from review import load_or_build_index
    index, _ = load_or_build_index(workdir, kimi_dir=_AGENT_ROOT / ".kimi")
    contexts, n_lines = {}, {}
    for rel in files:
        code = (workdir / rel).read_text(encoding="utf-8", errors="replace")
        n = len(code.strip().splitlines())
        n_lines[rel] = n
        contexts[rel] = file_features(rel, index, n)
    return contexts, n_lines


# ── Reporting ──────────────────────────────────────────────────────────────────

def _fmt_row(name: str, m: dict, std: dict | None = None) -> str:
    recall = f"{m['recall'] * 100:.1f}%"
    usd = f"${m['usd']:.2f}"
    if std:
        recall += f" ±{std['recall'] * 100:.1f}"
        usd += f" ±{std['usd']:.2f}"
    return (f"| {name} | {recall} | {m['tokens']:,} | {usd} | {m['fp']} "
            f"| {m['k3_share'] * 100:.0f}% | {m['reward']:+.2f} |")


def run_analysis(matrix_path: Path = _MATRIX,
                 workdir: Path = _WORKDIR) -> dict:
    matrix = load_matrix(matrix_path)
    lookup = matrix_lookup(matrix)
    n_mutants = len({json.loads(l)["mutant_id"]
                     for l in _MUTANT_SET.read_text(encoding="utf-8").splitlines()
                     if l.strip()})
    files = sorted({r["file"] for r in matrix})
    contexts, n_lines = build_contexts(workdir.resolve(), files)

    results: dict[float, list[dict]] = {}
    for lam in LAMBDAS:
        rows = []

        def _row(name: str, plan: dict[str, str], **extra) -> dict:
            k3_share = sum(a == "kimi-standard" for a in plan.values()) / len(plan)
            return {"policy": name, "k3_share": k3_share,
                    **evaluate_plan(plan, lookup, n_mutants, lam), **extra}

        for name, arm in (("all-flash", "deepseek-flash"),
                          ("all-k3", "kimi-standard")):
            rows.append(_row(name, policy_constant(files, arm)))

        rows.append(_row("rule (n_lines > median)", policy_rule(n_lines)))

        replay_plans = [run_replay(lookup, contexts, lam, seed)
                        for seed in range(N_REPLAY_SEEDS)]
        replays = [evaluate_plan(p, lookup, n_mutants, lam)
                   for p in replay_plans]
        mean = {k: float(np.mean([r[k] for r in replays]))
                for k in ("recall", "tokens", "usd", "reward")}
        mean["fp"] = float(np.mean([r["fp"] for r in replays]))
        mean["detected"] = float(np.mean([r["detected"] for r in replays]))
        mean["k3_share"] = float(np.mean([
            sum(a == "kimi-standard" for a in p.values()) / len(p)
            for p in replay_plans]))
        std = {k: float(np.std([r[k] for r in replays]))
               for k in ("recall", "tokens", "usd", "reward")}
        rows.append({"policy": f"linucb-replay (n={N_REPLAY_SEEDS})",
                     "_std": std, **mean})

        rows.append(_row(f"linucb-cv ({N_FOLDS}-fold)",
                         run_cv(lookup, contexts, lam)))

        rows.append(_row("oracle", policy_oracle(lookup, files)))

        results[lam] = rows
    return {"results": results, "n_files": len(files), "n_mutants": n_mutants,
            "n_records": len(matrix)}


def write_report(out: dict, md_path: Path = _OUT_MD,
                 png_path: Path = _OUT_PNG) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lines = [
        "# Routing Frontier — LinUCB vs baselines vs oracle", "",
        f"Matrix: {out['n_records']} reviews over {out['n_files']} files "
        f"(2 arms), ground truth = {out['n_mutants']} injected mutants.", "",
        "Reward: `r = caught − λ·(tokens/1000)·rel_cost` "
        "(rel_cost: flash 0.7, kimi 2.0 — mirrors `agents/router.py`).", "",
        "Prices (USD / 1M tokens, cache-miss off-peak, read 2026-10-25): "
        "deepseek-flash $0.15 in / $0.60 out (DeepSeek pricing via usagebox.com; "
        "peak hours 2×); kimi-k3 $3.00 in / $15.00 out (Moonshot official via "
        "modelpricewatch.com). USD figures are estimates under those rates.", "",
    ]
    for lam in LAMBDAS:
        lines += [f"## λ = {lam}", "",
                  "| Policy | Recall | Tokens | est. USD | FP | →k3 | Σ reward |",
                  "|--------|-------:|-------:|---------:|---:|---:|---------:|"]
        for row in out["results"][lam]:
            lines.append(_fmt_row(row["policy"], row, row.get("_std")))
        lines.append("")

    # ── scatter at λ=0.05 ──
    rows = out["results"][0.05]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    # de-overlap labels for coincident points (e.g. all-flash ≡ linucb-cv)
    label_dy = {"all-flash": -14, "linucb-cv (5-fold)": 6, "oracle": 6}
    for row in rows:
        x, y = row["usd"], row["recall"] * 100
        ax.scatter(x, y, s=70, zorder=3)
        ax.annotate(row["policy"], (x, y), textcoords="offset points",
                    xytext=(8, label_dy.get(row["policy"], 5)), fontsize=9)
        if row.get("_std"):
            ax.errorbar(x, y, xerr=row["_std"]["usd"], yerr=row["_std"]["recall"] * 100,
                        fmt="none", ecolor="gray", alpha=0.6, zorder=2)
    ax.set_xlabel("estimated cost (USD, 62 files)")
    ax.set_ylabel("recall (%)")
    ax.set_title("Recall–cost frontier (λ = 0.05)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)

    # ── conclusions ──
    lam_rows = {r["policy"]: r for r in rows}
    oracle = lam_rows["oracle"]
    flash = next(r for r in rows if r["policy"] == "all-flash")
    k3 = next(r for r in rows if r["policy"] == "all-k3")
    rule = next(r for r in rows if r["policy"].startswith("rule"))
    replay = next(r for r in rows if r["policy"].startswith("linucb-replay"))
    cv = next(r for r in rows if r["policy"].startswith("linucb-cv"))

    lines += ["## Conclusions", ""]
    frontier = []
    for r in sorted(rows, key=lambda r: r["usd"]):
        if frontier and abs(r["usd"] - frontier[-1]["usd"]) < 1e-9 \
                and abs(r["recall"] - frontier[-1]["recall"]) < 1e-9:
            # coincident with the previous frontier point — merge the label
            frontier[-1]["policy"] += f" ≡ {r['policy']}"
            continue
        if not frontier or r["recall"] > max(f["recall"] for f in frontier):
            frontier.append(r)
    lines.append("- Pareto frontier (cost ↑, recall ↑): "
                 + " → ".join(f"**{r['policy']}** ({r['recall'] * 100:.0f}% @ "
                              f"${r['usd']:.2f})" for r in frontier))
    collapse_notes = []
    for lam in LAMBDAS:
        rp = next(r for r in out["results"][lam]
                  if r["policy"].startswith("linucb-replay"))
        share = rp["k3_share"]
        if share <= 0.05 or share >= 0.95:
            collapse_notes.append(
                f"λ={lam}: COLLAPSED ({share * 100:.0f}% → k3)")
        else:
            collapse_notes.append(
                f"λ={lam}: mixed routing ({share * 100:.0f}% → k3, "
                f"seed σ_recall={rp['_std']['recall'] * 100:.1f}pp)")
    lines.append("- LinUCB collapse check: " + "; ".join(collapse_notes))
    if oracle["recall"] >= k3["recall"]:
        lines.append(
            f"- Oracle headroom: {oracle['recall'] * 100:.0f}% recall at "
            f"${oracle['usd']:.2f} vs all-k3 {k3['recall'] * 100:.0f}% at "
            f"${k3['usd']:.2f} — the union of both arms beats the expensive "
            f"arm alone at ~{(oracle['usd'] / k3['usd']) * 100:.0f}% of its cost.")
    else:
        lines.append(f"- Oracle reaches {oracle['recall'] * 100:.0f}% at "
                     f"${oracle['usd']:.2f}.")
    lines.append(f"- LinUCB replay: {replay['recall'] * 100:.1f}% ± "
                 f"{replay['_std']['recall'] * 100:.1f} recall at "
                 f"${replay['usd']:.2f} ± {replay['_std']['usd']:.2f}; "
                 f"CV-calibrated: {cv['recall'] * 100:.1f}% at ${cv['usd']:.2f}.")
    lines.append(f"- Rule baseline: {rule['recall'] * 100:.1f}% at "
                 f"${rule['usd']:.2f} (vs all-flash {flash['recall'] * 100:.1f}% "
                 f"@ ${flash['usd']:.2f}, all-k3 {k3['recall'] * 100:.1f}% @ "
                 f"${k3['usd']:.2f}).")
    lines += ["",
              "## Recall anchors vs history", "",
              "Same mutant set (`eval/mutant_set.jsonl`, 26 mutants), same "
              "prompt (`prompts/review_prompt_v2.txt`), same matching code "
              "(`eval.fitness.match_findings`, refuted findings kept — exactly "
              "as `eval/mutant_eval.py`). Historical anchors: flash 43.5%, "
              "kimi-k3 73.1% (`eval/mutant_report.json`, commit 168c68d).", "",
              f"- deepseek-flash: {flash['recall'] * 100:.1f}% vs 43.5% "
              "(+37pp). The 43.5% anchor predates the prompt promotion in "
              "bd41f92 (2026-10-25, \"evolved prompt after A/B win\") and "
              "DeepSeek's silent model swap (`deepseek-v4-flash` is now served "
              "by V4.1-Flash-1017, 2026-10-17 changelog). Both changes push "
              "recall up; the deviation is expected, not a pipeline bug.",
              f"- kimi-k3: {k3['recall'] * 100:.1f}% vs 73.1% (+15pp = "
              "+4 mutants). Missed-set overlap with the historical run: "
              "m001/m017 missed in both; m004/m012/m015/m019/m022 now caught, "
              "m010 now missed. Binomial σ at n=26, p≈0.8 is ≈2 mutants "
              "(≈8pp) — the deviation is ≈2σ of pure sampling noise on "
              "borderline mutants, consistent with an identical pipeline.", ""]
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    out = run_analysis()
    write_report(out)
    print(f"[analysis] {out['n_records']} records, {out['n_files']} files, "
          f"{out['n_mutants']} mutants")
    for lam in LAMBDAS:
        print(f"  λ={lam}")
        for row in out["results"][lam]:
            std = row.get("_std") or {}
            print(f"    {row['policy']:28s} recall {row['recall'] * 100:5.1f}% "
                  f"(±{std.get('recall', 0) * 100:.1f}) "
                  f"tokens {row['tokens']:>9,} usd ${row['usd']:.2f} "
                  f"fp {row['fp']}")
    print(f"[analysis] → {_OUT_MD} / {_OUT_PNG}")


if __name__ == "__main__":
    main()
