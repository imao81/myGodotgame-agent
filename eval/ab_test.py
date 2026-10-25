#!/usr/bin/env python3
"""eval/ab_test.py — production-arm A/B: base prompt vs evolved prompt.

Same provider, same index, same golden-covered files; the only variable is
the prompt template. This tests whether a prompt evolved against the cheap
eval tier (deepseek-flash) transfers to the production tier (kimi).

Usage:
    python eval/ab_test.py                          # full 32-file A/B
    python eval/ab_test.py --limit 8                # cheap smoke run

Output:
    eval/ab_report.md + eval/ab_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_AGENT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_AGENT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt A/B on the production tier")
    parser.add_argument("--model", default="kimi-k2.7-code")
    parser.add_argument("--provider", default="kimi")
    parser.add_argument("--limit", type=int, default=0,
                        help="Cap the number of golden-covered files (0 = all)")
    parser.add_argument("--evolved", default="prompts/review_prompt_evolved.txt")
    args = parser.parse_args()

    from eval.fitness import (
        load_golden, match_findings, replay_with_prompt, score_match,
    )
    from review import RealLLMProvider, load_or_build_index

    base_template = (_AGENT_ROOT / "prompts" / "review_prompt_v2.txt") \
        .read_text(encoding="utf-8")
    evolved_path = _AGENT_ROOT / args.evolved
    evolved_template = evolved_path.read_text(encoding="utf-8")

    golden = load_golden(_AGENT_ROOT / "eval" / "golden_set.jsonl")
    files = sorted({g["file"] for g in golden})
    if args.limit:
        files = files[: args.limit]
    golden_scope = [g for g in golden if g["file"] in set(files)]

    project_path = (_AGENT_ROOT.parent / "godot-2d-space-game" / "project").resolve()
    index, _ = load_or_build_index(project_path, kimi_dir=_AGENT_ROOT / ".kimi")

    # Production review budget (1536) — match what the pipeline actually uses.
    provider = RealLLMProvider(model=args.model, provider=args.provider)

    results: dict[str, dict] = {}
    for name, template in [("base", base_template),
                           ("evolved", evolved_template)]:
        print(f"[ab] replaying {len(files)} files with {name} prompt "
              f"on {args.provider}/{args.model}…")
        t0 = time.perf_counter()
        produced, errors = replay_with_prompt(template, files, project_path,
                                              index, provider)
        match = match_findings(produced, golden_scope)
        scored = score_match(match, golden_scope)
        scored["errors"] = len(errors)
        scored["wall_seconds"] = round(time.perf_counter() - t0, 1)
        results[name] = {**scored,
                         "missed_files": sorted({g["file"]
                                                 for g in match["missed"]})}
        print(f"[ab]   {name}: fitness {scored['fitness']:+.4f} "
              f"(recall {scored['recall_true']:.2f}, "
              f"fp_repro {scored['fp_repro_rate']:.2f}, "
              f"novel {scored['novel_count']}, "
              f"{scored['wall_seconds']}s)")

    delta = results["evolved"]["fitness"] - results["base"]["fitness"]
    verdict = ("evolved WINS on production tier" if delta > 0
               else "no transfer gain — keep base prompt" if delta == 0
               else "evolved LOSES on production tier")
    print(f"[ab] delta {delta:+.4f} → {verdict}")

    report = {
        "model": args.model, "provider": args.provider,
        "files": len(files), "golden_findings": len(golden_scope),
        "base": results["base"], "evolved": results["evolved"],
        "fitness_delta": round(delta, 4), "verdict": verdict,
    }
    (_AGENT_ROOT / "eval" / "ab_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Prompt A/B Report (production tier)", "",
        f"Model: `{args.provider}/{args.model}` — {len(files)} files, "
        f"{len(golden_scope)} labelled findings", "",
        "| Prompt | Fitness | Recall | FP repro | Novel | Wall |",
        "|--------|--------:|-------:|---------:|------:|-----:|",
    ]
    for name in ("base", "evolved"):
        r = results[name]
        lines.append(f"| {name} | {r['fitness']:+.4f} | {r['recall_true']:.2f} "
                     f"| {r['fp_repro_rate']:.2f} | {r['novel_count']} "
                     f"| {r['wall_seconds']}s |")
    lines += ["", f"**Delta: {delta:+.4f} — {verdict}**", ""]
    (_AGENT_ROOT / "eval" / "ab_report.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print("[ab] report → eval/ab_report.md")


if __name__ == "__main__":
    main()
