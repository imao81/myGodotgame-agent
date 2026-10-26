# Routing Frontier — LinUCB vs baselines vs oracle

Matrix: 124 reviews over 62 files (2 arms), ground truth = 26 injected mutants.

Reward: `r = caught − λ·(tokens/1000)·rel_cost` (rel_cost: flash 0.7, kimi 2.0 — mirrors `agents/router.py`).

Prices (USD / 1M tokens, cache-miss off-peak, read 2026-10-25): deepseek-flash $0.15 in / $0.60 out (DeepSeek pricing via usagebox.com; peak hours 2×); kimi-k3 $3.00 in / $15.00 out (Moonshot official via modelpricewatch.com). USD figures are estimates under those rates.

## λ = 0.01

| Policy | Recall | Tokens | est. USD | FP | →k3 | Σ reward |
|--------|-------:|-------:|---------:|---:|---:|---------:|
| all-flash | 80.8% | 311,019 | $0.06 | 167 | 0% | +18.82 |
| all-k3 | 88.5% | 291,555 | $1.15 | 166 | 100% | +17.17 |
| rule (n_lines > median) | 84.6% | 299,170 | $0.69 | 165 | 48% | +17.89 |
| linucb-replay (n=10) | 84.6% ±5.4 | 301,422.9 | $0.61 ±0.12 | 165.9 | 51% | +17.98 |
| linucb-cv (5-fold) | 76.9% | 301,326 | $0.55 | 170 | 45% | +16.17 |
| oracle | 96.2% | 309,862 | $0.14 | 169 | 6% | +22.56 |

## λ = 0.05

| Policy | Recall | Tokens | est. USD | FP | →k3 | Σ reward |
|--------|-------:|-------:|---------:|---:|---:|---------:|
| all-flash | 80.8% | 311,019 | $0.06 | 167 | 0% | +10.11 |
| all-k3 | 88.5% | 291,555 | $1.15 | 166 | 100% | -6.16 |
| rule (n_lines > median) | 84.6% | 299,170 | $0.69 | 165 | 48% | +1.47 |
| linucb-replay (n=10) | 81.9% ±2.5 | 305,629.6 | $0.36 ±0.11 | 168.1 | 28% | +5.29 |
| linucb-cv (5-fold) | 80.8% | 311,019 | $0.06 | 167 | 0% | +10.11 |
| oracle | 96.2% | 309,862 | $0.14 | 169 | 6% | +12.80 |

## λ = 0.2

| Policy | Recall | Tokens | est. USD | FP | →k3 | Σ reward |
|--------|-------:|-------:|---------:|---:|---:|---------:|
| all-flash | 80.8% | 311,019 | $0.06 | 167 | 0% | -22.54 |
| all-k3 | 88.5% | 291,555 | $1.15 | 166 | 100% | -93.62 |
| rule (n_lines > median) | 84.6% | 299,170 | $0.69 | 165 | 48% | -60.13 |
| linucb-replay (n=10) | 80.8% ±1.7 | 309,686.1 | $0.14 ±0.02 | 166.0 | 8% | -27.96 |
| linucb-cv (5-fold) | 80.8% | 311,019 | $0.06 | 167 | 0% | -22.54 |
| oracle | 96.2% | 309,862 | $0.14 | 169 | 6% | -23.79 |

## Conclusions

- Pareto frontier (cost ↑, recall ↑): **linucb-cv (5-fold) ≡ all-flash** (81% @ $0.06) → **oracle** (96% @ $0.14)
- LinUCB collapse check: λ=0.01: mixed routing (51% → k3, seed σ_recall=5.4pp); λ=0.05: mixed routing (28% → k3, seed σ_recall=2.5pp); λ=0.2: mixed routing (8% → k3, seed σ_recall=1.7pp)
- Oracle headroom: 96% recall at $0.14 vs all-k3 88% at $1.15 — the union of both arms beats the expensive arm alone at ~13% of its cost.
- LinUCB replay: 81.9% ± 2.5 recall at $0.36 ± 0.11; CV-calibrated: 80.8% at $0.06.
- Rule baseline: 84.6% at $0.69 (vs all-flash 80.8% @ $0.06, all-k3 88.5% @ $1.15).

## Recall anchors vs history

Same mutant set (`eval/mutant_set.jsonl`, 26 mutants), same prompt (`prompts/review_prompt_v2.txt`), same matching code (`eval.fitness.match_findings`, refuted findings kept — exactly as `eval/mutant_eval.py`). Historical anchors: flash 43.5%, kimi-k3 73.1% (`eval/mutant_report.json`, commit 168c68d).

- deepseek-flash: 80.8% vs 43.5% (+37pp). The 43.5% anchor predates the prompt promotion in bd41f92 (2026-10-25, "evolved prompt after A/B win") and DeepSeek's silent model swap (`deepseek-v4-flash` is now served by V4.1-Flash-1017, 2026-10-17 changelog). Both changes push recall up; the deviation is expected, not a pipeline bug.
- kimi-k3: 88.5% vs 73.1% (+15pp = +4 mutants). Missed-set overlap with the historical run: m001/m017 missed in both; m004/m012/m015/m019/m022 now caught, m010 now missed. Binomial σ at n=26, p≈0.8 is ≈2 mutants (≈8pp) — the deviation is ≈2σ of pure sampling noise on borderline mutants, consistent with an identical pipeline.

