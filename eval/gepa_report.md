# GEPA Prompt Evolution Report

Faithful skeleton: per-instance score matrix, acceptance gate,
Pareto-frontier parent sampling (arXiv 2507.19457 Algorithm 1/2).

Winner: candidate 1
Evolved prompt: `D:\VSCodeWorkspace\AI Agent\GameAgent\harvester-agent\prompts\review_prompt_evolved.txt`
D_pareto files: 12

## Candidates (mean score on D_pareto)

| ID | Parent | D_pareto mean | Files scored |
|----|--------|--------------:|-------------:|
| 0 (baseline) | — | -0.3683 | 28 |
| 1 | 0 | -0.1650 | 27 |
| 3 | 1 | -0.3350 | 20 |
| 4 | 1 | -0.2467 | 20 |
| 5 | 0 | -0.2517 | 20 |
| 6 | 0 | -0.2900 | 20 |

## Rejected at the acceptance gate

| ID | Parent | Minibatch mean |
|----|--------|---------------:|
| 2 | 0 | +0.1825 |
