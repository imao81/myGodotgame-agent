# Pipeline Benchmark Report

**Golden set:** `D:\VSCodeWorkspace\AI Agent\GameAgent\harvester-agent\eval\golden_set.jsonl`  
**Labelled findings:** 49  
> `partial` findings count as 0.5 TP for precision.

---

## Overall Quality

| Metric | Value |
|--------|-------|
| Precision (TP+partial / all) | **50.6%** |
| False-positive rate          | 42.9% |

## Precision by `final_severity`

| Severity | Count | Precision | FP Rate |
|----------|------:|----------:|--------:|
| `error` | 7 | 81.8% | 14.3% |
| `info` | 14 | 40.7% | 57.1% |
| `warning` | 28 | 48.9% | 42.9% |

## Dormant Bug Accuracy

> 11 findings with `dormant_bug=true` were labelled.

| Metric | Value |
|--------|-------|
| Dormant precision (are they real defects?) | **76.5%** |
| Dormant FP rate                            | 18.2% |

## False-Positive Rate by `check_method`

| Method | Count | Precision | FP Rate |
|--------|------:|----------:|--------:|
| `llm` | 29 | 45.1% | 48.3% |
| `static` | 20 | 58.8% | 35.0% |

## Severity Calibration

> No severity calibration data found.
> To enable: add `sev:<value>` token to `human_notes`,
> e.g. `"human_notes": "sev:warning real issue but not a crash"`.

---

## Interpretation

| Range | Meaning |
|-------|---------|
| Precision ≥ 80% | Pipeline is production-quality |
| Precision 60–79% | Acceptable for assisted review; prompt tuning advised |
| Precision < 60%  | High noise; prompt or verifier work needed |
| FP rate (static) < FP rate (llm) | Verifier is working correctly |
| Dormant accuracy ≥ 70% | Dormant-bug escalation is well-calibrated |
