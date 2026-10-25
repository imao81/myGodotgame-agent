"""eval/fitness.py — replay-based fitness scorer for prompt candidates.

The golden set holds past findings with human labels (tp/fp/partial). To score
a *candidate prompt*, we replay review_file() with that prompt on the files
the golden set covers, then match the new findings against the labels:

  - re-discovered human-confirmed bug (tp/partial)  → recall credit
  - re-produced human-rejected false positive (fp)  → penalty
  - finding with no golden counterpart ("novel")    → small penalty each:
    truth is unknown, so we can't call it wrong — but a prompt that sprays
    unverifiable output is worse than a quiet one. Capped to stay honest.

fitness = recall_true − fp_repro_rate − novel_penalty

  recall_true    = matched (tp + 0.5·partial) weight / total true weight
  fp_repro_rate  = matched fp count / total golden fp count
  novel_penalty  = 0.02 · min(n_novel, 25)

Range is roughly [−1.5, 1.0]; 1.0 means "re-found every true bug, reproduced
zero false positives, produced zero noise". Recall is only over *known* true
bugs — a genuinely new real bug scores as novel noise. Accepted limitation:
the golden set is a labelled finding list, not a complete bug census.

Pure matching/scoring is LLM-free and unit-tested; replay_with_prompt() is
the only function that touches an LLM provider.
"""

from __future__ import annotations

import json
from pathlib import Path

# ── Golden set loading ─────────────────────────────────────────────────────────

_TRUE_LABELS = ("tp", "partial")
_PARTIAL_WEIGHT = 0.5
_NOVEL_PENALTY_EACH = 0.02
_NOVEL_PENALTY_CAP = 25
_LINE_TOLERANCE = 3


def load_golden(path: Path) -> list[dict]:
    """Labelled records only; 'disputed' is excluded from scoring entirely."""
    records: list[dict] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("human_label") in ("tp", "fp", "partial"):
                records.append(rec)
    return records


# ── Matching ───────────────────────────────────────────────────────────────────

def _entity_of(finding: dict) -> str | None:
    """Stable entity key from a finding's assertion (mirrors review.py cascade)."""
    a = finding.get("assertion") or {}
    t = a.get("type", "")
    field = {
        "identifier_not_defined": "identifier",
        "method_is_not_async": "method",
        "signal_not_declared": "signal_name",
        "runtime_signal_not_emitted": "signal_name",
        "node_not_found": "node_path",
        "node_type_mismatch": "node_path",
        "file_not_exists": "path",
    }.get(t)
    if field and a.get(field):
        return f"{t}:{a[field]}"
    return None


def _matches(produced: dict, golden: dict) -> bool:
    """A produced finding matches a golden record by file + entity, or file + nearby line."""
    if produced.get("file") != golden.get("file"):
        return False
    pe, ge = _entity_of(produced), _entity_of(golden)
    if pe and ge:
        return pe == ge
    pl, gl = produced.get("line"), golden.get("line")
    if isinstance(pl, int) and isinstance(gl, int):
        return abs(pl - gl) <= _LINE_TOLERANCE
    return False


def match_findings(produced: list[dict], golden: list[dict]) -> dict:
    """Match produced findings against golden records.

    Returns {"matched": [(produced, golden)], "novel": [produced],
             "missed": [golden]} — missed = true-labelled golden records the
    candidate failed to re-discover.
    """
    matched: list[tuple[dict, dict]] = []
    novel: list[dict] = []
    used_golden: set[int] = set()

    for p in produced:
        hit = None
        for i, g in enumerate(golden):
            if i in used_golden:
                continue
            if _matches(p, g):
                hit = i
                break
        if hit is None:
            novel.append(p)
        else:
            used_golden.add(hit)
            matched.append((p, golden[hit]))

    missed = [g for i, g in enumerate(golden)
              if i not in used_golden and g.get("human_label") in _TRUE_LABELS]
    return {"matched": matched, "novel": novel, "missed": missed}


# ── Scoring ────────────────────────────────────────────────────────────────────

def score_match(match: dict, golden: list[dict]) -> dict:
    """Compute fitness from a match result. Deterministic, LLM-free."""
    total_true = sum(
        1.0 if g["human_label"] == "tp" else _PARTIAL_WEIGHT
        for g in golden if g["human_label"] in _TRUE_LABELS
    )
    total_fp = sum(1 for g in golden if g["human_label"] == "fp")

    true_hit = fp_hit = 0.0
    for _, g in match["matched"]:
        if g["human_label"] == "tp":
            true_hit += 1.0
        elif g["human_label"] == "partial":
            true_hit += _PARTIAL_WEIGHT
        elif g["human_label"] == "fp":
            fp_hit += 1.0

    recall = true_hit / total_true if total_true else 0.0
    fp_rate = fp_hit / total_fp if total_fp else 0.0
    novel_penalty = _NOVEL_PENALTY_EACH * min(len(match["novel"]),
                                              _NOVEL_PENALTY_CAP)
    fitness = recall - fp_rate - novel_penalty

    return {
        "fitness": round(fitness, 4),
        "recall_true": round(recall, 4),
        "fp_repro_rate": round(fp_rate, 4),
        "novel_count": len(match["novel"]),
        "novel_penalty": round(novel_penalty, 4),
        "matched_true": true_hit,
        "matched_fp": int(fp_hit),
        "missed_true": len(match["missed"]),
        "n_produced": len(match["matched"]) + len(match["novel"]),
    }


# ── Per-instance scoring (GEPA's Pareto needs instance-level scores) ──────────

def score_files(produced: list[dict], golden: list[dict]) -> dict[str, float]:
    """Per-file instance score: true_weight − fp_hits − novel penalty.

    Instance = one file. Scores are compared only within the same instance
    (per-instance Pareto frontier), so files with different numbers of golden
    records need no cross-file normalisation.
    """
    files = {g["file"] for g in golden} | {f.get("file") for f in produced}
    scores: dict[str, float] = {}
    for f in files:
        g_sub = [g for g in golden if g["file"] == f]
        p_sub = [p for p in produced if p.get("file") == f]
        m = match_findings(p_sub, g_sub)
        true_hit = sum(1.0 if g["human_label"] == "tp" else _PARTIAL_WEIGHT
                       for _, g in m["matched"] if g["human_label"] in _TRUE_LABELS)
        fp_hit = sum(1 for _, g in m["matched"] if g["human_label"] == "fp")
        novel_pen = _NOVEL_PENALTY_EACH * min(len(m["novel"]), _NOVEL_PENALTY_CAP)
        scores[f] = round(true_hit - fp_hit - novel_pen, 4)
    return scores


# ── LLM replay (the only LLM-touching part) ────────────────────────────────────

def replay_with_prompt(
    template: str,
    files: list[str],
    project_root: Path,
    index: dict,
    provider,
) -> tuple[list[dict], list[str]]:
    """Run review_file with a candidate template on golden-covered files.

    Returns (all_findings, errors). Caller owns provider choice (cheap tier
    recommended for fitness evaluation).
    """
    from review import review_file  # local import: keeps unit tests LLM-free

    all_findings: list[dict] = []
    errors: list[str] = []
    root = Path(project_root)
    for rel in files:
        findings, err = review_file(root / rel, root, index, provider,
                                    template=template)
        if err:
            errors.append(f"{rel}: {err}")
        all_findings.extend(findings)
    return all_findings, errors


# ── Reflection text for the mutator (GEPA's textual gradient) ──────────────────

def feedback_text(scored: dict, match: dict, max_items: int = 8) -> str:
    """Natural-language fitness feedback for the reflection/mutation prompt."""
    lines = [
        f"Fitness {scored['fitness']:+.3f} "
        f"(recall {scored['recall_true']:.2f}, "
        f"FP reproduction {scored['fp_repro_rate']:.2f}, "
        f"novel findings {scored['novel_count']}).",
    ]
    if match["missed"]:
        lines.append("Known true bugs the prompt FAILED to find:")
        for g in match["missed"][:max_items]:
            lines.append(f"  - {g.get('file')}:{g.get('line')} — "
                         f"{g.get('message', '')[:140]}")
    fp_hits = [(p, g) for p, g in match["matched"] if g["human_label"] == "fp"]
    if fp_hits:
        lines.append("Known FALSE POSITIVES the prompt reproduced "
                     "(humans rejected these):")
        for _, g in fp_hits[:max_items]:
            lines.append(f"  - {g.get('file')}:{g.get('line')} — "
                         f"{g.get('message', '')[:140]}")
            if g.get("human_notes"):
                lines.append(f"    human: {g['human_notes'][:140]}")
    return "\n".join(lines)


def trace_feedback(produced: list[dict], golden: list[dict],
                   max_items: int = 6) -> str:
    """Trace-level feedback: the candidate's OWN findings with their verdicts.

    GEPA reflects on trajectories, not aggregate metrics. Our trajectory is the
    per-file finding list enriched by the static verifier — plus the golden
    verdict when the finding matches a labelled record. Four trace categories:
    what held up (confirmed), what the verifier shot down (refuted), what no
    human ever saw (novel), and what was missed entirely.
    """
    match = match_findings(produced, golden)
    lines: list[str] = []

    def _fmt(f: dict) -> str:
        return (f"  - {f.get('file')}:{f.get('line', '?')} "
                f"[verifier:{f.get('verified', '?')}] "
                f"{f.get('message', '')[:130]}")

    confirmed = [p for p, g in match["matched"]
                 if g["human_label"] in _TRUE_LABELS]
    if confirmed:
        lines.append("Findings that matched human-confirmed bugs (keep doing this):")
        lines += [_fmt(f) for f in confirmed[:max_items]]

    fps = [(p, g) for p, g in match["matched"] if g["human_label"] == "fp"]
    if fps:
        lines.append("Findings matching human-REJECTED false positives "
                     "(stop producing these):")
        for p, g in fps[:max_items]:
            lines.append(_fmt(p))
            if g.get("human_notes"):
                lines.append(f"    human: {g['human_notes'][:150]}")

    refuted_novel = [f for f in match["novel"] if f.get("verified") == "refuted"]
    if refuted_novel:
        lines.append("Novel findings the static verifier REFUTED "
                     "(the prompt asserts things that are statically false):")
        lines += [_fmt(f) for f in refuted_novel[:max_items]]

    other_novel = [f for f in match["novel"] if f.get("verified") != "refuted"]
    if other_novel:
        lines.append(f"Novel findings with no human label: {len(other_novel)} "
                     f"(sample below — unverifiable volume hurts the score):")
        lines += [_fmt(f) for f in other_novel[:3]]

    if match["missed"]:
        lines.append("Human-confirmed bugs the prompt MISSED:")
        for g in match["missed"][:max_items]:
            lines.append(f"  - {g.get('file')}:{g.get('line')} — "
                         f"{g.get('message', '')[:130]}")
            if g.get("human_notes"):
                lines.append(f"    human: {g['human_notes'][:150]}")

    return "\n".join(lines) if lines else "No findings produced at all."
