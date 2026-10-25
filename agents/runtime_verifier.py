"""Runtime Verifier — re-judge dormant runtime assertions with playtest evidence.

Static verification cannot settle claims like "this Events signal is never
emitted" (dead emission path, always-false guard). Those land in the dormant
bucket (error severity + unverifiable). After the automated playtest we hold
ground truth: which bus signals actually fired, and whether the gameplay
context that should trigger them was exercised.

Honesty rule (context gating): a missing signal is NOT proof of a bug. The
scenario may simply never have triggered it. We only upgrade dormant→error
when every context signal fired while the target did not. A signal observed
firing refutes the claim outright. Anything else is left untouched.

Public API
----------
    verify_runtime(findings, observed_signals, playtest_ok) -> (findings, stats)
    coverage_notes(scope_files, coverage) -> list[meta_finding]
"""

from __future__ import annotations

# Default trigger context: signals that must have fired for the playtest to
# count as having exercised the target signal's scenario. An assertion may
# override this with its own expected_context list. Empty context (and no
# override) means the scenario is not reachable by the current playtest, so
# absence can never confirm — only observation can refute.
TRIGGER_CONTEXT: dict[str, list[str]] = {
    "mine_finished":     ["mine_started"],
    "station_spawned":   [],   # spawn phase — absence is meaningful on its own
    "asteroid_spawned":  [],
    "pirate_spawned":    [],
    "player_died":       ["__never__"],  # surviving spawn is correct behaviour
    "mine_started":      ["__never__"],  # controller does not pilot the ship
    "upgrade_unlocked":  ["__never__"],  # requires mining/docking gameplay
    "damaged":           ["__never__"],  # requires combat
}

# Sentinel meaning "the current playtest never exercises this scenario" —
# makes the context check unsatisfiable on purpose.
_NEVER = "__never__"

_RUNTIME_ASSERTION = "runtime_signal_not_emitted"


def _apply_runtime_verdict(finding: dict, status: str, details: str,
                           confidence: float) -> dict:
    updated = dict(finding)
    if status == "refuted":
        updated["verified"] = "refuted"
        updated["final_severity"] = "drop"
        updated["dormant_bug"] = False
        updated["trigger_condition"] = None
    else:  # confirmed
        updated["verified"] = "confirmed"
        updated["final_severity"] = "error"
        updated["dormant_bug"] = False
        updated["trigger_condition"] = None
    updated["_verifier"] = {
        **(finding.get("_verifier") or {}),
        "confidence": confidence,
        "details": details,
        "fix_suggestion": (finding.get("_verifier") or {}).get("fix_suggestion"),
        "runtime_evidence": True,
    }
    return updated


def verify_runtime(
    findings: list[dict],
    observed_signals: dict[str, int],
    playtest_ok: bool = True,
) -> tuple[list[dict], dict]:
    """Re-judge runtime_signal_not_emitted findings against playtest observations.

    Only touches findings still marked verified="unverifiable"; everything
    else passes through unchanged.

    Returns (updated_findings, stats) with counts for reporting.
    """
    stats = {"runtime_refuted": 0, "runtime_confirmed": 0, "runtime_unchecked": 0}
    if not playtest_ok or not observed_signals:
        stats["runtime_unchecked"] = sum(
            1 for f in findings
            if (f.get("assertion") or {}).get("type") == _RUNTIME_ASSERTION
            and f.get("verified") == "unverifiable"
        )
        return findings, stats

    updated: list[dict] = []
    for f in findings:
        assertion = f.get("assertion") or {}
        if (assertion.get("type") != _RUNTIME_ASSERTION
                or f.get("verified") != "unverifiable"):
            updated.append(f)
            continue

        signal = assertion.get("signal_name", "")
        if not signal:
            stats["runtime_unchecked"] += 1
            updated.append(f)
            continue

        fired = observed_signals.get(signal, 0)
        if fired > 0:
            stats["runtime_refuted"] += 1
            updated.append(_apply_runtime_verdict(
                f, "refuted",
                f"playtest ground truth: Events.{signal} fired {fired}x — "
                f"the emission path is reachable",
                0.95,
            ))
            continue

        context = assertion.get("expected_context")
        if context is None:
            context = TRIGGER_CONTEXT.get(signal, [_NEVER])
        context_met = bool(context) and all(
            c != _NEVER and observed_signals.get(c, 0) > 0 for c in context
        )
        if context_met:
            stats["runtime_confirmed"] += 1
            updated.append(_apply_runtime_verdict(
                f, "confirmed",
                f"playtest ground truth: context exercised "
                f"({', '.join(context)} fired) but Events.{signal} never did — "
                f"emission path confirmed dead at runtime",
                0.85,
            ))
        else:
            stats["runtime_unchecked"] += 1
            updated.append(f)  # scenario not exercised — leave dormant

    return updated, stats


def coverage_notes(scope_files: list[str], coverage: list[str]) -> list[dict]:
    """Info-level meta findings for in-scope scripts the playtest never loaded.

    coverage holds res:// paths collected from the live scene tree; scope_files
    are project-relative posix paths. Wording deliberately avoids "dead code" —
    the playtest only covers the spawn phase, so no-touch is a hint, not a verdict.
    """
    covered = {p[len("res://"):] for p in coverage if p.startswith("res://")}
    notes: list[dict] = []
    for rel in sorted(scope_files):
        rel_posix = rel.replace("\\", "/")
        if rel_posix.endswith("test_controller.gd"):
            continue
        if rel_posix not in covered:
            notes.append({
                "type": "runtime_coverage",
                "severity": "info",
                "message": f"{rel_posix} was not exercised by the automated "
                           f"playtest (spawn-phase observation only)",
            })
    return notes
