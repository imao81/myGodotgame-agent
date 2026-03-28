"""tools/post_processor.py — Stage 3 verification enrichment (Kimi's layer).

Takes sanitized LLM findings, verifies each assertion against the project
index, then applies the severity transform matrix.

Severity matrix
---------------
  LLM severity | verified  | final_severity | dormant_bug
  -------------|-----------|----------------|------------
  error        | True      | error          | False
  error        | False     | (DROP)         | —
  error        | None      | warning        | True
  warning      | any       | warning        | False
  info         | any       | info           | False

Public API
----------
    apply_verification(findings: list[dict], index: dict) -> list[dict]

    Each output finding gains:
        verified          bool | None
        final_severity    str
        dormant_bug       bool
        trigger_condition str | None  (only when dormant_bug is True)
"""

from __future__ import annotations

import logging
from typing import Any

from tools.assertion_verifier import verify_assertion

_log = logging.getLogger(__name__)

# Map verifier status strings to the verified tristate.
_STATUS_TO_VERIFIED: dict[str, bool | None] = {
    "confirmed": True,
    "refuted": False,
    "unverifiable": None,
}


def _compute_final_severity(llm_severity: str, verified: bool | None) -> str | None:
    """Return the final_severity string, or None if the finding should be dropped."""
    if llm_severity == "error":
        if verified is True:
            return "error"
        if verified is False:
            return None  # DROP signal
        return "warning"  # verified is None

    if llm_severity == "warning":
        return "warning"

    return "info"   # covers "info" and any unexpected value


def _enrich(finding: dict, verified: bool | None, final_severity: str) -> dict:
    """Return a new dict with verification fields added to finding."""
    llm_severity = finding.get("severity", "info")
    dormant_bug = (llm_severity == "error" and verified is None)

    enriched = {
        **finding,
        "verified": verified,
        "final_severity": final_severity,
        "dormant_bug": dormant_bug,
        "trigger_condition": finding["message"] if dormant_bug else None,
    }
    return enriched


def apply_verification(findings: list[dict], index: dict) -> list[dict]:
    """Verify each finding's assertion and apply the severity transform matrix.

    Parameters
    ----------
    findings:
        Sanitized LLM findings (output of tools.parser.sanitize_findings).
    index:
        Project index dict from tools.index_builder.build_project_index.

    Returns
    -------
    list[dict] — enriched findings; findings with verified=False and
    severity=error are silently dropped (refuted errors).
    """
    results: list[dict] = []

    for finding in findings:
        assertion: dict = finding.get("assertion", {})
        llm_severity: str = finding.get("severity", "info")

        # Run the static verifier
        try:
            vr = verify_assertion(assertion, index)
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "[post_processor] verifier raised for assertion %r: %s",
                assertion.get("type"), exc,
            )
            vr = {"status": "unverifiable", "confidence": 0.0,
                  "details": f"verifier error: {exc}", "fix_suggestion": None}

        verified = _STATUS_TO_VERIFIED.get(vr["status"], None)
        final_severity = _compute_final_severity(llm_severity, verified)

        if final_severity is None:
            # Matrix says DROP: error that was refuted by the verifier
            _log.debug(
                "[post_processor] dropping refuted error: %s", finding.get("message", "")
            )
            continue

        results.append(_enrich(finding, verified, final_severity))

    return results
