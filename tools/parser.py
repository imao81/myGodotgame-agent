"""tools/parser.py — Stage 3 defensive parser (Claude's layer).

Sanitizes raw LLM JSON output before it reaches the verifier:
  - Strips keys the LLM must never emit (verified, dormant_bug, etc.)
  - Validates required fields are present
  - Normalises the assertion sub-object

Public API
----------
    sanitize_llm_output(raw: dict) -> dict
        Raises ValueError if required fields are missing.

    sanitize_findings(raw_list: list[dict]) -> list[dict]
        Applies sanitize_llm_output to each element.
        Findings that fail validation are logged and dropped.
"""

from __future__ import annotations

import logging

_log = logging.getLogger(__name__)

# Fields the LLM must never emit — stripped unconditionally.
_FORBIDDEN: frozenset[str] = frozenset(
    {"verified", "dormant_bug", "trigger_condition", "final_severity"}
)

# Fields every valid finding must have.
_REQUIRED: frozenset[str] = frozenset({"line", "severity", "category", "message", "assertion"})

# Valid severity values.
_VALID_SEVERITIES: frozenset[str] = frozenset({"error", "warning", "info"})

# Required keys inside the assertion sub-object.
_ASSERTION_REQUIRED: frozenset[str] = frozenset({"type"})

# Placeholder values the LLM uses when it has no real assertion — treated as garbage.
_GARBAGE_VALUES: frozenset[str] = frozenset({"__no_signal__", "__no_identifier__", ""})


def sanitize_llm_output(raw: dict) -> dict:
    """Clean a single raw LLM finding dict.

    Steps
    -----
    1. Strip forbidden keys (verified, dormant_bug, trigger_condition, final_severity).
    2. Validate required top-level keys are present.
    3. Normalise severity to lower-case; default to "info" if invalid.
    4. Ensure assertion is a dict with at least a "type" key.

    Returns
    -------
    dict — clean finding, ready for the verifier.

    Raises
    ------
    ValueError — if any required field is absent after stripping.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"finding must be a dict, got {type(raw).__name__}")

    # Step 1 — strip forbidden keys
    clean = {k: v for k, v in raw.items() if k not in _FORBIDDEN}

    # Step 2 — validate required fields
    missing = _REQUIRED - clean.keys()
    if missing:
        raise ValueError(
            f"finding is missing required fields: {sorted(missing)}  "
            f"(present: {sorted(clean.keys())})"
        )

    # Step 3 — normalise severity
    sev = str(clean.get("severity", "")).lower()
    if sev not in _VALID_SEVERITIES:
        _log.debug(
            "[parser] unknown severity %r, defaulting to 'info'", clean.get("severity")
        )
        clean["severity"] = "info"
    else:
        clean["severity"] = sev

    # Step 4 — validate assertion sub-object
    assertion = clean.get("assertion")
    if not isinstance(assertion, dict):
        raise ValueError(
            f"'assertion' must be a dict, got {type(assertion).__name__}"
        )
    missing_a = _ASSERTION_REQUIRED - assertion.keys()
    if missing_a:
        raise ValueError(
            f"assertion sub-object missing required fields: {sorted(missing_a)}"
        )

    # Step 5 — detect garbage assertions and normalise to "unverifiable"
    atype = assertion.get("type", "")
    if atype == "signal_not_declared" and assertion.get("signal_name") in _GARBAGE_VALUES:
        _log.debug("[parser] garbage signal_not_declared assertion normalised to unverifiable")
        assertion["type"] = "unverifiable"
        atype = "unverifiable"
    if atype == "identifier_not_defined" and assertion.get("identifier") in _GARBAGE_VALUES:
        _log.debug("[parser] garbage identifier_not_defined assertion normalised to unverifiable")
        assertion["type"] = "unverifiable"
        atype = "unverifiable"

    # Step 6 — unverifiable findings cannot be severity "error"
    if atype == "unverifiable" and clean["severity"] == "error":
        _log.debug("[parser] demoting unverifiable finding from 'error' to 'warning'")
        clean["severity"] = "warning"

    return clean


def sanitize_findings(raw_list: list) -> list[dict]:
    """Apply sanitize_llm_output to each element in a list.

    Invalid findings are logged and dropped rather than raising, so a single
    malformed finding cannot abort the entire review.
    """
    results: list[dict] = []
    for i, item in enumerate(raw_list):
        try:
            results.append(sanitize_llm_output(item))
        except (ValueError, TypeError) as exc:
            _log.warning("[parser] dropping finding[%d]: %s", i, exc)
    return results
