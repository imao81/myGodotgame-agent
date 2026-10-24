"""tools/assertion_verifier.py — Stage 2 of Propose-Verify code review.

Statically verifies LLM assertions against the project index produced by
Stage 1 (tools/index_builder.py).  No LLM calls, no disk writes.

Public API
----------
    verify_assertion(assertion: dict, index: dict) -> dict

    assertion fields:
        type        str   — one of the assertion types below
        + type-specific fields (see each handler)

    returns:
        status          "confirmed" | "refuted" | "unverifiable"
        confidence      float  0.0–1.0
        details         str    human-readable explanation
        fix_suggestion  str | None

Supported assertion types
-------------------------
    method_is_not_async     — checks scripts[file].methods[method].async
    node_type_mismatch      — checks scenes[scene_file].nodes[node_path].type
    identifier_not_defined  — checks class_names + autoloads (+ case-insensitive)
    signal_not_declared     — checks scripts[file].signals + extends chain
    node_not_found          — checks scenes[scene_file].nodes ($ / % / nested leaf)
    file_not_exists         — checks the full files[] listing in the index

Stub (always returns unverifiable):
    unverifiable            — the LLM's explicit opt-out type
"""

from __future__ import annotations

import re
from typing import Any


# ── Result builder ─────────────────────────────────────────────────────────────

def _result(
    status: str,
    confidence: float,
    details: str,
    fix_suggestion: str | None = None,
) -> dict:
    return {
        "status": status,
        "confidence": round(confidence, 2),
        "details": details,
        "fix_suggestion": fix_suggestion,
    }


def _unverifiable(reason: str) -> dict:
    return _result("unverifiable", 0.0, reason)


def _confirmed(details: str, confidence: float = 0.9, fix: str | None = None) -> dict:
    return _result("confirmed", confidence, details, fix)


def _refuted(details: str, confidence: float = 0.95, fix: str | None = None) -> dict:
    return _result("refuted", confidence, details, fix)


# ── Index lookup helpers ───────────────────────────────────────────────────────

def _script_for_class(class_name: str, index: dict) -> str | None:
    """Return the res:// path for a class_name, or None if unknown."""
    return index.get("class_names", {}).get(class_name)


def _script_entry(res_path: str, index: dict) -> dict | None:
    return index.get("scripts", {}).get(res_path)


def _scene_entry(scene_file: str, index: dict) -> dict | None:
    return index.get("scenes", {}).get(scene_file)


def _normalise_node_path(node_path: str) -> str:
    """Ensure the node path has a $ prefix (e.g. 'Foo' → '$Foo')."""
    if not node_path.startswith(("$", "%")):
        return f"${node_path}"
    return node_path


# ── Handler: method_is_not_async ──────────────────────────────────────────────

def _verify_method_is_not_async(a: dict, index: dict) -> dict:
    """Verify that a method is NOT declared async.

    Assertion fields:
        target_class  str   — class_name as declared in the script
        method        str   — method name
        file_hint     str   — optional res:// path (used when class_name is null)
    """
    target_class: str = a.get("target_class", "")
    method_name: str = a.get("method", "")
    file_hint: str = a.get("file_hint", "")

    if not method_name:
        return _unverifiable("assertion missing 'method' field")

    # Resolve script path: prefer class lookup, fall back to file_hint
    res_path: str | None = None
    if target_class:
        res_path = _script_for_class(target_class, index)
    if res_path is None and file_hint:
        # file_hint may be a bare path without res:// prefix
        if not file_hint.startswith("res://"):
            file_hint = "res://" + file_hint.lstrip("/")
        # Verify it exists in the index
        if file_hint in index.get("scripts", {}):
            res_path = file_hint

    if res_path is None:
        who = target_class or file_hint or "(unknown)"
        return _unverifiable(
            f"class or file '{who}' not found in index; cannot verify method"
        )

    script = _script_entry(res_path, index)
    if script is None:
        return _unverifiable(f"script '{res_path}' not in index")

    methods: dict = script.get("methods", {})
    if method_name not in methods:
        return _unverifiable(
            f"method '{method_name}' not found in '{res_path}'; "
            "may be inherited or dynamically added"
        )

    is_async: bool = methods[method_name].get("async", False)
    loc = f"{res_path}:{methods[method_name].get('line', '?')}"

    if not is_async:
        return _confirmed(
            f"'{method_name}' in {loc} is declared without 'async' keyword",
            confidence=0.95,
        )
    else:
        return _refuted(
            f"'{method_name}' in {loc} IS declared async — "
            "the assertion that it is not async is wrong",
            confidence=0.98,
            fix=f"Remove 'async' from 'func {method_name}' or update the assertion.",
        )


# ── Handler: node_type_mismatch ───────────────────────────────────────────────

def _common_prefix_len(a: str, b: str) -> int:
    """Return the length of the longest common case-insensitive prefix of a and b."""
    i = 0
    while i < len(a) and i < len(b) and a[i].lower() == b[i].lower():
        i += 1
    return i


def _find_similar_nodes(node_path: str, nodes: dict) -> list[str]:
    """Return node keys whose name is similar to node_path (name without prefix)."""
    name = node_path.lstrip("$%")
    name_lower = name.lower()
    return [
        k for k in nodes
        if (
            name_lower in k.lower()
            or k.lower().lstrip("$%") in name_lower
            or _common_prefix_len(name, k.lstrip("$%")) >= 6
        )
    ]


def _lookup_node(node_path: str, nodes: dict) -> dict | None:
    """Look up a node by path, trying the alternate $ / % prefix as fallback."""
    entry = nodes.get(node_path)
    if entry is None:
        alt_prefix = "%" if node_path.startswith("$") else "$"
        entry = nodes.get(f"{alt_prefix}{node_path[1:]}")
    return entry


def _node_missing_result(node_path: str, scene_file: str, nodes: dict) -> dict:
    """Return a confirmed result for a node that is entirely absent from the scene."""
    similar = _find_similar_nodes(node_path, nodes)
    fix = None
    if similar:
        best = similar[0]
        node_type = nodes[best].get("type") or "(inherited)"
        fix = f"Node '{node_path}' not found. Did you mean {best} (type: {node_type})?"
    return _confirmed(
        f"Node '{node_path}' does not exist in '{scene_file}' at all "
        f"(not just a type mismatch). Scene nodes: {list(nodes.keys())}",
        confidence=0.98,
        fix=fix,
    )


def _type_match_result(
    node_path: str, scene_file: str, indexed_type: str,
    expected_type: str, actual_type: str,
) -> dict:
    """Return confirmed/refuted/unverifiable based on the indexed type."""
    if indexed_type is None:
        return _unverifiable(
            f"Node '{node_path}' exists in '{scene_file}' but has no explicit type "
            "(it is instanced from another scene); cannot verify type"
        )
    if actual_type and indexed_type == actual_type:
        fix = None
        if expected_type and expected_type != actual_type:
            fix = f"Update code to use the correct type '{actual_type}' instead of '{expected_type}'."
        return _confirmed(
            f"Node '{node_path}' in '{scene_file}' is type '{indexed_type}', "
            "matching the asserted actual_type — mismatch confirmed",
            confidence=0.97,
            fix=fix,
        )
    if expected_type and indexed_type == expected_type:
        return _refuted(
            f"Node '{node_path}' in '{scene_file}' is type '{indexed_type}', "
            f"which matches expected_type '{expected_type}' — no mismatch",
            confidence=0.97,
        )
    return _confirmed(
        f"Node '{node_path}' in '{scene_file}' has type '{indexed_type}', "
        f"which is neither expected '{expected_type}' nor actual '{actual_type}' — "
        "type mismatch assertion is broadly correct",
        confidence=0.75,
        fix=f"Code references type '{expected_type}'; actual node type is '{indexed_type}'.",
    )


def _verify_node_type_mismatch(a: dict, index: dict) -> dict:
    """Verify an asserted node-type mismatch against the scene index.

    Assertion fields:
        node_path      str  — e.g. "$CollisionShape3D" or "CollisionShape3D"
        expected_type  str  — type the code expects (e.g. "CollisionShape3D")
        actual_type    str  — type the LLM claims the node actually is
        scene_file     str  — res:// path to the .tscn file
    """
    node_path: str = _normalise_node_path(a.get("node_path", ""))
    expected_type: str = a.get("expected_type", "")
    actual_type: str = a.get("actual_type", "")
    scene_file: str = a.get("scene_file", "")

    if not scene_file:
        return _unverifiable("assertion missing 'scene_file' field")
    if not node_path or node_path in ("$", "%"):
        return _unverifiable("assertion missing 'node_path' field")

    scene = _scene_entry(scene_file, index)
    if scene is None:
        return _unverifiable(f"scene '{scene_file}' not in index; cannot verify node type")

    nodes: dict = scene.get("nodes", {})
    node_entry = _lookup_node(node_path, nodes)

    if node_entry is None:
        return _node_missing_result(node_path, scene_file, nodes)

    return _type_match_result(
        node_path, scene_file, node_entry.get("type"),
        expected_type, actual_type,
    )


# ── Handler: identifier_not_defined ───────────────────────────────────────────

def _verify_identifier_not_defined(a: dict, index: dict) -> dict:
    """Verify that an identifier is genuinely undefined in the project scope.

    Checks class_names and autoloads.  Falls back to case-insensitive search
    to catch typos (e.g. 'objectRegistry' vs 'ObjectRegistry').

    Assertion fields:
        identifier      str  — the symbol the LLM flagged as undefined
        suggested_match str  — optional correction the LLM proposed
        search_scope    str  — optional hint ("class_names"|"autoloads"|"all")
    """
    identifier: str = a.get("identifier", "")
    suggested_match: str = a.get("suggested_match", "")

    if not identifier:
        return _unverifiable("assertion missing 'identifier' field")

    class_names: dict = index.get("class_names", {})
    autoloads: dict = index.get("autoloads", {})

    # ── Exact match ────────────────────────────────────────────────────────────
    if identifier in class_names:
        res = class_names[identifier]
        return _refuted(
            f"'{identifier}' IS defined as a class_name in '{res}'",
            confidence=0.99,
        )

    if identifier in autoloads:
        res = autoloads[identifier]
        return _refuted(
            f"'{identifier}' IS a registered autoload singleton ('{res}')",
            confidence=0.99,
        )

    # ── Case-insensitive fallback ──────────────────────────────────────────────
    ident_lower = identifier.lower()

    ci_class = next(
        (k for k in class_names if k.lower() == ident_lower), None
    )
    if ci_class:
        return _confirmed(
            f"'{identifier}' not found (case-sensitive). "
            f"Close match: '{ci_class}' defined in '{class_names[ci_class]}'",
            confidence=0.85,
            fix=f"Did you mean '{ci_class}'? (Check capitalisation.)",
        )

    ci_auto = next(
        (k for k in autoloads if k.lower() == ident_lower), None
    )
    if ci_auto:
        return _confirmed(
            f"'{identifier}' not found (case-sensitive). "
            f"Autoload '{ci_auto}' has similar name.",
            confidence=0.85,
            fix=f"Did you mean autoload '{ci_auto}'? (Check capitalisation.)",
        )

    # ── Truly not found ────────────────────────────────────────────────────────
    fix = None
    if suggested_match:
        # Validate the LLM's own suggestion against the index
        if suggested_match in class_names or suggested_match in autoloads:
            fix = f"'{suggested_match}' exists in the project — use that instead."
        else:
            fix = (
                f"LLM suggested '{suggested_match}' but that is also not in the index. "
                "Check for a preload or local variable."
            )

    return _confirmed(
        f"'{identifier}' not found in class_names or autoloads",
        confidence=0.80,
        fix=fix,
    )


# ── Handler: signal_not_declared ──────────────────────────────────────────────

def _verify_signal_not_declared(a: dict, index: dict) -> dict:
    """Verify that a signal is NOT declared in a script or its ancestors.

    Walks the extends chain via class_names.  Engine built-in signals (e.g.
    ``finished``) are not indexed, so "confirmed" here means "not declared in
    any script in the inheritance chain" — stated honestly in the details.

    Assertion fields:
        signal_name  str  — the signal the LLM claims is missing
        file         str  — res:// path of the script where it should be declared
    """
    signal_name: str = a.get("signal_name", "")
    file: str = a.get("file", "")

    if not signal_name:
        return _unverifiable("assertion missing 'signal_name' field")
    if not file:
        return _unverifiable("assertion missing 'file' field")
    if not file.startswith("res://"):
        file = "res://" + file.lstrip("/")

    scripts: dict = index.get("scripts", {})
    script = scripts.get(file)
    if script is None:
        return _unverifiable(
            f"script '{file}' not in index; cannot check signal declarations"
        )

    # Walk the extends chain (extends may be a class_name → ancestor script)
    class_names: dict = index.get("class_names", {})
    visited: set[str] = set()
    current = script
    current_path: str | None = file

    while current is not None and current_path is not None and current_path not in visited:
        visited.add(current_path)
        if signal_name in current.get("signals", []):
            where = (
                "in the script itself" if current_path == file
                else f"in ancestor script '{current_path}'"
            )
            return _refuted(
                f"signal '{signal_name}' IS declared {where}",
                confidence=0.97,
            )
        extends = current.get("extends")
        parent_path = class_names.get(extends) if extends else None
        current = scripts.get(parent_path) if parent_path else None
        current_path = parent_path

    # Not declared anywhere in the script chain — check for a typo near-match
    close = next(
        (s for s in script.get("signals", []) if s.lower() == signal_name.lower()),
        None,
    )
    fix = f"Did you mean '{close}'? (Check capitalisation.)" if close else None
    return _confirmed(
        f"signal '{signal_name}' is not declared in '{file}' or its script ancestors "
        "(engine built-in signals are not indexed)",
        confidence=0.85,
        fix=fix,
    )


# ── Handler: node_not_found ────────────────────────────────────────────────────

def _verify_node_not_found(a: dict, index: dict) -> dict:
    """Verify that a node does NOT exist in a scene.

    The index stores child node names ($ / % prefixed), not full nested paths;
    for a nested path like ``$HUD/HealthBar`` only the leaf is checked, at
    reduced confidence.

    Assertion fields:
        node_path   str  — e.g. "$HealthBar" or "HUD/HealthBar"
        scene_file  str  — res:// path to the .tscn
    """
    node_path: str = a.get("node_path", "")
    scene_file: str = a.get("scene_file", "")

    if not node_path:
        return _unverifiable("assertion missing 'node_path' field")
    if not scene_file:
        return _unverifiable("assertion missing 'scene_file' field")
    if not scene_file.startswith("res://"):
        scene_file = "res://" + scene_file.lstrip("/")

    scene = _scene_entry(scene_file, index)
    if scene is None:
        return _unverifiable(f"scene '{scene_file}' not in index; cannot verify node")

    nodes: dict = scene.get("nodes", {})

    nested = "/" in node_path.lstrip("$%")
    leaf = node_path.rsplit("/", 1)[-1]
    key = _normalise_node_path(leaf)
    entry = _lookup_node(key, nodes)

    if entry is not None:
        note = " (leaf of nested path; parent chain not indexed)" if nested else ""
        return _refuted(
            f"Node '{key}' EXISTS in '{scene_file}'{note}",
            confidence=0.9 if nested else 0.97,
        )

    similar = _find_similar_nodes(key, nodes)
    fix = None
    if similar:
        best = similar[0]
        node_type = nodes[best].get("type") or "(inherited)"
        fix = f"Did you mean {best} (type: {node_type})?"
    return _confirmed(
        f"Node '{key}' does not exist in '{scene_file}'. "
        f"Scene nodes: {list(nodes.keys())}",
        confidence=0.9 if nested else 0.95,
        fix=fix,
    )


# ── Handler: file_not_exists ───────────────────────────────────────────────────

def _verify_file_not_exists(a: dict, index: dict) -> dict:
    """Verify that a res:// file does NOT exist in the project.

    Uses the full ``files`` listing when present (any asset type); falls back
    to scripts/scenes/autoloads only for indexes built by an older builder.

    Assertion fields:
        path  str  — res:// path the LLM claims is missing
    """
    path: str = a.get("path", "")

    if not path:
        return _unverifiable("assertion missing 'path' field")
    if not path.startswith("res://"):
        path = "res://" + path.lstrip("/")

    files: list = index.get("files", [])
    if files:
        if path in files:
            return _refuted(f"File '{path}' EXISTS in the project", confidence=0.98)
        base = path.rsplit("/", 1)[-1]
        same_name = [f for f in files if f.rsplit("/", 1)[-1] == base]
        fix = f"A file with the same name exists at: {same_name[0]}" if same_name else None
        return _confirmed(
            f"File '{path}' does not exist in the project",
            confidence=0.95,
            fix=fix,
        )

    # Legacy index without a file listing — partial check, honest about coverage
    partial_known = (
        set(index.get("scripts", {}))
        | set(index.get("scenes", {}))
        | set(index.get("autoloads", {}).values())
    )
    if path in partial_known:
        return _refuted(f"File '{path}' EXISTS in the project", confidence=0.98)
    return _unverifiable(
        f"'{path}' not among indexed scripts/scenes, but this index has no full "
        "file listing — rebuild the index to verify asset paths"
    )


# ── Dispatch table ─────────────────────────────────────────────────────────────

_HANDLERS = {
    "method_is_not_async": _verify_method_is_not_async,
    "node_type_mismatch": _verify_node_type_mismatch,
    "identifier_not_defined": _verify_identifier_not_defined,
    "signal_not_declared": _verify_signal_not_declared,
    "node_not_found": _verify_node_not_found,
    "file_not_exists": _verify_file_not_exists,
}

_STUBS = {"unverifiable"}


# ── Public API ─────────────────────────────────────────────────────────────────

def verify_assertion(assertion: dict, index: dict) -> dict:
    """Verify one LLM assertion against the static project index.

    Parameters
    ----------
    assertion:
        dict with at minimum a ``type`` key and type-specific fields.
    index:
        The project index dict produced by ``tools.index_builder.build_project_index``.

    Returns
    -------
    dict with keys:
        status          "confirmed" | "refuted" | "unverifiable"
        confidence      float
        details         str
        fix_suggestion  str | None
    """
    try:
        assertion_type: str = assertion.get("type", "")

        if assertion_type in _STUBS:
            return _stub(assertion_type)

        handler = _HANDLERS.get(assertion_type)
        if handler is None:
            return _unverifiable(
                f"unknown assertion type '{assertion_type}'; "
                f"supported: {sorted(_HANDLERS) + sorted(_STUBS)}"
            )

        return handler(assertion, index)

    except Exception as exc:  # noqa: BLE001
        # Never let a bug in the verifier crash the pipeline
        return _unverifiable(
            f"verifier raised an unexpected error ({type(exc).__name__}: {exc})"
        )


# ── Batch helper ───────────────────────────────────────────────────────────────

def verify_batch(assertions: list[dict], index: dict) -> list[dict]:
    """Verify a list of assertions and return results in the same order."""
    return [verify_assertion(a, index) for a in assertions]
