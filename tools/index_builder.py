"""tools/index_builder.py — Stage 1 of Propose-Verify code review.

Scans a Godot project root and emits .kimi/project_index.json with a
rich, authoritative structural index.  Stage 2 (assertion verifier) uses
this index to ground-truth every LLM assertion before it reaches the report.

Schema emitted
--------------
{
  "generated_at": "ISO-8601",
  "godot_version": "4.x",
  "scripts":   { "res://...gd":  { class_name, extends, is_tool, signals,
                                   methods, preloads, onready_vars,
                                   has_async_methods } },
  "scenes":    { "res://...tscn": { format, root_class, root_type,
                                    nodes, script_path } },
  "autoloads": { "Events": "res://autoload/events.gd", ... },
  "class_names":       { "ClassName": "res://...gd" },
  "signal_signatures": { "res://...gd": { "signal_name": ["arg: Type", ...] } },
  "files":             [ "res://...", ... ]   # every project file (dot-dirs excluded)
}

Usage
-----
    python tools/index_builder.py --project /path/to/godot/project
    python tools/index_builder.py --project ./test-game --output ./out/index.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# ── GDScript regex patterns ────────────────────────────────────────────────────

_TOOL_RE = re.compile(r"^\s*@tool\b", re.MULTILINE)
_CLASS_NAME_RE = re.compile(r"^\s*class_name\s+(\w+)", re.MULTILINE)
_EXTENDS_RE = re.compile(r"^\s*extends\s+(\S+)", re.MULTILINE)

# Signal: "signal foo" or "signal foo(bar: float, baz: Node2D)"
_SIGNAL_RE = re.compile(r"^\s*signal\s+(\w+)\s*(?:\(([^)]*)\))?", re.MULTILINE)

# @onready var name …
_ONREADY_RE = re.compile(r"^\s*@onready\s+var\s+(\w+)", re.MULTILINE)

# preload("res://…") or load("res://…")
_PRELOAD_RE = re.compile(r'(?:preload|load)\s*\(\s*"(res://[^"]+)"')

# Function declarations.  Captures: indent | name | args-string
# Handles all orderings of async / static in front of func.
_FUNC_DECL_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?:(?:async|static)\s+)*func\s+"
    r"(?P<name>\w+)\s*\((?P<args>[^)]*)\)",
    re.MULTILINE,
)

# Detect the 'async' keyword specifically on a func declaration line
_ASYNC_KEYWORD_RE = re.compile(r"\basync\b")

# ── .tscn regex patterns ───────────────────────────────────────────────────────

# Any key="value" attribute inside a section header
_ATTR_RE = re.compile(r'(\w+)\s*=\s*"([^"]*)"')

# script = ExtResource("id") inside a node body
_SCRIPT_REF_RE = re.compile(r'^\s*script\s*=\s*ExtResource\("([^"]+)"\)', re.MULTILINE)

# unique_name_in_owner = true
_UNIQUE_NAME_RE = re.compile(r'\bunique_name_in_owner\s*=\s*true\b')

# ── project.godot patterns ─────────────────────────────────────────────────────

_GODOT_FEATURES_RE = re.compile(
    r'config/features\s*=\s*PackedStringArray\s*\("([^"]+)"'
)
# Fallback: config_version=4 → Godot 3.x; config_version=5 → Godot 4.x
_CONFIG_VERSION_RE = re.compile(r'^config_version\s*=\s*(\d+)', re.MULTILINE)
_AUTOLOAD_SECTION_RE = re.compile(r'\[autoload\](.*?)(?=\n\[|\Z)', re.DOTALL)
_AUTOLOAD_ENTRY_RE = re.compile(r'^(\w+)\s*=\s*"\*?(res://[^"]+)"', re.MULTILINE)


# ── Low-level helpers ──────────────────────────────────────────────────────────

def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _parse_tscn_attrs(attr_str: str) -> dict[str, str]:
    """Extract key="value" pairs from a .tscn section-header attribute string."""
    return {m.group(1): m.group(2) for m in _ATTR_RE.finditer(attr_str)}


def _parse_func_args(args_str: str) -> list[dict[str, str | None]]:
    """'x: int, y: Node2D = null' → [{"name":"x","type":"int"}, {"name":"y","type":"Node2D"}].

    Strips default values.  Returns empty list for empty/blank input.
    """
    args_str = args_str.strip()
    if not args_str:
        return []
    result: list[dict[str, str | None]] = []
    for raw in args_str.split(","):
        raw = raw.strip()
        if not raw:
            continue
        # Drop default value
        raw = re.split(r"\s*=\s*", raw, maxsplit=1)[0].strip()
        if ":" in raw:
            name_part, type_part = raw.split(":", 1)
            result.append({"name": name_part.strip(), "type": type_part.strip()})
        else:
            result.append({"name": raw, "type": None})
    return result


# ── .gd file parser ────────────────────────────────────────────────────────────

def _parse_gd(res_path: str, text: str) -> dict:
    """Return the scripts-entry dict for one .gd file."""

    is_tool = bool(_TOOL_RE.search(text))

    m = _CLASS_NAME_RE.search(text)
    class_name = m.group(1) if m else None

    m = _EXTENDS_RE.search(text)
    extends = m.group(1) if m else None

    # Signals (names only)
    signals: list[str] = [m.group(1) for m in _SIGNAL_RE.finditer(text)]

    # @onready var names
    onready_vars: list[str] = [m.group(1) for m in _ONREADY_RE.finditer(text)]

    # preload / load paths (deduplicated, order-preserved)
    seen: set[str] = set()
    preloads: list[str] = []
    for p in _PRELOAD_RE.findall(text):
        if p not in seen:
            seen.add(p)
            preloads.append(p)

    # Methods ----------------------------------------------------------------
    # Collect all func-declaration match objects to slice bodies between them.
    func_matches = list(_FUNC_DECL_RE.finditer(text))

    methods: dict[str, dict] = {}
    has_async_methods = False

    for idx, fm in enumerate(func_matches):
        name = fm.group("name")
        args_str = fm.group("args")
        indent = fm.group("indent")
        is_async = bool(_ASYNC_KEYWORD_RE.search(fm.group(0)))

        # 1-indexed line number of the func declaration
        line_no = text[: fm.start()].count("\n") + 1

        # Body: text between end of this decl and start of the next func decl
        # (or end of file).  This is a heuristic but sufficient for 'await' detection.
        body_start = fm.end()
        body_end = func_matches[idx + 1].start() if idx + 1 < len(func_matches) else len(text)
        body = text[body_start:body_end]

        body_has_await = bool(re.search(r"\bawait\b", body))
        if body_has_await:
            has_async_methods = True

        methods[name] = {
            "async": is_async,
            "args": _parse_func_args(args_str),
            "line": line_no,
        }

    return {
        "class_name": class_name,
        "extends": extends,
        "is_tool": is_tool,
        "signals": signals,
        "methods": methods,
        "preloads": preloads,
        "onready_vars": onready_vars,
        "has_async_methods": has_async_methods,
    }


# ── signal_signatures extractor ────────────────────────────────────────────────

def _extract_signal_signatures(res_path: str, text: str) -> dict[str, list[str]]:
    """Return {signal_name: ["arg: Type", ...]} for signals that have parameters."""
    sigs: dict[str, list[str]] = {}
    for m in _SIGNAL_RE.finditer(text):
        name = m.group(1)
        params_str = (m.group(2) or "").strip()
        if not params_str:
            continue
        # Each param may be "name: Type" or just "name"
        params = [p.strip() for p in params_str.split(",") if p.strip()]
        sigs[name] = params
    return sigs


# ── .tscn file parser ──────────────────────────────────────────────────────────

def _parse_tscn_sections(text: str) -> list[tuple[str, dict[str, str], str]]:
    """Split a .tscn file into (section_type, attrs_dict, body_text) tuples.

    section_type is the first word inside the brackets, e.g. "node",
    "ext_resource", "sub_resource", "gd_scene".
    """
    sections: list[tuple[str, dict[str, str], str]] = []
    lines = text.splitlines(keepends=True)
    i = 0
    current_type: str | None = None
    current_attrs: dict[str, str] = {}
    current_body_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        # New section header?
        header_m = re.match(r'^\[(\w+)\s*(.*?)\]$', stripped)
        if header_m:
            # Flush previous section
            if current_type is not None:
                sections.append((
                    current_type,
                    current_attrs,
                    "".join(current_body_lines),
                ))
            current_type = header_m.group(1)
            current_attrs = _parse_tscn_attrs(header_m.group(2))
            current_body_lines = []
        else:
            if current_type is not None:
                current_body_lines.append(line)

    # Flush last section
    if current_type is not None:
        sections.append((current_type, current_attrs, "".join(current_body_lines)))

    return sections


def _parse_tscn(tscn_res_path: str, text: str, script_entries: dict) -> dict:
    """Return the scenes-entry dict for one .tscn file.

    script_entries is the already-parsed scripts dict so we can resolve
    root_type / root_class from the attached script's extends / class_name.
    """
    sections = _parse_tscn_sections(text)

    # Detect file format: gd_scene header carries format=3 (text) or format=2 (binary)
    fmt_str = "text"
    for sec_type, attrs, _ in sections:
        if sec_type == "gd_scene":
            fmt_str = "binary" if attrs.get("format", "3") == "2" else "text"
            break

    # Build ext_resource id → {type, path}
    ext_res: dict[str, dict[str, str | None]] = {}
    for sec_type, attrs, _ in sections:
        if sec_type == "ext_resource" and "id" in attrs:
            ext_res[attrs["id"]] = {
                "type": attrs.get("type"),
                "path": attrs.get("path"),
            }

    # Parse node sections
    nodes: dict[str, dict] = {}
    root_type: str | None = None
    root_class: str | None = None
    scene_script_path: str | None = None

    for sec_type, attrs, body in sections:
        if sec_type != "node":
            continue

        node_name = attrs.get("name", "")
        node_type = attrs.get("type")      # absent for instanced/inherited nodes
        is_root = "parent" not in attrs    # root node has no parent attribute

        # Unique name: look for property in body
        is_unique = bool(_UNIQUE_NAME_RE.search(body))

        # Script reference in body
        sr_m = _SCRIPT_REF_RE.search(body)
        node_script_path: str | None = None
        if sr_m:
            res_id = sr_m.group(1)
            er = ext_res.get(res_id, {})
            if er.get("type") == "Script":
                node_script_path = er.get("path")

        if is_root:
            scene_script_path = node_script_path

            # root_type: prefer explicit type attr, else infer from script
            if node_type:
                root_type = node_type
            elif node_script_path and node_script_path in script_entries:
                root_type = script_entries[node_script_path].get("extends")

            # root_class: from script class_name
            if node_script_path and node_script_path in script_entries:
                root_class = script_entries[node_script_path].get("class_name")

        else:
            # Child / descendant node
            key = f"%{node_name}" if is_unique else f"${node_name}"
            nodes[key] = {
                "type": node_type,   # may be None for instanced sub-nodes
                "unique_name": is_unique,
            }

    return {
        "format": fmt_str,
        "root_class": root_class,
        "root_type": root_type,
        "nodes": nodes,
        "script_path": scene_script_path,
    }


# ── project.godot parser ───────────────────────────────────────────────────────

def _parse_project_godot(project_root: Path) -> tuple[str, dict[str, str]]:
    """Return (godot_version, autoloads_dict) from project.godot.

    autoloads_dict: {"Events": "res://autoload/events.gd", ...}
    """
    godot_file = project_root / "project.godot"
    if not godot_file.exists():
        return "unknown", {}

    text = _read(godot_file)

    # Godot version: prefer config/features (exact), fall back to config_version field
    ver = "unknown"
    m = _GODOT_FEATURES_RE.search(text)
    if m:
        ver = m.group(1)   # e.g. "4.3"
    else:
        cv_m = _CONFIG_VERSION_RE.search(text)
        if cv_m:
            cv = int(cv_m.group(1))
            ver = "4.x" if cv >= 5 else "3.x" if cv == 4 else "unknown"

    # Autoloads
    autoloads: dict[str, str] = {}
    sec_m = _AUTOLOAD_SECTION_RE.search(text)
    if sec_m:
        for entry_m in _AUTOLOAD_ENTRY_RE.finditer(sec_m.group(1)):
            autoloads[entry_m.group(1)] = entry_m.group(2)

    return ver, autoloads


# ── Main builder ───────────────────────────────────────────────────────────────

def build_project_index(project_path: str | Path) -> dict:
    """Scan project_path and return the full index dict (not written to disk)."""
    root = Path(project_path).resolve()
    t0 = time.perf_counter()

    # ── 1. project.godot ──────────────────────────────────────────────────────
    godot_version, autoloads = _parse_project_godot(root)

    # ── 2. Scan .gd files ─────────────────────────────────────────────────────
    scripts: dict[str, dict] = {}
    class_names: dict[str, str] = {}
    signal_signatures: dict[str, dict[str, list[str]]] = {}

    for gd_path in sorted(root.rglob("*.gd")):
        res = "res://" + gd_path.relative_to(root).as_posix()
        text = _read(gd_path)

        entry = _parse_gd(res, text)
        scripts[res] = entry

        # Populate class_names (all files, including addons)
        if entry["class_name"]:
            class_names[entry["class_name"]] = res

        # Signal signatures (only when params exist)
        sigs = _extract_signal_signatures(res, text)
        if sigs:
            signal_signatures[res] = sigs

    # ── 3. Scan .tscn files ───────────────────────────────────────────────────
    # Pass the already-built scripts dict so TSCN parser can resolve root types.
    scenes: dict[str, dict] = {}

    for tscn_path in sorted(root.rglob("*.tscn")):
        res = "res://" + tscn_path.relative_to(root).as_posix()
        text = _read(tscn_path)
        scenes[res] = _parse_tscn(res, text, scripts)

    # ── 4. Full file listing (for file_not_exists verification) ───────────────
    files: list[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue  # skip .godot/, .git/, etc.
        files.append("res://" + rel.as_posix())

    elapsed = round(time.perf_counter() - t0, 3)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "godot_version": godot_version,
        "build_seconds": elapsed,
        "scripts": scripts,
        "scenes": scenes,
        "autoloads": autoloads,
        "class_names": class_names,
        "signal_signatures": signal_signatures,
        "files": files,
    }


def write_index(index: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(index, indent=2), encoding="utf-8")


# ── CLI entry-point ────────────────────────────────────────────────────────────

def _default_output() -> Path:
    """Default output: <agent-root>/.kimi/project_index.json."""
    return Path(__file__).parent.parent / ".kimi" / "project_index.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a static index of a Godot project for Propose-Verify review."
    )
    parser.add_argument(
        "--project",
        required=True,
        metavar="PATH",
        help="Path to the Godot project root (contains project.godot).",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="Where to write project_index.json (default: .kimi/project_index.json).",
    )
    args = parser.parse_args(argv)

    project_path = Path(args.project).resolve()
    if not project_path.is_dir():
        print(f"ERROR: project path does not exist or is not a directory: {project_path}",
              file=sys.stderr)
        return 1

    output_path = Path(args.output).resolve() if args.output else _default_output()

    print(f"Scanning: {project_path}")
    index = build_project_index(project_path)
    write_index(index, output_path)

    n_scripts = len(index["scripts"])
    n_scenes = len(index["scenes"])
    n_classes = len(index["class_names"])
    elapsed = index["build_seconds"]

    print(
        f"Done in {elapsed}s — "
        f"{n_scripts} scripts, {n_scenes} scenes, {n_classes} class_names"
    )
    print(f"Index written to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
