"""Project-wide symbol index for Godot projects.

Scans the entire project once and builds a JSON index containing:
  - class_name declarations (class → file)
  - preload/load references (file → list of loaded paths)
  - autoload singletons (from project.godot)
  - scene node trees (.tscn → {node_name: node_type})
  - script-to-scene mapping (.gd → scene file)
  - signal declarations and connections

The index is cached as project_index.json and only rebuilt when
any .gd or .tscn file has a newer mtime than the cache.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from tools.project_parser import get_known_globals

# ── Regex patterns ─────────────────────────────────────────────────────────────

_CLASS_NAME_RE = re.compile(r"^\s*class_name\s+(\w+)", re.MULTILINE)
_PRELOAD_RE = re.compile(r'(?:preload|load)\s*\(\s*"(res://[^"]+)"')
_SIGNAL_DECL_RE = re.compile(r"^\s*signal\s+(\w+)", re.MULTILINE)
_SIGNAL_EMIT_RE = re.compile(r"(\w+)\.emit\(|emit_signal\([\"'](\w+)[\"']")
_SIGNAL_CONNECT_RE = re.compile(r'(\w+)\s*\.\s*connect\s*\(')

# .tscn node line: [node name="Foo" type="Bar" parent="..."]
_TSCN_NODE_RE = re.compile(r'\[node\s+name="([^"]+)"(?:\s+type="([^"]+)")?')
# .tscn ext_resource: [ext_resource type="Script" path="res://..." id="..."]
_TSCN_EXT_RES_RE = re.compile(
    r'\[ext_resource\s+type="Script"\s+path="(res://[^"]+)"\s+id="([^"]+)"'
)
# script = ExtResource("id") inside a node block
_TSCN_SCRIPT_REF_RE = re.compile(r'^\s*script\s*=\s*ExtResource\("([^"]+)"\)', re.MULTILINE)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


# ── Indexing helpers ────────────────────────────────────────────────────────────

def _index_gd_files(root: Path) -> tuple[dict, dict, dict, dict]:
    """Return (class_names, preloads, signal_decls, signal_emits) from .gd files."""
    class_names: dict[str, str] = {}   # ClassName -> res:// path
    preloads: dict[str, list[str]] = {}  # res:// path -> [res:// paths loaded]
    signal_decls: dict[str, list[str]] = {}  # res:// path -> [signal names]
    signal_emits: dict[str, list[str]] = {}  # res:// path -> [signal names emitted]

    for gd in root.rglob("*.gd"):
        if "libraries" in gd.parts:
            continue
        rel = "res://" + gd.relative_to(root).as_posix()
        text = _read(gd)

        # class_name
        for m in _CLASS_NAME_RE.finditer(text):
            class_names[m.group(1)] = rel

        # preload/load
        loaded = _PRELOAD_RE.findall(text)
        if loaded:
            preloads[rel] = loaded

        # signal declarations
        decls = _SIGNAL_DECL_RE.findall(text)
        if decls:
            signal_decls[rel] = decls

        # signal emits
        emitted: list[str] = []
        for m in _SIGNAL_EMIT_RE.finditer(text):
            name = m.group(1) or m.group(2)
            if name:
                emitted.append(name)
        if emitted:
            signal_emits[rel] = emitted

    return class_names, preloads, signal_decls, signal_emits


def _index_tscn_files(root: Path) -> tuple[dict, dict]:
    """Return (scene_nodes, script_to_scene) from .tscn files.

    scene_nodes: scene_res_path -> {node_name: node_type}
    script_to_scene: script_res_path -> scene_res_path
    """
    scene_nodes: dict[str, dict[str, str]] = {}
    script_to_scene: dict[str, str] = {}

    for tscn in root.rglob("*.tscn"):
        text = _read(tscn)
        scene_res = "res://" + tscn.relative_to(root).as_posix()

        # Build ext_resource id -> script path map for this scene
        ext_res_map: dict[str, str] = {}
        for m in _TSCN_EXT_RES_RE.finditer(text):
            script_path, res_id = m.group(1), m.group(2)
            ext_res_map[res_id] = script_path

        # Parse node tree
        nodes: dict[str, str] = {}
        for m in _TSCN_NODE_RE.finditer(text):
            node_name = m.group(1)
            node_type = m.group(2) or ""  # may be absent for inherited nodes
            if node_type:
                nodes[node_name] = node_type
        if nodes:
            scene_nodes[scene_res] = nodes

        # Find which script is attached to the root node (first node block)
        # by looking for script = ExtResource("id") near a node declaration
        # We do a simple linear scan: find each node block's script ref
        # Split on [node  boundaries to process per-node
        node_blocks = re.split(r'(?=\[node\s)', text)
        for block in node_blocks:
            script_match = _TSCN_SCRIPT_REF_RE.search(block)
            if script_match:
                res_id = script_match.group(1)
                if res_id in ext_res_map:
                    script_path = ext_res_map[res_id]
                    script_to_scene[script_path] = scene_res

    return scene_nodes, script_to_scene


def _build_reverse_deps(preloads: dict[str, list[str]]) -> dict[str, list[str]]:
    """Build file -> [files that depend on it] from preloads map."""
    reverse: dict[str, list[str]] = {}
    for loader, loaded_list in preloads.items():
        for loaded in loaded_list:
            reverse.setdefault(loaded, []).append(loader)
    return reverse


def _get_max_mtime(root: Path) -> float:
    """Return max mtime of all .gd and .tscn files (for cache invalidation)."""
    mt = 0.0
    for ext in ("*.gd", "*.tscn"):
        for f in root.rglob(ext):
            if "libraries" not in f.parts:
                mt = max(mt, f.stat().st_mtime)
    return mt


# ── Public API ─────────────────────────────────────────────────────────────────

def build_index(project_path: str, cache_path: str | None = None) -> dict:
    """Build (or load cached) project symbol index.

    Args:
        project_path: Absolute path to the Godot project root.
        cache_path: Where to store project_index.json.
                    Defaults to <project_path>/../project_index.json
                    (i.e., next to the project folder, in the agent working dir).

    Returns:
        dict with keys: class_names, preloads, autoloads, scene_nodes,
                        script_to_scene, signal_decls, signal_emits,
                        reverse_deps, built_at
    """
    root = Path(project_path)
    if cache_path is None:
        # Store alongside the pipeline working dir (parent of project)
        cache_file = Path(__file__).parent.parent / "project_index.json"
    else:
        cache_file = Path(cache_path)

    max_mtime = _get_max_mtime(root)

    # Try loading cache
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached.get("source_mtime", 0) >= max_mtime:
                return cached
        except (json.JSONDecodeError, KeyError):
            pass  # Rebuild

    t0 = time.perf_counter()

    known = get_known_globals(project_path)
    class_names, preloads, signal_decls, signal_emits = _index_gd_files(root)
    scene_nodes, script_to_scene = _index_tscn_files(root)
    reverse_deps = _build_reverse_deps(preloads)

    index = {
        "class_names": class_names,         # {"MathUtils": "res://utils/math_utils.gd"}
        "preloads": preloads,               # {"res://foo.gd": ["res://bar.gd"]}
        "autoloads": known["autoloads"],    # ["Events", "ObjectRegistry"]
        "scene_nodes": scene_nodes,         # {"res://pirate.tscn": {"CollisionShape": "CollisionPolygon2D"}}
        "script_to_scene": script_to_scene, # {"res://pirate_ship.gd": "res://pirate.tscn"}
        "signal_decls": signal_decls,       # {"res://foo.gd": ["squad_leader_changed"]}
        "signal_emits": signal_emits,       # {"res://foo.gd": ["destroyed"]}
        "reverse_deps": reverse_deps,       # {"res://foo.gd": ["res://bar.gd"]}
        "source_mtime": max_mtime,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "build_seconds": round(time.perf_counter() - t0, 2),
    }

    cache_file.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index
