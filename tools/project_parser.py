"""Shared Godot project metadata utilities.

Parses project.godot to extract structural information (autoload singletons, etc.)
that multiple agents need — avoids duplicated parsing logic across agents.
"""

from __future__ import annotations

import re
from pathlib import Path

# Godot 4 built-in global singletons — always available without any import or preload.
GODOT_BUILTIN_GLOBALS: list[str] = [
    "Input",
    "Engine",
    "OS",
    "ProjectSettings",
    "Performance",
    "PhysicsServer2D",
    "PhysicsServer3D",
    "RenderingServer",
    "DisplayServer",
    "NavigationServer2D",
    "NavigationServer3D",
    "AudioServer",
    "ResourceLoader",
    "ResourceSaver",
    "Time",
    "ClassDB",
    "Marshalls",
    "TranslationServer",
]

_AUTOLOAD_SECTION_RE = re.compile(r"\[autoload\]([^[]*)")


def get_autoload_names(project_path: str) -> list[str]:
    """Return all autoload singleton names registered in project.godot."""
    import logging
    _log = logging.getLogger(__name__)

    project_godot = Path(project_path) / "project.godot"
    if not project_godot.exists():
        _log.debug("[project_parser] project.godot not found at %s", project_godot)
        return []
    text = project_godot.read_text(encoding="utf-8", errors="replace")
    section_match = _AUTOLOAD_SECTION_RE.search(text)
    if not section_match:
        _log.debug("[project_parser] no [autoload] section found in %s", project_godot)
        return []
    names = [
        line.split("=")[0].strip()
        for line in section_match.group(1).splitlines()
        if "=" in line and not line.lstrip().startswith(";")
        and line.split("=")[0].strip()
    ]
    _log.debug("[project_parser] parsed autoload_names from %s: %s", project_godot, names)
    return names


def get_known_globals(project_path: str) -> dict[str, list[str]]:
    """Return all globally available names for a Godot project.

    Returns a dict with keys:
      "autoloads" — names from the [autoload] section of project.godot
      "builtins"  — standard Godot 4 built-in singleton names
      "all"       — combined deduplicated list (autoloads first)
    """
    autoloads = get_autoload_names(project_path)
    combined = list(dict.fromkeys(autoloads + GODOT_BUILTIN_GLOBALS))
    return {"autoloads": autoloads, "builtins": GODOT_BUILTIN_GLOBALS, "all": combined}
