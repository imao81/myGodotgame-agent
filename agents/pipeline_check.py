"""Pipeline Check Agent.

Validates the Godot project without any LLM calls:
  1. godot --headless --import  (all assets parse + import correctly)
  2. godot --headless --check-only --script  (per-file GDScript syntax)
  3. res:// path validation  (all resource references exist on disk)
"""

import logging
import re
from pathlib import Path
from tools.godot_runner import run_import, run_script
from tools.project_parser import get_known_globals

_log = logging.getLogger(__name__)

_RES_RE = re.compile(r'"(res://[^"]+)"')


def _issue(type_: str, path: str, message: str) -> dict:
    return {"type": type_, "path": path, "message": message, "check_method": "static"}


def _check_import(project_path: str) -> tuple[list[dict], bool]:
    """Run headless import; return (issues, success)."""
    result = run_import(project_path, timeout=120)
    issues = [_issue("import", "", err) for err in result["errors"]]
    return issues, result["success"]


_IDENT_NOT_FOUND = "Identifier not found: "

# Cascade error fragments: these lines are emitted by Godot after a root compile
# failure and carry no additional diagnostic value — always suppress them.
_CASCADE_FRAGMENTS = (
    "Failed to load script",
    "Failed to compile depended scripts",
)


def _is_autoload_false_positive(err: str, autoload_names: set) -> bool:
    """Return True when the error is a known-false-positive autoload reference.

    Godot emits "Identifier not found: <Name>" when --check-only --script
    runs a file in isolation (project.godot autoloads are invisible).
    Uses substring search so "SCRIPT ERROR: Compile Error: Identifier not found: X"
    is matched correctly regardless of the leading prefix.
    """
    idx = err.find(_IDENT_NOT_FOUND)
    if idx == -1:
        return False
    rest = err[idx + len(_IDENT_NOT_FOUND):]
    ident = rest.split()[0] if rest.split() else ""
    return ident in autoload_names


def _is_cascade_error(err: str) -> bool:
    """Return True for Godot cascade/summary lines that follow a root compile error.

    These lines ("Failed to load script ... Compilation failed",
    "Failed to compile depended scripts") are always secondary — the root error
    is reported separately and is the actionable item.  Suppressing these prevents
    a single autoload issue from producing multiple duplicate findings per file.
    """
    return any(fragment in err for fragment in _CASCADE_FRAGMENTS)


def _check_syntax(
    game_files: list, root: Path, project_path: str, autoload_names: set
) -> list[dict]:
    """Run per-file GDScript syntax check; suppress autoload false positives and cascades.

    A file that ONLY fails due to autoload references produces zero findings.
    A file with genuine syntax errors reports those errors; autoload and cascade
    lines are still filtered out so they don't inflate the count.
    """
    issues: list[dict] = []
    for gd in game_files:
        rel = str(gd.relative_to(root))
        result = run_script(project_path, f"res://{rel.replace(chr(92), '/')}", timeout=15)
        for err in result["errors"]:
            if _is_autoload_false_positive(err, autoload_names):
                _log.debug("[pipeline_check] suppressed autoload FP in %s: %s", rel, err)
                continue
            if _is_cascade_error(err):
                _log.debug("[pipeline_check] suppressed cascade error in %s: %s", rel, err)
                continue
            issues.append(_issue("syntax", rel, err))
    return issues


def _check_resources(game_files: list, root: Path, autoload_names: set) -> list[dict]:
    """Validate all res:// references exist on disk; skip known autoload script paths."""
    issues: list[dict] = []
    for gd in game_files:
        text = gd.read_text(encoding="utf-8", errors="replace")
        for match in _RES_RE.finditer(text):
            res_path = match.group(1)
            abs_path = root / res_path[len("res://"):]
            if abs_path.exists() or abs_path.stem in autoload_names:
                continue
            issues.append(_issue("missing_resource", str(gd.relative_to(root)),
                                 f"Referenced resource not found: {res_path}"))
    return issues


def run(state: dict) -> dict:
    project_path = state["project_path"]
    known = get_known_globals(project_path)
    autoload_names = set(known["autoloads"])
    _log.debug("[pipeline_check] autoload_names=%r (project_path=%s)", autoload_names, project_path)

    import_issues, import_ok = _check_import(project_path)

    root = Path(project_path)
    all_gd = [f for f in sorted(root.rglob("*.gd")) if "libraries" not in f.parts]
    scope = set(state.get("scope", []))
    game_files = [f for f in all_gd if not scope or str(f.relative_to(root)) in scope]

    syntax_issues = _check_syntax(game_files, root, project_path, autoload_names)
    resource_issues = _check_resources(game_files, root, autoload_names)

    return {
        "pipeline_issues": import_issues + syntax_issues + resource_issues,
        "agent_status": {"pipeline_check": "ok" if import_ok else "failed"},
    }
