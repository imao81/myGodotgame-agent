"""Subprocess wrapper for the Godot CLI."""

import shutil
import subprocess
import os
from pathlib import Path

_GODOT_FALLBACK = (
    r"C:\Users\windows\AppData\Local\Microsoft\WinGet\Packages"
    r"\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\Godot_v4.6.1-stable_win64_console.exe"
)

# Resolution order: GODOT_EXECUTABLE env var → godot on PATH (CI) → local fallback
GODOT_EXECUTABLE = os.environ.get("GODOT_EXECUTABLE") or shutil.which("godot") or _GODOT_FALLBACK


def run_import(project_path: str, timeout: int = 60) -> dict:
    """Run godot --headless --import to validate all assets.

    Returns {"success": bool, "output": str, "errors": list[str]}
    """
    result = _run([
        GODOT_EXECUTABLE,
        "--headless",
        "--import",
        "--path", str(Path(project_path).resolve()),
    ], timeout=timeout)
    errors = [
        line for line in result["output"].splitlines()
        if any(kw in line.lower() for kw in ("error", "failed", "missing", "cannot"))
    ]
    return {**result, "errors": errors}


def run_script(project_path: str, script_path: str,
               extra_args: list[str] | None = None,
               timeout: int = 30) -> dict:
    """Run godot --headless --check-only --script <script> to syntax-check a GDScript file.

    Returns {"success": bool, "output": str, "errors": list[str]}
    """
    cmd = [
        GODOT_EXECUTABLE,
        "--headless",
        "--check-only",
        "--script", script_path,
        "--path", str(Path(project_path).resolve()),
    ]
    if extra_args:
        cmd += ["--"] + extra_args
    result = _run(cmd, timeout=timeout)
    errors = [
        line for line in result["output"].splitlines()
        if any(kw in line.lower() for kw in ("error:", "parse error", "unexpected"))
    ]
    return {**result, "errors": errors}


def run_test_controller(project_path: str, test_script: str, timeout: int = 60) -> dict:
    """Run the injected test_controller.gd via SceneTree script.

    Returns {"success": bool, "output": str, "json_results": list[dict]}
    """
    import json

    result = _run([
        GODOT_EXECUTABLE,
        "--headless",
        "--path", str(Path(project_path).resolve()),
        "--script", test_script,
        "--",
        "--test",
    ], timeout=timeout)

    json_results = []
    coverage: list[str] = []
    signals: dict[str, int] = {}
    for line in result["output"].splitlines():
        line = line.strip()
        if line.startswith("TEST_RESULT:"):
            try:
                json_results.append(json.loads(line[len("TEST_RESULT:"):]))
            except json.JSONDecodeError:
                pass
        elif line.startswith("COVERAGE:"):
            try:
                parsed = json.loads(line[len("COVERAGE:"):])
                if isinstance(parsed, list):
                    coverage = [str(p) for p in parsed]
            except json.JSONDecodeError:
                pass
        elif line.startswith("SIGNALS:"):
            try:
                parsed = json.loads(line[len("SIGNALS:"):])
                if isinstance(parsed, dict):
                    signals = {str(k): int(v) for k, v in parsed.items()}
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

    return {**result, "json_results": json_results,
            "coverage": coverage, "signals": signals}


def _run(cmd: list[str], timeout: int = 60) -> dict:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        output = proc.stdout + proc.stderr
        return {"success": proc.returncode == 0, "output": output, "returncode": proc.returncode}
    except subprocess.TimeoutExpired:
        return {"success": False, "output": f"Timed out after {timeout}s", "returncode": -1}
    except FileNotFoundError:
        return {
            "success": False,
            "output": f"Godot executable not found: {cmd[0]}",
            "returncode": -1,
        }
    except Exception as e:
        return {"success": False, "output": str(e), "returncode": -1}
