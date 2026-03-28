"""Automated Playtest Agent.

Layer 1 — Logic testing (headless):
  Copies test_controller.gd into project/autoload/, runs Godot headlessly,
  parses TEST_RESULT: JSON lines from stdout, then removes the injected file.

Layer 2 — Visual testing (requires display):
  Runs the game windowed with --visual-test, collects screenshots saved by
  the test controller, then calls Claude Vision on each one.
  Skipped if --no-display is in state flags.
"""

import shutil
import os
from pathlib import Path

from agents.llm_client import LLMClient
from tools.godot_runner import run_test_controller
from tools.vision_analyzer import analyze_all

# Path to the test controller source (relative to this file)
_SCRIPT_DIR = Path(__file__).parent.parent
TEST_CONTROLLER_SRC = _SCRIPT_DIR / "godot_test_scripts" / "test_controller.gd"


def run(state: dict) -> dict:
    project_path = state["project_path"]
    no_display = state.get("no_display", False)
    pipeline_ok = state.get("agent_status", {}).get("pipeline_check") != "failed"

    results: list[dict] = []
    screenshots: list[str] = []
    focus_files: list[str] = []
    balance_issues: list[str] = []

    # If pipeline check failed hard, skip playtest to avoid misleading results
    if not pipeline_ok:
        return {
            "playtest_results": [{"scenario": "all", "passed": False,
                                   "error": "Skipped: pipeline check failed"}],
            "screenshots": [],
            "focus_files": [],
            "balance_issues": [],
            "agent_status": {"playtest": "skipped"},
        }

    injected_path = Path(project_path) / "autoload" / "test_controller.gd"

    try:
        # ── Inject test controller ─────────────────────────────────────────
        shutil.copy2(TEST_CONTROLLER_SRC, injected_path)

        # ── Layer 1: headless logic tests ──────────────────────────────────
        godot_result = run_test_controller(
            project_path,
            test_script="res://autoload/test_controller.gd",
            timeout=90,
        )

        for r in godot_result["json_results"]:
            results.append(r)
            if not r.get("passed", True):
                # Collect files involved in failures for human follow-up
                if r.get("file"):
                    focus_files.append(r["file"])
                if r.get("balance_issue"):
                    balance_issues.append(r["balance_issue"])

        # If Godot crashed with no results, record that
        if not godot_result["json_results"] and not godot_result["success"]:
            results.append({
                "scenario": "startup",
                "passed": False,
                "error": godot_result["output"][-500:],  # last 500 chars
            })

        # ── Layer 2: visual tests (skipped in no-display mode) ─────────────
        if not no_display:
            screenshot_dir = Path(project_path) / "test_output"
            screenshot_dir.mkdir(exist_ok=True)

            # Screenshots are saved by test_controller.gd during visual run;
            # here we just collect what was produced by Layer 1 (test_controller
            # saves screenshots even in headless mode for key state transitions)
            pngs = sorted(screenshot_dir.glob("*.png"))
            if pngs:
                client = LLMClient()
                visual_specs = [
                    {"path": str(p), "expected": _expected_state_for(p.name)}
                    for p in pngs
                ]
                vision_results = analyze_all(client, visual_specs)
                for vr in vision_results:
                    screenshots.append(vr["path"])
                    if not vr.get("ok", True):
                        results.append({
                            "scenario": f"visual:{Path(vr['path']).name}",
                            "passed": False,
                            "error": "; ".join(vr.get("issues", [])),
                        })

    finally:
        # ── Always remove injected file ────────────────────────────────────
        if injected_path.exists():
            injected_path.unlink()

    passed = sum(1 for r in results if r.get("passed", False))
    failed = sum(1 for r in results if not r.get("passed", True))
    status = "ok" if failed == 0 else "ok_with_failures"

    return {
        "playtest_results": results,
        "screenshots": screenshots,
        "focus_files": list(set(focus_files)),
        "balance_issues": balance_issues,
        "agent_status": {"playtest": status},
        "playtest_summary": f"{passed} passed, {failed} failed out of {len(results)} scenarios",
    }


def _expected_state_for(filename: str) -> str:
    """Map screenshot filenames to human-readable expected state descriptions."""
    mapping = {
        "startup": "Main menu visible with title and start button",
        "game_world": "Space world with player ship at center, asteroids visible",
        "mining": "Player docked at asteroid, mining in progress, cargo bar filling",
        "hud": "HUD overlay showing shield bar, cargo indicator, and controls hint",
        "upgrade_menu": "Upgrade menu open showing 5 upgrade options as buttons",
        "map": "Minimap overlay showing asteroid positions and player location",
    }
    for key, desc in mapping.items():
        if key in filename:
            return desc
    return "Game running normally without visual artifacts"
