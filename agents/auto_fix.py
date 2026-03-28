"""Auto-Fix Agent.

Reads error-severity issues from state["code_issues"] (or a JSON file) and
automatically corrects each GDScript file using Claude.

Safety model:
  - git stash push (labelled) before any write — stash is NEVER auto-popped;
    run `git stash drop` after you're satisfied, or `git stash pop` to revert.
  - Per-file: writes corrected code to <name>__autofix_tmp.gd, validates with
    godot --check-only, then overwrites the original only if validation passes.

Scope:
  - Error-severity issues only (warnings and infos are left for human review).
  - Issue-pinned prompt: Claude is instructed to fix only the listed lines and
    change nothing else.

Usage (standalone):
  # Dry-run (safe default) — prints fix plan, writes nothing:
  python -m agents.auto_fix --project ../godot-2d-space-game/project

  # Apply fixes (requires --apply flag):
  python -m agents.auto_fix --project ../godot-2d-space-game/project --apply

  # Read issues from a specific report file:
  python -m agents.auto_fix --project ../godot-2d-space-game/project \\
      --issues reports/20260324_120000/code_issues.json --apply
"""

import argparse
import json
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from agents.llm_client import LLMClient
from tools.godot_runner import run_script

_FIX_PROMPT = """\
You are a GDScript bug fixer for Harvester, a Godot 4 top-down space mining game.

Fix ONLY the exact issues listed below. Do NOT refactor, rename variables, add comments,
reformat code, or change anything not directly required to fix each listed issue.
Return the complete corrected file and nothing else — no explanations, no markdown fences.

File: {filename}

Issues to fix (error severity only):
{issue_list}

--- BEGIN FILE ---
{code}
--- END FILE ---
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _git_stash(project_path: str, label: str) -> bool:
    """Push a labelled git stash. Returns True on success."""
    try:
        result = subprocess.run(
            ["git", "stash", "push", "-m", label],
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except Exception:
        return False


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences if Claude adds them; return text unchanged otherwise."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    # Has fences — strip them and return the inner content
    stripped = re.sub(r"^```[a-zA-Z]*\n?", "", stripped)
    stripped = re.sub(r"\n?```\s*$", "", stripped)
    return stripped


def _validate_gd_file(project_path: str, res_path: str) -> bool:
    """Run godot --check-only on a res:// path. Returns True if syntax is valid."""
    result = run_script(project_path, script_path=res_path, timeout=20)
    return result["success"]


# ── Core run function ─────────────────────────────────────────────────────────

def run(state: dict, apply: bool = False) -> dict:
    """Auto-fix error-severity GDScript issues.

    Args:
        state:  Pipeline state dict — must contain "project_path" and "code_issues".
        apply:  False (default) = dry-run, print plan only.
                True            = write fixes (git stash first).

    Returns dict with keys:
        fix_results:  list[dict]  — per-file outcome records
        fix_summary:  str         — human-readable summary
        agent_status: dict        — {"auto_fix": "ok"|"ok_with_failures"|...}
    """
    project_path = state["project_path"]
    all_issues: list[dict] = state.get("code_issues", [])

    # Scope: errors only
    error_issues = [i for i in all_issues if i.get("severity") == "error"]

    if not error_issues:
        return {
            "fix_results": [],
            "fix_summary": "No error-severity issues to fix.",
            "agent_status": {"auto_fix": "nothing_to_fix"},
        }

    # Group by file
    by_file: dict[str, list[dict]] = defaultdict(list)
    for issue in error_issues:
        if issue.get("file"):
            by_file[issue["file"]].append(issue)

    if not by_file:
        return {
            "fix_results": [],
            "fix_summary": "All error issues are missing file references — cannot fix.",
            "agent_status": {"auto_fix": "nothing_to_fix"},
        }

    if apply:
        label = f"auto_fix {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        if not _git_stash(project_path, label):
            return {
                "fix_results": [],
                "fix_summary": (
                    "git stash failed — aborting to avoid data loss. "
                    "Make sure this is a git repository."
                ),
                "agent_status": {"auto_fix": "stash_failed"},
            }

    client = LLMClient() if apply else None
    root = Path(project_path)
    fix_results: list[dict] = []

    for rel_path, issues in by_file.items():
        abs_path = root / rel_path

        if not abs_path.exists():
            fix_results.append({
                "file": rel_path,
                "status": "skipped",
                "reason": "file not found on disk",
            })
            continue

        original_code = abs_path.read_text(encoding="utf-8", errors="replace")
        issue_list = "\n".join(
            f"  - line {i.get('line', '?')}: [{i.get('rule', '')}] {i.get('message', '')}"
            for i in issues
        )

        if not apply:
            fix_results.append({
                "file": rel_path,
                "status": "would_fix",
                "issues": len(issues),
                "issue_list": [
                    f"line {i.get('line', '?')}: {i.get('message', '')}"
                    for i in issues
                ],
            })
            continue

        # ── Call LLM ───────────────────────────────────────────────────────
        try:
            fixed_code = _strip_code_fences(client.chat(
                [{
                    "role": "user",
                    "content": _FIX_PROMPT.format(
                        filename=rel_path,
                        issue_list=issue_list,
                        code=original_code,
                    ),
                }],
                max_tokens=4096,
            ))
        except Exception as e:
            fix_results.append({
                "file": rel_path,
                "status": "failed",
                "reason": f"LLM API error: {e}",
            })
            continue

        # ── Validate via temp file ─────────────────────────────────────────
        tmp_path = abs_path.parent / (abs_path.stem + "__autofix_tmp.gd")
        res_tmp = "res://" + str(tmp_path.relative_to(root)).replace("\\", "/")

        try:
            tmp_path.write_text(fixed_code, encoding="utf-8")
            valid = _validate_gd_file(project_path, res_tmp)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        if valid:
            abs_path.write_text(fixed_code, encoding="utf-8")
            fix_results.append({
                "file": rel_path,
                "status": "fixed",
                "issues_fixed": len(issues),
            })
        else:
            fix_results.append({
                "file": rel_path,
                "status": "skipped",
                "reason": "godot --check-only failed after fix — original preserved",
            })

    # ── Build summary ──────────────────────────────────────────────────────
    if not apply:
        total_issues = sum(r["issues"] for r in fix_results if r["status"] == "would_fix")
        summary = (
            f"Dry-run: {len(fix_results)} file(s) with {total_issues} error(s) "
            f"would be fixed. Pass --apply to write."
        )
    else:
        fixed   = sum(1 for r in fix_results if r["status"] == "fixed")
        skipped = sum(1 for r in fix_results if r["status"] == "skipped")
        failed  = sum(1 for r in fix_results if r["status"] == "failed")
        summary = (
            f"{fixed} file(s) fixed, {skipped} skipped (validation failed), "
            f"{failed} failed (API/IO error). "
            f"git stash kept — run `git stash drop` when satisfied."
        )

    had_failures = any(r["status"] in ("failed",) for r in fix_results)
    return {
        "fix_results": fix_results,
        "fix_summary": summary,
        "agent_status": {"auto_fix": "ok_with_failures" if had_failures else "ok"},
    }


# ── Standalone CLI ────────────────────────────────────────────────────────────

def _load_issues(issues_path: str | None, project_path: str) -> list[dict]:
    if issues_path:
        return json.loads(Path(issues_path).read_text(encoding="utf-8"))
    # Try the most recent report
    reports_dir = Path(project_path).parent.parent / "reports"  # harvester-agent/reports
    if not reports_dir.exists():
        # Fall back to same directory as script
        reports_dir = Path(__file__).parent.parent / "reports"
    candidates = sorted(reports_dir.glob("*/code_issues.json"), reverse=True)
    if candidates:
        print(f"[auto_fix] Using issues from: {candidates[0]}")
        return json.loads(candidates[0].read_text(encoding="utf-8"))
    return []


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Auto-Fix Agent — correct GDScript errors")
    parser.add_argument("--project", required=True, help="Path to Godot project directory")
    parser.add_argument("--issues", default=None, help="Path to code_issues.json (optional)")
    parser.add_argument("--apply", action="store_true", help="Write fixes (default: dry-run)")
    args = parser.parse_args()

    from rich.console import Console
    from rich.panel import Panel
    console = Console()

    project_path = str(Path(args.project).resolve())
    issues = _load_issues(args.issues, project_path)

    if not issues:
        console.print("[yellow]No code issues found. Run the pipeline first to generate code_issues.json[/yellow]")
        raise SystemExit(0)

    console.print(Panel(
        f"[bold]Auto-Fix Agent[/bold]\n"
        f"Project: {project_path}\n"
        f"Mode:    {'APPLY (writes files)' if args.apply else 'DRY-RUN (no writes)'}",
        title="🔧 Auto-Fix",
    ))

    state = {"project_path": project_path, "code_issues": issues}
    result = run(state, apply=args.apply)

    console.print(f"\n[bold]{result['fix_summary']}[/bold]\n")

    for r in result["fix_results"]:
        status = r["status"]
        color = {"fixed": "green", "would_fix": "cyan", "skipped": "yellow", "failed": "red"}.get(status, "white")
        label = f"[{color}]{status.upper()}[/{color}]"
        console.print(f"  {label}  {r['file']}")
        if status == "would_fix":
            for issue_str in r.get("issue_list", []):
                console.print(f"         {issue_str}")
        elif status in ("skipped", "failed"):
            console.print(f"         {r.get('reason', '')}")
