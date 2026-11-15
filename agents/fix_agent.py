"""Fix agent: bounded feedback loop for verified findings.

The model gets limited loop sovereignty — at most `max_rounds` iterations —
and every iteration is constrained by deterministic acceptance gates (syntax,
assertion re-verification, playtest regression) inside an isolated sandbox.
Gate output feeds back into the next round; exhaustion is reported honestly.

Unlike the legacy whole-file rewrite, the model must output surgical edit
blocks (tools/patch_format). A whole-file response is accepted only when its
diff ratio stays below the boundary threshold (no drive-by refactoring).
"""

from __future__ import annotations

from pathlib import Path

from tools.patch_format import apply_blocks, boundary_report, parse_blocks
from tools.sandbox import Sandbox, SandboxError

_FIX_PROMPT = """\
You are fixing verified bugs in a Godot 4 GDScript file.

Output ONLY edit blocks in EXACTLY this format — no explanations, no fences:

<<<<<<< ORIGINAL
<exact text copied from the file>
=======
<corrected replacement text>
>>>>>>> FIXED

Rules:
- ORIGINAL must match the file content EXACTLY and occur EXACTLY ONCE.
- Change ONLY what the issues require: no refactoring, no renaming, no
  reformatting, no comments.
- Multiple issues → multiple blocks.

File: {filename}

Verified issues to fix:
{issue_list}

--- BEGIN FILE ---
{code}
--- END FILE ---
{feedback}"""

_FEEDBACK_TEMPLATE = """
Your previous attempt FAILED the acceptance gates:
{errors}
Fix these problems and try again. Output edit blocks only."""


def _issue_list(issues: list[dict]) -> str:
    return "\n".join(
        f"  - line {i.get('line', '?')}: [{i.get('rule', '')}] {i.get('message', '')}"
        for i in issues
    )


def _target_assertions(issues: list[dict]) -> list[dict]:
    return [i["assertion"] for i in issues if isinstance(i.get("assertion"), dict)]


def fix_file(rel_path: str, issues: list[dict], project_root: str | Path,
             *, client, max_rounds: int = 3) -> dict:
    """Fix one file's verified issues. Returns a JSON-friendly outcome dict.

    Never raises for LLM/gate failures — exhaustion is a reported outcome.
    SandboxError (environment-level) is also caught: no fix is possible.
    """
    outcome = {
        "file": rel_path,
        "status": "failed",
        "rounds_used": 0,
        "gate_report": {},
        "boundary": {},
        "fixed_code": None,
        "error": None,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    }

    try:
        sb = Sandbox(project_root)
        sb.create()
    except SandboxError as e:
        outcome["error"] = f"sandbox unavailable: {e}"
        return outcome

    sandbox_file = Path(sb.path) / rel_path
    if not sandbox_file.exists():
        sb.discard()
        outcome["error"] = "file not found in sandbox"
        return outcome

    original = sandbox_file.read_text(encoding="utf-8", errors="replace")
    assertions = _target_assertions(issues)
    feedback = ""

    # --check-only --script cannot resolve class_name globals, so the pristine
    # file may already show errors. Snapshot them as the baseline; the syntax
    # gate then only fails on NEW errors introduced by a fix attempt.
    from tools import godot_runner
    baseline_result = godot_runner.run_script(
        str(sb.path), "res://" + rel_path.replace("\\", "/"))
    syntax_baseline = {rel_path: set(baseline_result.get("errors") or [])}

    try:
        for round_no in range(1, max_rounds + 1):
            outcome["rounds_used"] = round_no
            if round_no > 1:
                # Each round re-fixes from the pristine original — a failed
                # attempt must not become the next round's starting state.
                sandbox_file.write_text(original, encoding="utf-8")
            prompt = _FIX_PROMPT.format(
                filename=rel_path,
                issue_list=_issue_list(issues),
                code=sandbox_file.read_text(encoding="utf-8", errors="replace"),
                feedback=feedback,
            )
            try:
                llm_out = client.chat([{"role": "user", "content": prompt}],
                                      max_tokens=4096)
            except Exception as e:
                outcome["error"] = f"LLM API error: {e}"
                return outcome
            usage = getattr(client, "last_usage", None) or {}
            for k in ("prompt_tokens", "completion_tokens"):
                outcome["usage"][k] += usage.get(k, 0)

            parsed = parse_blocks(llm_out)
            current = sandbox_file.read_text(encoding="utf-8", errors="replace")
            boundary = boundary_report(current, llm_out, parsed.blocks,
                                       parsed.leftover)
            outcome["boundary"] = boundary

            if parsed.blocks:
                applied = apply_blocks(current, parsed.blocks)
                if applied.new_text is None:
                    feedback = _FEEDBACK_TEMPLATE.format(
                        errors="\n".join(applied.errors))
                    continue
                sandbox_file.write_text(applied.new_text, encoding="utf-8")
            else:
                if not boundary["boundary_ok"]:
                    feedback = _FEEDBACK_TEMPLATE.format(
                        errors=(f"whole-file rewrite rejected: diff ratio "
                                f"{boundary['diff_ratio']} exceeds limit. "
                                f"Use edit blocks."))
                    continue
                sandbox_file.write_text(parsed.leftover, encoding="utf-8")

            report = sb.run_gates(target_assertions=assertions,
                                  changed_files=[rel_path],
                                  syntax_baseline=syntax_baseline)
            outcome["gate_report"] = {
                "syntax_ok": report.syntax_ok,
                "assertions_resolved": report.assertions_resolved,
                "regression_ok": report.regression_ok,
                "ok": report.ok,
                "details": report.details,
            }
            if report.ok and boundary["boundary_ok"]:
                outcome["status"] = "fixed"
                outcome["fixed_code"] = sandbox_file.read_text(
                    encoding="utf-8", errors="replace")
                sb.promote()
                return outcome

            feedback = _FEEDBACK_TEMPLATE.format(
                errors="\n".join(report.details[-4:]))

        outcome["error"] = f"exhausted {max_rounds} rounds without passing gates"
        return outcome
    finally:
        if outcome["status"] != "fixed":
            sb.discard()
        # fixed: keep the sandbox dir until caller merges/discards via
        # Sandbox.discard_at(project_root, sb.path)
        if outcome["status"] == "fixed":
            outcome["sandbox_path"] = str(sb.path)
