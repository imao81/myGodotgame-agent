"""Code Review Agent.

Reads all game .gd files (excluding third-party libraries) and calls Claude
to review them for GDScript-specific issues: logic bugs, performance, signal
connection patterns, state machine correctness, and resource leaks.
"""

import json
import logging
import re
import time
from pathlib import Path

from agents.llm_client import LLMClient
from tools.project_parser import get_known_globals
from tools.project_indexer import build_index
from tools.context_builder import build_file_context

_log = logging.getLogger(__name__)

# Errors that are worth retrying (transient); auth errors are not.
_RETRIABLE_TYPES = {"timeout", "rate_limit", "overloaded", "api_error"}


def _classify_api_error(exc: Exception) -> str:
    """Map an exception to a machine-readable error type."""
    msg = str(exc).lower()
    # Also check the class name: httpx.ReadTimeout / httpx.ConnectTimeout have
    # an empty or URL-only str(), so the message check alone would miss them.
    class_name = type(exc).__name__.lower()
    if "401" in msg or "authentication" in msg or "invalid x-api-key" in msg:
        return "auth_error"
    if "429" in msg or "rate_limit" in msg:
        return "rate_limit"
    if "timeout" in msg or "timeout" in class_name:
        return "timeout"
    if "overloaded" in msg or "529" in msg:
        return "overloaded"
    return "api_error"


_REVIEW_PROMPT = """\
You are reviewing GDScript code for Harvester, a Godot 4 top-down space mining game.

The game uses:
- A state machine library (State / StateMachine base classes)
- An Events autoload signal bus (events.gd)
- An ObjectRegistry autoload for tracking projectiles/effects
- Steering AI framework (GSAIxxx classes) for enemy movement

IMPORTANT — Godot global singletons (do NOT flag as undefined):
The following names are globally available at runtime in every script WITHOUT any
preload() or class_name declaration. Never flag them as "undefined reference",
"missing import", or similar errors.

Project autoloads (from project.godot [autoload]): {autoload_list}
Godot built-in globals (always present): Input, Engine, OS, ProjectSettings,
  Performance, PhysicsServer2D, RenderingServer, DisplayServer, AudioServer,
  ResourceLoader, ResourceSaver, Time, ClassDB

If a file references one of these names and the reference looks intentional, emit an
info-level note instead of an error: use rule "autoload_ref" and message
"References autoload singleton '<Name>' registered in project.godot" (or "Godot
built-in singleton" for built-ins). Only emit this when it adds useful context;
omit it for files that simply use these singletons correctly.

The project context block (if present above the code) shows cross-file dependencies.
Use it to avoid false positives:
- Do NOT flag class_names listed in "Available class_names" as undefined
- Do NOT flag node type mismatches if the scene context shows the correct type
- Do NOT flag signal issues if the signal is declared in a file listed in the
  dependency chain
- If uncertain whether something is a real issue or a cross-file dependency you
  cannot verify, use severity "warning" not "error"

Severity guide:
 - error: will crash at runtime or cause data loss
 - warning: code smell, potential bug, or best practice violation
 - info: style suggestion or documentation note

If the game is known to run successfully, lean toward "warning" for issues that
MIGHT cause problems but clearly don't crash the game.

GDScript language rules — DO NOT flag these as errors:
- `:=` is inferred static typing. `var x := SomeClass.new()` IS type-hinted via
  inference. Do NOT suggest adding a type annotation when `:=` is already used.
- `@export var x: PackedScene` IS already type-hinted. The type after `:` is the
  annotation. Do NOT flag this as "lacks type hint".
- `@onready var x := $NodePath` IS type-hinted via inference. Do NOT flag this.
- `var x: TypeName` with no `:=` is also type-hinted. Do NOT flag it.
- Node paths like `$CollisionShape2D` reference nodes defined in the attached .tscn
  scene file, NOT in this script. The project context block lists the actual node
  names and types from the scene. Use ONLY the scene context to validate node
  references — do NOT guess node types from the node name string alone.
- If you are unsure whether something is a real error or a GDScript feature you are
  unfamiliar with, classify it as "warning" not "error".

You are reviewing this code independently. You have NO prior information about which
lines may have issues. Do not assume any particular line is more likely to have bugs
than another. Analyze the entire file with equal attention.

Review the following file for:
1. Logic bugs in state machine transitions (missing state cleanup, double-transitions)
2. Performance: per-frame allocations in _process/_physics_process, repeated get_node() calls
3. Signal hygiene: signals connected in _ready (good) vs _process (bad), disconnected on exit
4. Resource leaks: nodes added but never freed, timers not stopped
5. Type safety: missing type hints on @export vars and function signatures that use neither
   `:` annotations nor `:=` inference (flag ONLY when BOTH forms are absent)
6. GDScript style: snake_case, no magic numbers, clear variable names

Note: The project has `gdscript/warnings/unused_signal=false` — do NOT flag unused signals.

File: {filename}

{context_and_code}

Reply with a JSON array of issues (empty array if none):
[
  {{"line": <int or null>, "severity": "error|warning|info", "rule": "<short tag>", "message": "<description>"}}
]
Only output the JSON array, nothing else.
"""


def _log_prompt(log_dir: Path, rel: str, prompt: str) -> None:
    """Write the full LLM prompt to logs/review_prompts/<rel>.txt for diagnosis."""
    try:
        safe_name = rel.replace("\\", "_").replace("/", "_")
        out = log_dir / f"{safe_name}.txt"
        out.write_text(prompt, encoding="utf-8")
    except OSError as e:
        _log.debug("[code_review] could not write prompt log for %s: %s", rel, e)


def _call_with_retry(
    client: LLMClient,
    messages: list,
    max_tokens: int,
    max_retries: int = 3,
) -> str:
    """Call client.chat with exponential-backoff retry for transient errors.

    Raises the last exception if all retries are exhausted or the error is
    non-retriable (e.g. auth_error).
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return client.chat(messages, max_tokens=max_tokens)
        except Exception as e:
            error_type = _classify_api_error(e)
            if error_type not in _RETRIABLE_TYPES:
                raise  # Auth errors etc. — no point retrying
            last_exc = e
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                _log.warning(
                    "[code_review] transient %s on attempt %d/%d, retrying in %ds: %s",
                    error_type, attempt + 1, max_retries, wait, e,
                )
                time.sleep(wait)
    raise last_exc  # type: ignore[misc]


def run(state: dict) -> dict:
    project_path = state["project_path"]
    client = LLMClient()
    root = Path(project_path)
    all_issues: list[dict] = []
    agent_errors: list[dict] = []  # infrastructure errors — kept separate from findings

    # Prompt log directory — created once per run
    log_dir = Path(__file__).parent.parent / "logs" / "review_prompts"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Build known-globals list once — shared across all file reviews
    known_globals = get_known_globals(project_path)
    autoloads = known_globals["autoloads"]
    autoload_list = ", ".join(autoloads) if autoloads else "(none detected — check project.godot)"

    # Build project-wide symbol index (cached; rebuilt only when sources change)
    try:
        project_index = build_index(project_path)
    except Exception as e:
        _log.warning("[code_review] project_indexer failed, continuing without context: %s", e)
        project_index = {}

    gd_files = sorted(root.rglob("*.gd"))
    game_files = [f for f in gd_files if "libraries" not in f.parts]

    # Scope filtering: only review files in the impact set (empty scope = full project)
    scope = set(state.get("scope", []))
    if scope:
        game_files = [f for f in game_files if str(f.relative_to(root)) in scope]

    for gd in game_files:
        rel = str(gd.relative_to(root))
        code = gd.read_text(encoding="utf-8", errors="replace")

        # Skip very short files (autoload stubs, empty state files)
        if len(code.strip().splitlines()) < 5:
            continue

        # Build per-file context block and prepend to the code
        context_block = build_file_context(rel, project_index, project_path)
        if context_block:
            context_and_code = (
                f"{context_block}\n"
                f"=== CODE TO REVIEW BELOW ===\n"
                f"{code}"
            )
        else:
            context_and_code = code

        prompt = _REVIEW_PROMPT.format(
            filename=rel,
            context_and_code=context_and_code,
            autoload_list=autoload_list,
        )

        # Diagnostic: log the full prompt so context injection can be verified
        _log_prompt(log_dir, rel, prompt)

        try:
            text = _call_with_retry(
                client,
                [{"role": "user", "content": prompt}],
                max_tokens=1024,
            ).strip()

            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                issues = json.loads(match.group())
                for issue in issues:
                    issue["file"] = rel
                    issue.setdefault("check_method", "llm")
                    all_issues.append(issue)
        except Exception as e:
            error_type = _classify_api_error(e)
            _log.warning("[code_review] %s failed after retries (%s): %s", rel, error_type, e)
            if error_type == "auth_error":
                # Auth errors are permanent — nothing to retry, record as infra error
                agent_errors.append({
                    "file": rel,
                    "error_type": error_type,
                    "message": str(e),
                })
            else:
                # Transient failure (timeout / rate_limit / overloaded) that exhausted
                # retries: emit a static-only placeholder so the file still appears in
                # the report with context instead of vanishing into Infrastructure Errors.
                all_issues.append({
                    "file": rel,
                    "line": None,
                    "severity": "info",
                    "rule": "llm_review_skipped",
                    "message": (
                        f"LLM review unavailable after retries ({error_type}); "
                        "static analysis only — re-run to attempt LLM review"
                    ),
                    "check_method": "static",
                })

    status = "ok_with_errors" if agent_errors else "ok"
    return {
        "code_issues": all_issues,
        "code_review_errors": agent_errors,
        "agent_status": {"code_review": status},
    }
