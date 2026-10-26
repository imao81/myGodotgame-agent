#!/usr/bin/env python3
"""mcp_server.py — Expose harvester-agent's review/Q&A capabilities over MCP.

Thin packaging layer: every tool delegates to existing functions
(review.run_pipeline, review.load_or_build_index, agents.qa_engine.*).
No new review logic lives here.

Run (stdio transport):
    .venv/Scripts/python.exe mcp_server.py

Long-running reviews use a start/poll pattern (start_review returns a
task_id immediately, get_review_result polls) because MCP tool calls are
request/response — a synchronous multi-minute review would look hung to
the host and offers no progress channel.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

# ── Bootstrap path so sibling modules are importable regardless of CWD ────────
_AGENT_ROOT = Path(__file__).parent.resolve()
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

# ── Auto-load .env if present (same precedent as review.py) ──────────────────
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(_AGENT_ROOT / ".env")
except ImportError:
    pass

# Windows GBK terminals can't encode some glyphs — replace instead of crashing
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="replace")
        except Exception:
            pass

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from review import load_or_build_index, run_pipeline
from agents.qa_engine import answer_question, lookup_symbol as _qa_lookup_symbol

# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_PROJECT = (_AGENT_ROOT / "../godot-2d-space-game/project").resolve()
_REPORTS_BASE = _AGENT_ROOT / "reports"
_INDEX_CACHE = _AGENT_ROOT / ".kimi" / "project_index.json"

_API_KEY_ENV = {
    "claude": "ANTHROPIC_API_KEY",
    "kimi": "MOONSHOT_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}

# ── Async review task table ───────────────────────────────────────────────────
# {task_id: {"status": running|done|failed, "started_at": float,
#            "project": str, "result": {...} | None, "error": str | None}}
_TASKS: dict[str, dict] = {}
_TASKS_LOCK = threading.Lock()
# Per-project single-flight: one running review per project at a time.
_PROJECT_LOCKS: dict[str, threading.Lock] = {}


def _project_lock(project_key: str) -> threading.Lock:
    with _TASKS_LOCK:
        lock = _PROJECT_LOCKS.get(project_key)
        if lock is None:
            lock = threading.Lock()
            _PROJECT_LOCKS[project_key] = lock
        return lock


def _resolve_project(project: str) -> Path:
    """Resolve a project argument to an absolute Path (ToolError if missing).

    Empty string means the default sibling Godot project. Relative paths
    anchor at this script's directory, not the MCP host's CWD.
    """
    raw = project.strip() if project else ""
    p = Path(raw).expanduser() if raw else _DEFAULT_PROJECT
    if not p.is_absolute():
        p = _AGENT_ROOT / p
    p = p.resolve()
    if not p.is_dir():
        raise ToolError(f"project path not found: {p}")
    return p


def _run_review_task(task_id: str, project_root: Path, target: str,
                     lock: threading.Lock) -> None:
    """Background worker: run the pipeline, record outcome, release the lock."""
    try:
        _findings, json_path, md_path = run_pipeline(
            project_root, target=target or ".",
        )
        with _TASKS_LOCK:
            _TASKS[task_id].update({
                "status": "done",
                "result": {"report_json": str(json_path),
                           "report_md": str(md_path)},
            })
    except Exception as exc:  # noqa: BLE001 — surface any failure to the poller
        with _TASKS_LOCK:
            _TASKS[task_id].update({"status": "failed", "error": str(exc)})
    finally:
        lock.release()


def _has_any_api_key() -> bool:
    return any(os.environ.get(v) for v in _API_KEY_ENV.values())


def _make_client():
    """LLMClient when a key is configured, else None. Never raises."""
    if not _has_any_api_key():
        return None
    try:
        from agents.llm_client import LLMClient
        return LLMClient()
    except Exception:
        return None


def _read_report_summary(json_path: Path) -> dict:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    return data.get("summary", {})


# ── MCP server ────────────────────────────────────────────────────────────────

mcp = FastMCP("harvester-agent")


@mcp.tool()
def start_review(project: str = "", target: str = "") -> dict:
    """Start a full propose-verify code review of a Godot project, in the
    background. Returns a task_id immediately — poll with get_review_result.

    Only one review per project runs at a time; a second call while one is
    running raises a conflict error.

    project: Godot project root (contains project.godot). Empty = default
             sibling project (../godot-2d-space-game/project).
    target:  Relative sub-path to scan. Empty = whole project.
    """
    project_root = _resolve_project(project)
    lock = _project_lock(str(project_root))
    if not lock.acquire(blocking=False):
        raise ToolError(
            f"a review is already running for {project_root}; "
            "poll it with get_review_result or wait for it to finish")

    task_id = uuid.uuid4().hex[:12]
    with _TASKS_LOCK:
        _TASKS[task_id] = {
            "status": "running",
            "started_at": time.time(),
            "project": str(project_root),
            "result": None,
            "error": None,
        }
    thread = threading.Thread(
        target=_run_review_task,
        args=(task_id, project_root, target, lock),
        daemon=True,
    )
    thread.start()
    return {"task_id": task_id, "status": "started"}


@mcp.tool()
def get_review_result(task_id: str) -> dict:
    """Poll a review task started by start_review.

    running → {"status": "running", "elapsed_s": N}
    done    → {"status": "done", "summary": {...}, "report_json", "report_md"}
    failed  → {"status": "failed", "error": msg}
    Unknown task_id raises an error.
    """
    with _TASKS_LOCK:
        task = _TASKS.get(task_id)
        if task is None:
            raise ToolError(f"unknown task_id: {task_id}")
        snapshot = dict(task)

    status = snapshot["status"]
    if status == "running":
        return {"status": "running",
                "elapsed_s": round(time.time() - snapshot["started_at"], 1)}
    if status == "failed":
        return {"status": "failed", "error": snapshot["error"]}
    result = snapshot["result"] or {}
    return {
        "status": "done",
        "summary": _read_report_summary(result["report_json"]),
        "report_json": result["report_json"],
        "report_md": result["report_md"],
    }


@mcp.tool()
def get_latest_report(project: str = "") -> dict:
    """Return the newest stored review report for a project (from a previous
    run), without running anything. {"status": "no_reports"} when none exist.

    project: Godot project root. Empty = default sibling project.
    """
    project_root = _resolve_project(project)
    if not _REPORTS_BASE.is_dir():
        return {"status": "no_reports"}

    # Newest first; match the report's recorded project_path when present.
    candidates = sorted(
        (d for d in _REPORTS_BASE.glob("REVIEW_*")
         if (d / "verified_report.json").exists()),
        key=lambda d: d.name, reverse=True)
    for report_dir in candidates:
        json_path = report_dir / "verified_report.json"
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        recorded = (data.get("metadata") or {}).get("project_path")
        if recorded and os.path.normcase(os.path.normpath(recorded)) \
                != os.path.normcase(str(project_root)):
            continue
        findings = data.get("findings", [])
        top = [{
            "file": f.get("file", ""),
            "line": f.get("line", 0),
            "severity": f.get("final_severity", f.get("severity", "info")),
            "message": f.get("message", ""),
        } for f in findings[:5]]
        return {
            "generated_at": data.get("generated_at", ""),
            "summary": data.get("summary", {}),
            "top_findings": top,
        }
    return {"status": "no_reports"}


@mcp.tool()
def lookup_symbol(project: str, name: str) -> dict:
    """Find where a class, autoload, script, signal, method or scene node is
    declared. Pure index scan — never calls an LLM. Returns {"query", "hits"}.
    """
    project_root = _resolve_project(project)
    index, _index_path = load_or_build_index(project_root)
    return _qa_lookup_symbol(index, name)


@mcp.tool()
def ask_codebase(project: str, question: str) -> dict:
    """Answer a natural-language question about the codebase via index RAG:
    top-k retrieval over the project index, then an LLM answer with
    file:line citations. Without an API key it degrades honestly to a
    structured summary of the retrieved entries (llm_used=False).

    Returns {"answer", "sources", "llm_used"}.
    """
    project_root = _resolve_project(project)
    index, _index_path = load_or_build_index(project_root)
    return answer_question(index, project_root, question,
                           client=_make_client())


@mcp.tool()
def health() -> dict:
    """Server self-check: which LLM providers have API keys configured,
    whether the default project exists, and whether the index cache is warm.
    """
    return {
        "providers": {name: bool(os.environ.get(env))
                      for name, env in _API_KEY_ENV.items()},
        "default_project_exists": _DEFAULT_PROJECT.is_dir(),
        "index_cached": _INDEX_CACHE.exists(),
    }


if __name__ == "__main__":
    mcp.run()  # stdio transport
