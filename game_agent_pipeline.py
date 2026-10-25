"""Harvester Game Agent Pipeline.

Graph topology:
  START
    ↓
  preflight_check ──abort──► END
    ↓ continue
  build_index
    ↓
  resolve_scope
    ↓
  route_files                     [LinUCB bandit: file → model tier]
    ├────────────────────────────┐
  pipeline_check           code_review   [parallel]
    └────────────────────────────┘
              ↓ join
          playtest                     [headless Godot: runtime ground truth]
              ↓
        runtime_verifier               [dormant runtime claims re-judged]
              ↓
        cross_correlator
              ↓
        filter_refuted
              ↓
        findings_dedup
              ↓
        report_merger ──► END            [default]
              ↓                          [--auto-fix only]
        human_gate (interrupt ── human approval checkpoint)
              ↓ approved
        auto_fix ──► fix_verify ──► END  [re-run playtest to catch regressions]

Production features:
  - SqliteSaver checkpointer (.kimi/checkpoints/pipeline.sqlite) — every node
    boundary is persisted, so a crashed run resumes with --resume instead of
    restarting. Granularity is node-level: a crash inside code_review re-runs
    that node (per-file resume is future work).
  - --auto-fix wires a human approval gate (LangGraph interrupt) into the
    graph: the pipeline pauses after the report, a human approves/rejects,
    then auto_fix writes fixes (git stash + godot --check-only protected).
  - Declarative retry_policy on pipeline_check (transient Godot subprocess
    errors); code_review keeps per-file retry because node-level retry would
    re-run the whole ~30min review loop.

Usage:
  python game_agent_pipeline.py --project "../godot-2d-space-game/project"
  python game_agent_pipeline.py --project "../godot-2d-space-game/project" --no-display
  python game_agent_pipeline.py --project ... --auto-fix      # pause for approval
  python game_agent_pipeline.py --project ... --resume        # resume after crash
  python game_agent_pipeline.py --project ... --resume --approve   # answer the gate
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, RetryPolicy, interrupt
from typing_extensions import TypedDict
from rich.console import Console
from rich.panel import Panel

load_dotenv()

# Windows GBK terminals can't encode glyphs like ▶ — replace them instead of crashing
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="replace")
        except Exception:
            pass

console = Console()

# Bootstrap path so review.py helpers are importable
_PIPELINE_ROOT = Path(__file__).parent
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))


# ── Shared state ──────────────────────────────────────────────────────────────

def _merge_dicts(a: dict, b: dict) -> dict:
    """Reducer: merge two dicts, b wins on conflict."""
    return {**a, **b}


class GameReport(TypedDict):
    # Inputs
    project_path: str
    no_display: bool
    timestamp: str

    # Preflight & scope
    abort: bool
    fallback_mode: Optional[str]          # "static_only" | None
    degradation_reason: Optional[str]     # human-readable reason for fallback, if any
    broken_providers: list[str]           # providers whose keys failed preflight ping
    scope: list[str]                      # project-relative paths (empty = full project)

    # Index — path only; full dict lives on disk
    index_path: str                       # absolute path to project_index.json
    anchor_path: Optional[str]            # path to anchor findings file, if any

    # Unified findings list (single list, enriched in-place as it flows through pipeline)
    # code_review adds: line, severity, message, assertion, check_method
    # verify (inside code_review) adds: verified, final_severity, dormant_bug
    # cross_correlator adds: root_cause_ref (if correlated)
    # dedup adds: cascade_from (if cascading)
    findings: list[dict]                  # all findings including refuted (final_severity="drop")
    skipped_files: list[str]              # files that failed review after 3 retries
    meta_findings: list[dict]             # index_gap patterns from cross_correlator
    code_review_errors: list[dict]        # infrastructure errors (auth, rate-limit, timeout)
    memory_stats: dict                    # ReviewMemory curation/retrieval stats

    # Pipeline check (separate — static, no LLM enrichment)
    pipeline_issues: list[dict]

    # Playtest (runtime verification layer — headless Godot, LLM-free)
    playtest_results: list[dict]          # TEST_RESULT scenarios from test_controller.gd
    playtest_summary: str
    screenshots: list[str]
    focus_files: list[str]                # files involved in failed scenarios
    balance_issues: list[str]
    observed_signals: dict                # Events bus signals actually fired: name → count
    runtime_coverage: list[str]           # res:// scripts present in the live scene tree
    playtest_infra_error: bool            # True = playtest couldn't run (not a verdict)
    fix_verify_results: list[dict]        # playtest re-run after auto_fix (--auto-fix)

    # Model routing (contextual bandit)
    no_router: bool                       # --no-router: all files use the default model
    routing_plan: dict                    # file → {"arm", "provider", "model", "context"}
    routing_stats: dict                   # per-arm pulls/rewards/escalations this run

    # Post-correlation & dedup
    correlated_findings: list[dict]       # from cross_correlator
    deduplicated_findings: list[dict]     # from findings_dedup

    # Agent status — merge_dicts reducer lets parallel nodes write safely
    agent_status: Annotated[dict, _merge_dicts]

    # Final
    summary: str
    overall_health: str

    # Auto-fix (--auto-fix only)
    output_dir: str                       # report dir; carried in checkpoints so --resume reuses it
    auto_fix_approved: bool               # set by the human_gate interrupt
    fix_results: list[dict]               # per-file outcome records from auto_fix
    fix_summary: str


def _default_state(project_path: str, no_display: bool, output_dir: str = "",
                   no_router: bool = False) -> GameReport:
    return GameReport(
        project_path=str(Path(project_path).resolve()),
        no_display=no_display,
        timestamp=datetime.now().isoformat(),
        abort=False,
        fallback_mode=None,
        degradation_reason=None,
        broken_providers=[],
        scope=[],
        index_path="",
        anchor_path=None,
        findings=[],
        skipped_files=[],
        meta_findings=[],
        code_review_errors=[],
        memory_stats={},
        pipeline_issues=[],
        playtest_results=[],
        playtest_summary="",
        screenshots=[],
        focus_files=[],
        balance_issues=[],
        observed_signals={},
        runtime_coverage=[],
        playtest_infra_error=False,
        fix_verify_results=[],
        no_router=no_router,
        routing_plan={},
        routing_stats={},
        correlated_findings=[],
        deduplicated_findings=[],
        agent_status={},
        summary="",
        overall_health="unknown",
        output_dir=output_dir,
        auto_fix_approved=False,
        fix_results=[],
        fix_summary="",
    )


# ── Utility helpers ────────────────────────────────────────────────────────────

_SEV_RANK: dict[str, int] = {"critical": 4, "error": 3, "warning": 2, "info": 1}

# Errors worth retrying at the node level (framework retry_policy). Everything
# else raised inside a node is a permanent failure → static fallback or abort.
_TRANSIENT_ERRORS = (subprocess.TimeoutExpired, OSError)


def _elapsed(t0: float) -> str:
    return f"{time.perf_counter() - t0:.1f}s"


# ── Scope resolution helpers ───────────────────────────────────────────────────

def _get_git_changed_files(project_path: Path) -> set[str]:
    """Return project-relative paths of changed .gd/.tscn/.tres files vs HEAD."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, cwd=str(project_path), timeout=10,
        )
        if r.returncode != 0:
            return set()
        git_root = Path(r.stdout.strip())

        # Run from git_root so both commands emit root-relative paths:
        # `git diff --name-only` is root-relative from any cwd, but
        # `git ls-files` is cwd-relative — mixing them silently drops files.
        r1 = subprocess.run(["git", "diff", "--name-only", "HEAD"],
                             capture_output=True, text=True,
                             cwd=str(git_root), timeout=10)
        r2 = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"],
                             capture_output=True, text=True,
                             cwd=str(git_root), timeout=10)

        relevant_exts = {".gd", ".tscn", ".tres"}
        changed: set[str] = set()
        for raw in r1.stdout.splitlines() + r2.stdout.splitlines():
            raw = raw.strip()
            if not raw:
                continue
            abs_path = git_root / raw
            if abs_path.suffix not in relevant_exts:
                continue
            try:
                changed.add(str(abs_path.relative_to(project_path)))
            except ValueError:
                pass  # file is outside project_path
        return changed
    except Exception:
        return set()


def _expand_dependency_scope(project_path: Path, seed: set[str]) -> set[str]:
    """BFS-expand seed files through the GDScript/scene/resource reference graph."""
    _REF_RE = re.compile(r'"(res://[^"]+\.(?:gd|tscn|tres))"')

    # Build reverse map: target_posix_rel → {source_posix_rel, ...}
    rev_refs: dict[str, set[str]] = defaultdict(set)
    for src in list(project_path.rglob("*.gd")) + list(project_path.rglob("*.tscn")):
        if "libraries" in src.parts:
            continue
        try:
            text = src.read_text(encoding="utf-8", errors="replace")
            src_rel = src.relative_to(project_path).as_posix()
            for m in _REF_RE.finditer(text):
                target = m.group(1)[len("res://"):]  # strip res://
                rev_refs[target].add(src_rel)
        except Exception:
            pass

    scope: set[str] = set(seed)
    queue = list(seed)
    while queue:
        current = queue.pop(0)
        current_posix = current.replace("\\", "/")
        for referrer in rev_refs.get(current_posix, set()):
            if referrer not in scope:
                scope.add(referrer)
                queue.append(referrer)
    return scope


# ── Node: preflight_check ─────────────────────────────────────────────────────

def _check_static_prerequisites(project_path: Path) -> list[str]:
    """Return hard-failure messages for missing project.godot or Godot binary."""
    from tools.godot_runner import GODOT_EXECUTABLE
    failures: list[str] = []
    if not (project_path / "project.godot").exists():
        failures.append(f"project.godot not found in {project_path}")
    if not Path(GODOT_EXECUTABLE).exists():
        failures.append(f"Godot executable not found: {GODOT_EXECUTABLE}")
    return failures


# Maps provider names (and aliases) to their required env var.
# None means no key is required (e.g. local inference).
_PROVIDER_KEY_MAP: dict[str, Optional[str]] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "kimi": "MOONSHOT_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
    "local": None,
}


def _ping_provider(provider: str, api_key: str) -> Optional[str]:
    """One-token connectivity check. Returns an error string, or None if OK.

    Uses the model the pipeline will actually route to, so a wrong model name
    fails here instead of producing per-file 404s downstream.
    """
    try:
        if provider in ("anthropic", "claude"):
            import anthropic
            from agents.llm_client import _CLAUDE_TEXT_MODEL
            anthropic.Anthropic(api_key=api_key).messages.create(
                model=_CLAUDE_TEXT_MODEL,
                max_tokens=1,
                messages=[{"role": "user", "content": "ping"}],
            )
        elif provider in ("moonshot", "kimi"):
            from openai import OpenAI
            from agents.llm_client import _KIMI_TEXT_MODEL
            OpenAI(api_key=api_key, base_url="https://api.moonshot.cn/v1").chat.completions.create(
                model=_KIMI_TEXT_MODEL,
                max_tokens=1,
                messages=[{"role": "user", "content": "ping"}],
            )
        elif provider == "deepseek":
            from openai import OpenAI
            from agents.llm_client import _DEEPSEEK_TEXT_MODEL
            OpenAI(api_key=api_key, base_url="https://api.deepseek.com").chat.completions.create(
                model=_DEEPSEEK_TEXT_MODEL,
                max_tokens=1,
                messages=[{"role": "user", "content": "ping"}],
            )
        elif provider == "openai":
            from openai import OpenAI
            OpenAI(api_key=api_key).chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=1,
                messages=[{"role": "user", "content": "ping"}],
            )
        return None
    except Exception as e:
        return str(e)


def _is_permanent_ping_failure(msg: str) -> bool:
    """Auth/unknown-model errors won't heal within a run; transient ones might."""
    m = msg.lower()
    return ("401" in m or "authentication" in m or "invalid" in m
            or "unauthorized" in m or "404" in m
            or "resource_not_found" in m or "not found the model" in m)


def _check_api_key() -> tuple[Optional[str], Optional[str], str]:
    """Ping the configured LLM provider.

    Returns (fallback_mode, degradation_reason, provider):
      - ("static_only", "<reason>", provider) when LLM calls must be skipped
      - (None, None, provider) when the provider is ready
    """
    provider = os.environ.get("LLM_PROVIDER", "").lower().strip()

    # Mirror llm_client._resolve_provider auto-detection logic
    if not provider:
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("MOONSHOT_API_KEY"):
            provider = "kimi"
        else:
            provider = "anthropic"  # Will produce a clear auth error below

    if provider not in _PROVIDER_KEY_MAP:
        reason = f"Unknown LLM_PROVIDER={provider!r}"
        console.log(f"[yellow]  {reason} → static-only mode[/yellow]")
        return "static_only", reason, provider

    env_var = _PROVIDER_KEY_MAP[provider]

    # Local providers need no API key
    if env_var is None:
        console.log(f"[dim]  Provider '{provider}' (local) — skipping API key check ✓[/dim]")
        return None, None, provider

    api_key = os.environ.get(env_var, "")
    if not api_key:
        reason = f"{env_var} not configured (provider: {provider})"
        console.log(f"[yellow]  {reason} → static-only mode[/yellow]")
        return "static_only", reason, provider

    err = _ping_provider(provider, api_key)
    if err is None:
        console.log(f"[dim]  {env_var} valid ({provider}) ✓[/dim]")
        return None, None, provider
    if _is_permanent_ping_failure(err):
        reason = f"{env_var} rejected by {provider} API: {err}"
        console.log(f"[yellow]  {reason} → static-only mode[/yellow]")
        return "static_only", reason, provider
    # Transient error (rate-limit, network) — warn but let agents retry
    console.log(f"[yellow]  API ping inconclusive (transient?): {err}[/yellow]")
    return None, None, provider


def preflight_check_node(state: GameReport) -> dict:
    """Validate API keys, Godot CLI binary, and project.godot.

    The primary provider decides full vs static-only mode. Additionally, every
    OTHER configured provider key gets a one-token ping: the router can only
    use arms whose provider actually works, so broken secondary providers are
    excluded up front instead of failing per-file deep into the run.
    """
    t0 = time.perf_counter()
    console.log("[bold cyan]▶ Preflight Check[/bold cyan]")

    project_path = Path(state["project_path"])
    hard_failures = _check_static_prerequisites(project_path)
    fallback_mode, degradation_reason, primary = _check_api_key()

    # Secondary providers (router arms): ping only when the primary is usable
    broken_providers: list[str] = []
    if fallback_mode is None:
        for pname, env_var in (("claude", "ANTHROPIC_API_KEY"),
                               ("kimi", "MOONSHOT_API_KEY"),
                               ("deepseek", "DEEPSEEK_API_KEY")):
            key = os.environ.get(env_var, "")
            if not key or pname == primary or (pname == "claude" and primary == "anthropic"):
                continue
            err = _ping_provider(pname, key)
            if err is None:
                console.log(f"[dim]  {env_var} valid ({pname}, router arm) ✓[/dim]")
            elif _is_permanent_ping_failure(err):
                broken_providers.append(pname)
                console.log(
                    f"[yellow]  {env_var} rejected — excluding {pname} "
                    f"router arms: {err[:120]}[/yellow]"
                )
            else:
                console.log(f"[yellow]  {pname} ping inconclusive (transient?)[/yellow]")

    abort = bool(hard_failures)
    for f in hard_failures:
        console.log(f"[red]  ✗ {f}[/red]")
    if abort:
        console.log("[red]  Aborting.[/red]")
    else:
        console.log(f"[dim]  ✓ Preflight passed ({_elapsed(t0)})[/dim]")

    return {
        "abort": abort,
        "fallback_mode": fallback_mode,
        "degradation_reason": degradation_reason,
        "broken_providers": broken_providers,
        "agent_status": {
            "preflight_check": "abort" if abort else f"ok:{fallback_mode or 'full'}",
        },
    }


def preflight_route(state: GameReport) -> str:
    return "abort" if state.get("abort") else "continue"


# ── Node: build_index ─────────────────────────────────────────────────────────

def build_index_node(state: GameReport) -> dict:
    """Scan the project and write project_index.json to disk (mtime-cached).

    Stores only the path in state — downstream nodes read from disk when needed.
    Cache is valid when all .gd/.tscn file mtimes are unchanged since last build.
    """
    t0 = time.perf_counter()
    console.log("[bold cyan]▶ Build Index[/bold cyan]")

    project_path = Path(state["project_path"])
    try:
        from review import load_or_build_index, _AGENT_ROOT
        kimi_dir = _AGENT_ROOT / ".kimi"
        index, index_path = load_or_build_index(project_path, kimi_dir=kimi_dir)
        n_scripts = len(index.get("scripts", {}))
        n_scenes = len(index.get("scenes", {}))
        n_classes = len(index.get("class_names", {}))
        console.log(
            f"[dim]  {n_scripts} scripts, {n_scenes} scenes, {n_classes} classes "
            f"→ {index_path} ({_elapsed(t0)})[/dim]"
        )
        return {
            "index_path": str(index_path),
            "agent_status": {"build_index": f"ok:{n_scripts}s/{n_scenes}sc"},
        }
    except Exception as e:
        console.log(f"[red]  Build index failed: {e}[/red]")
        return {
            "index_path": "",
            "agent_status": {"build_index": f"failed: {e}"},
        }


# ── Node: resolve_scope ───────────────────────────────────────────────────────

def resolve_scope_node(state: GameReport) -> dict:
    """Read project_index.json from disk + run git diff → output file scope set."""
    t0 = time.perf_counter()
    console.log("[bold cyan]▶ Resolve Scope[/bold cyan]")

    project_path = Path(state["project_path"])
    changed = _get_git_changed_files(project_path)

    if not changed:
        console.log("[dim]  No git diff — full project in scope[/dim]")
        console.log(f"[dim]  ✓ done in {_elapsed(t0)}[/dim]")
        return {"scope": [], "agent_status": {"resolve_scope": "ok:full_project"}}

    scope = _expand_dependency_scope(project_path, changed)
    console.log(
        f"[dim]  {len(changed)} changed → {len(scope)} in-scope ({_elapsed(t0)})[/dim]"
    )
    return {
        "scope": sorted(scope),
        "agent_status": {"resolve_scope": f"ok:{len(scope)}_files"},
    }


# ── Node: route_files ─────────────────────────────────────────────────────────

_ROUTER_DIR = _PIPELINE_ROOT / ".kimi" / "router"


def _discover_review_files(project_path: Path, scope: set) -> list:
    """Shared file discovery for routing + review (excludes the injected harness)."""
    from review import discover_files as _discover_files
    gd_files = [f for f in _discover_files(project_path)
                if f.name != "test_controller.gd"]
    if scope:
        gd_files = [
            f for f in gd_files
            if f.relative_to(project_path).as_posix() in scope
            or str(f.relative_to(project_path)) in scope
        ]
    return gd_files


def route_files_node(state: GameReport) -> dict:
    """Assign each in-scope file a model tier via the LinUCB contextual bandit.

    Skipped (passthrough to the default model) when: static_only mode,
    --no-router, or only one arm is available (single provider key).
    """
    t0 = time.perf_counter()
    console.log("[bold cyan]▶ Route Files[/bold cyan]")

    def _skip(reason: str) -> dict:
        console.log(f"[dim]  {reason} — all files use the default model[/dim]")
        return {"routing_plan": {}, "routing_stats": {},
                "agent_status": {"route_files": f"skipped:{reason}"}}

    if state.get("fallback_mode") == "static_only":
        return _skip("static_only")
    if state.get("no_router"):
        return _skip("disabled")

    from agents.router import LinUCB, available_arms, file_features
    broken = set(state.get("broken_providers", []))
    all_arms = available_arms()
    arms = [a for a in all_arms if a["provider"] not in broken]
    if broken:
        console.log(f"[dim]  Excluding broken provider(s): {sorted(broken)}[/dim]")
    if len(arms) <= 1:
        return _skip("single_arm" if arms else "no_arms")

    project_path = Path(state["project_path"])
    index = _load_index_from_state(state)
    try:
        gd_files = _discover_review_files(project_path, set(state.get("scope", [])))
    except Exception as e:
        console.log(f"[red]  File discovery failed: {e}[/red]")
        return _skip(f"discovery_failed")

    contexts: dict[str, list] = {}
    for gd in gd_files:
        rel = gd.relative_to(project_path).as_posix()
        try:
            with gd.open(encoding="utf-8", errors="replace") as fh:
                n_lines = sum(1 for _ in fh)
        except OSError:
            n_lines = 0
        contexts[rel] = file_features(rel, index, n_lines).tolist()

    bandit = LinUCB.load(_ROUTER_DIR / "linucb.json", [a["name"] for a in all_arms])
    assignment = bandit.select(contexts, allowed=[a["name"] for a in arms])
    arm_by_name = {a["name"]: a for a in arms}
    plan = {
        rel: {"arm": arm_name,
              "provider": arm_by_name[arm_name]["provider"],
              "model": arm_by_name[arm_name]["model"],
              "rel_cost": arm_by_name[arm_name]["rel_cost"],
              "context": contexts[rel]}
        for rel, arm_name in assignment.items()
    }

    dist: dict[str, int] = {}
    for entry in plan.values():
        dist[entry["arm"]] = dist.get(entry["arm"], 0) + 1
    console.log(
        f"[cyan]  {len(plan)} file(s) routed: "
        + ", ".join(f"{a}×{n}" for a, n in sorted(dist.items()))
        + f" ({_elapsed(t0)})[/cyan]"
    )
    return {
        "routing_plan": plan,
        "routing_stats": {
            "arms": [a["name"] for a in arms],
            "assignment": dist,
            "bandit_pulls_before": dict(bandit.pulls),
        },
        "agent_status": {"route_files": f"ok:{len(plan)}_files"},
    }


# ── Node: pipeline_check ──────────────────────────────────────────────────────

def _pipeline_check_fallback(_: dict) -> dict:
    return {"pipeline_issues": [], "agent_status": {"pipeline_check": "ok:static_fallback"}}


def pipeline_check_node(state: GameReport) -> dict:
    """Static Godot checks. Registered with a declarative retry_policy.

    Transient infra errors (subprocess timeout, OS errors) propagate so the
    framework retries the node; permanent errors degrade to the static fallback.
    If retries are exhausted the run fails — resume it with --resume.
    """
    t0 = time.perf_counter()
    console.log("[bold cyan]▶ Pipeline Check[/bold cyan]")

    try:
        from agents.pipeline_check import run
        result = run(state)
    except _TRANSIENT_ERRORS:
        raise
    except Exception as e:
        console.log(f"[yellow]  Pipeline check failed ({e}) — static fallback[/yellow]")
        result = _pipeline_check_fallback(state)

    n = len(result.get("pipeline_issues", []))
    console.log(f"[cyan]  {n} issue(s) ({_elapsed(t0)})[/cyan]")
    return result


# ── Node: code_review ─────────────────────────────────────────────────────────

def _code_review_static_fallback(state: GameReport) -> dict:
    """Run godot --check-only on in-scope files without LLM analysis.

    Returns findings in the unified enriched format (final_severity set to "warning"
    for syntax errors — they are unverifiable without LLM context).
    """
    from tools.godot_runner import run_script
    project_path = state["project_path"]
    root = Path(project_path)
    scope = set(state.get("scope", []))
    findings: list[dict] = []

    for gd in sorted(root.rglob("*.gd")):
        if "libraries" in gd.parts:
            continue
        if gd.name == "test_controller.gd":
            continue  # playtest-injected harness, not game code
        rel = str(gd.relative_to(root))
        if scope and rel not in scope:
            continue
        res_path = f"res://{rel.replace(chr(92), '/')}"
        for err in run_script(project_path, res_path, timeout=15).get("errors", []):
            findings.append({
                "file": rel,
                "line": None,
                "severity": "error",
                "rule": "syntax_error",
                "message": err,
                "check_method": "static",
                "assertion": {"type": "unverifiable"},
                "verified": "unverifiable",
                "final_severity": "warning",   # error+unverifiable → dormant
                "dormant_bug": True,
                "trigger_condition": err,
                "_verifier": {"confidence": 0.0, "details": "static fallback",
                              "fix_suggestion": None},
            })
    return {
        "findings": findings,
        "skipped_files": [],
        "code_review_errors": [],
        "agent_status": {"code_review": "ok:static_fallback"},
    }


def _is_auth_error(err: str) -> bool:
    """Auth failures are permanent — retrying the same key is pointless."""
    e = err.lower()
    return "401" in e or "authentication" in e or "invalid x-api-key" in e


def _review_one_file_with_retry(
    gd: Path,
    project_path: Path,
    index: dict,
    provider,
    memory_hints: list[dict] | None = None,
    provider_for=None,
    arm_ladder: list[str] | None = None,
) -> tuple[list[dict], str | None, dict | None, str | None, bool, list[str]]:
    """Review one file with up to 3 attempts and exponential backoff.

    Returns (findings, error, usage, arm_used, escalated, failed_arms).
    When arm_ladder is given, attempt N escalates to ladder[min(N, len-1)] —
    a cheap model that fails its API call retried on a stronger tier.
    Auth errors fail fast: no backoff sleep, straight to the next tier (or
    give up when the ladder is exhausted) — a bad key never heals by waiting.
    failed_arms records every arm that produced an error so the bandit can
    penalise them; a broken arm that is never updated stays "cold" forever
    and keeps attracting UCB traffic.
    """
    from review import review_file
    last_err: str | None = None
    rel = gd.relative_to(project_path).as_posix()
    arm_used: str | None = None
    escalated = False
    failed_arms: list[str] = []

    for attempt in range(3):
        p = provider
        if arm_ladder and provider_for:
            arm_used = arm_ladder[min(attempt, len(arm_ladder) - 1)]
            p = provider_for(arm_used)
            if attempt > 0 and arm_used != arm_ladder[0]:
                escalated = True
        try:
            file_findings, err = review_file(gd, project_path, index, p,
                                             memory_hints=memory_hints)
            if err:
                last_err = err
                if arm_used:
                    failed_arms.append(arm_used)
                auth = _is_auth_error(err)
                if attempt < 2 and (not auth or (arm_ladder and attempt + 1 < len(arm_ladder))):
                    if auth:
                        console.log(
                            f"[yellow]  {rel}: auth error on {arm_used or 'default'} — "
                            f"escalating immediately[/yellow]"
                        )
                    else:
                        wait = 2 ** attempt
                        console.log(
                            f"[yellow]  {rel}: attempt {attempt + 1}/3 failed "
                            f"({err}), retry in {wait}s[/yellow]"
                        )
                        time.sleep(wait)
                    continue
                break  # auth with no higher tier left, or final attempt
            return file_findings, None, getattr(p, "last_usage", None), arm_used, escalated, failed_arms
        except Exception as exc:
            last_err = str(exc)
            if arm_used:
                failed_arms.append(arm_used)
            auth = _is_auth_error(last_err)
            if attempt < 2 and (not auth or (arm_ladder and attempt + 1 < len(arm_ladder))):
                if not auth:
                    wait = 2 ** attempt
                    console.log(
                        f"[yellow]  {rel}: attempt {attempt + 1}/3 raised "
                        f"({exc}), retry in {wait}s[/yellow]"
                    )
                    time.sleep(wait)
                continue
            break

    return [], last_err, None, arm_used, escalated, failed_arms


def _load_index_from_state(state: GameReport) -> dict:
    """Load project_index.json from the path stored in state. Returns {} on failure."""
    index_path_str = state.get("index_path", "")
    if not index_path_str:
        return {}
    try:
        return json.loads(Path(index_path_str).read_text(encoding="utf-8"))
    except Exception as e:
        console.log(f"[yellow]  Could not load index from {index_path_str}: {e}[/yellow]")
        return {}


def code_review_node(state: GameReport) -> dict:
    """Per-file LLM review → sanitize → verify inline, keeping refuted findings.

    Change 2: verification happens inside the per-file loop (not a separate node).
    Change 4: per-file retry delegated to _review_one_file_with_retry.

    Refuted findings (final_severity="drop") are KEPT — the cross_correlator needs
    them to detect systematic refuted-assertion patterns (index_gap meta_findings).
    """
    t0 = time.perf_counter()
    console.log("[bold cyan]▶ Code Review[/bold cyan]")

    if state.get("fallback_mode") == "static_only":
        console.log("[yellow]  Static-only mode — running syntax-only fallback[/yellow]")
        result = _code_review_static_fallback(state)
        console.log(f"[dim]  ✓ done in {_elapsed(t0)}[/dim]")
        return result

    # Load index from disk (Change 1: index is on disk, state only has path)
    index = _load_index_from_state(state)
    project_path = Path(state["project_path"])
    scope = set(state.get("scope", []))

    from review import RealLLMProvider
    provider = RealLLMProvider()  # default model — used for unrouted files

    # ── Routing setup (empty plan = routing disabled/skipped upstream) ──────
    routing_plan = state.get("routing_plan", {})
    bandit = None
    arm_table: dict[str, dict] = {}
    provider_cache: dict[str, "RealLLMProvider"] = {}
    run_pulls: dict[str, int] = {}
    run_rewards: dict[str, float] = {}
    poisoned: dict[str, int] = {}
    escalations = 0
    if routing_plan:
        from agents.router import LinUCB, available_arms
        arms = available_arms()
        arm_table = {a["name"]: a for a in arms}
        # Bandit loads the FULL arm set so learned state survives a temporarily
        # broken provider; only the escalation ladder excludes it.
        bandit = LinUCB.load(_ROUTER_DIR / "linucb.json", list(arm_table))
        broken = set(state.get("broken_providers", []))
        ladder_arms = [a for a in arms if a["provider"] not in broken]

        def _provider_for(arm_name: str):
            if arm_name not in provider_cache:
                a = arm_table[arm_name]
                provider_cache[arm_name] = RealLLMProvider(
                    model=a["model"], provider=a["provider"])
            return provider_cache[arm_name]
    else:
        _provider_for = None

    try:
        gd_files = _discover_review_files(project_path, scope)
    except Exception as e:
        console.log(f"[red]  File discovery failed: {e}[/red]")
        return {
            "findings": [], "skipped_files": [], "code_review_errors": [],
            "agent_status": {"code_review": f"failed: {e}"},
        }

    all_findings: list[dict] = []
    skipped_files: list[str] = []

    from agents.review_memory import ReviewMemory
    memory = ReviewMemory(_PIPELINE_ROOT / ".kimi" / "memory")
    mem_retrieved = 0

    for gd in gd_files:
        rel = gd.relative_to(project_path).as_posix()
        mem_hints = memory.retrieve(rel, index)
        mem_retrieved += bool(mem_hints)

        ladder = None
        entry = routing_plan.get(rel)
        if entry and bandit is not None:
            from agents.router import escalation_ladder
            ladder = escalation_ladder(ladder_arms, entry["arm"])

        file_findings, err, usage, arm_used, escalated, failed_arms = _review_one_file_with_retry(
            gd, project_path, index, provider, memory_hints=mem_hints or None,
            provider_for=_provider_for, arm_ladder=ladder,
        )
        if err is not None:
            console.log(f"[yellow]  Skipping {rel} after 3 failures: {err}[/yellow]")
            skipped_files.append(rel)
            # Feed failures back as negative reward — otherwise a broken arm
            # stays cold-started forever and keeps attracting UCB traffic.
            if entry and bandit is not None:
                penalty = -2.0 if _is_auth_error(err) else -1.0
                for fa in set(failed_arms):
                    bandit.update(fa, entry["context"], penalty)
                    poisoned[fa] = poisoned.get(fa, 0) + 1
            continue

        all_findings.extend(file_findings)
        if entry and bandit is not None and arm_used:
            from agents.router import compute_reward
            reward = compute_reward(file_findings, usage, entry["rel_cost"])
            bandit.update(arm_used, entry["context"], reward)
            run_pulls[arm_used] = run_pulls.get(arm_used, 0) + 1
            run_rewards[arm_used] = run_rewards.get(arm_used, 0.0) + reward
            if escalated:
                escalations += 1

    memory_stats = memory.curate(all_findings, run_id=state.get("timestamp", ""))
    if memory_stats["added"] or memory_stats["updated"]:
        console.log(
            f"[dim]  Memory: {memory_stats['added']} added, "
            f"{memory_stats['updated']} updated "
            f"(total {memory_stats['total']})[/dim]"
        )

    n_active = sum(1 for f in all_findings if f.get("final_severity") != "drop")
    n_refuted = sum(1 for f in all_findings if f.get("final_severity") == "drop")
    console.log(
        f"[cyan]  {n_active} finding(s) + {n_refuted} refuted"
        + (f" ({len(skipped_files)} skipped)" if skipped_files else "")
        + f" ({_elapsed(t0)})[/cyan]"
    )
    status = f"ok:{len(skipped_files)}_skipped" if skipped_files else "ok"

    routing_stats: dict = {}
    if bandit is not None:
        bandit.save(_ROUTER_DIR / "linucb.json")
        routing_stats = {
            **state.get("routing_stats", {}),
            "pulls": run_pulls,
            "total_reward": {a: round(r, 3) for a, r in run_rewards.items()},
            "mean_reward": {a: round(run_rewards[a] / run_pulls[a], 3)
                            for a in run_pulls if run_pulls[a]},
            "escalations": escalations,
            "poisoned_arms": poisoned,
            "bandit_pulls_after": dict(bandit.pulls),
        }
        console.log(
            f"[dim]  Router: {run_pulls} escalations={escalations} "
            f"(bandit saved)[/dim]"
        )

    return {
        "findings": all_findings,
        "skipped_files": skipped_files,
        "code_review_errors": [],
        "memory_stats": {**memory_stats, "files_with_hints": mem_retrieved},
        "routing_stats": routing_stats,
        "agent_status": {"code_review": status},
    }


# ── Node: playtest ────────────────────────────────────────────────────────────

def playtest_node(state: GameReport) -> dict:
    """Run the automated playtest (headless Godot + injected test controller).

    LLM-free (Layer 1), so it also runs in static_only mode. Infrastructure
    failures never raise: a playtest that couldn't run is "skipped", and the
    runtime_verifier treats missing observations as "no evidence", not "bug".
    """
    t0 = time.perf_counter()
    console.log("[bold cyan]▶ Playtest[/bold cyan]")

    try:
        from agents.playtest import run as playtest_run
        result = playtest_run(state)
    except Exception as e:
        console.log(f"[yellow]  Playtest could not run: {e}[/yellow]")
        return {
            "playtest_results": [], "playtest_summary": f"skipped: {e}",
            "screenshots": [], "focus_files": [], "balance_issues": [],
            "observed_signals": {}, "runtime_coverage": [],
            "playtest_infra_error": True,
            "agent_status": {"playtest": f"skipped: {e}"},
        }

    n_signals = sum(result.get("observed_signals", {}).values())
    console.log(
        f"[cyan]  {result.get('playtest_summary', '')} — "
        f"{n_signals} signal observation(s), "
        f"{len(result.get('runtime_coverage', []))} script(s) exercised "
        f"({_elapsed(t0)})[/cyan]"
    )
    return result


# ── Node: runtime_verifier ────────────────────────────────────────────────────

def runtime_verifier_node(state: GameReport) -> dict:
    """Re-judge dormant runtime assertions against playtest ground truth.

    runtime_signal_not_emitted claims stay unverifiable statically; here they
    are refuted (signal fired) or confirmed dormant→error (context exercised,
    signal silent). Findings without runtime evidence pass through untouched.
    """
    t0 = time.perf_counter()
    console.log("[bold cyan]▶ Runtime Verifier[/bold cyan]")

    findings = state.get("findings", [])
    observed = state.get("observed_signals", {})
    playtest_ok = bool(observed) and not state.get("playtest_infra_error", False)

    from agents.runtime_verifier import verify_runtime, coverage_notes
    updated, stats = verify_runtime(findings, observed, playtest_ok)

    # Coverage notes: in-scope scripts the playtest never loaded
    scope = state.get("scope", [])
    if not scope:
        index = _load_index_from_state(state)
        scope = [p[len("res://"):] for p in index.get("scripts", {})
                 if p.startswith("res://")]
    notes = coverage_notes(scope, state.get("runtime_coverage", [])) if observed else []

    console.log(
        f"[cyan]  {stats['runtime_confirmed']} upgraded dormant→error, "
        f"{stats['runtime_refuted']} refuted by runtime evidence, "
        f"{stats['runtime_unchecked']} unchecked; "
        f"{len(notes)} coverage note(s) ({_elapsed(t0)})[/cyan]"
    )
    return {
        "findings": updated,
        "meta_findings": state.get("meta_findings", []) + notes,
        "agent_status": {
            "runtime_verifier": (
                f"ok:+{stats['runtime_confirmed']}/-{stats['runtime_refuted']}"
                if playtest_ok else "skipped:no_runtime_evidence"
            ),
        },
    }


# ── Node: cross_correlator ────────────────────────────────────────────────────

def _infer_root_cause(p_issues: list[dict], c_issues: list[dict], hints: str) -> str:
    """Determine a human-readable root cause from two lists of co-located issues."""
    p_types = {i.get("type", "") for i in p_issues}
    c_rules = {i.get("rule", "") for i in c_issues}
    if "missing_resource" in p_types and any(
        "signal" in r or "connect" in r for r in c_rules
    ):
        return "Missing resource causing signal connection failure"
    if "syntax" in p_types and any(i.get("severity") == "error" for i in c_issues):
        return "Syntax error compounding code-level errors"
    base = f"{len(p_issues)} pipeline + {len(c_issues)} code issue(s)"
    return base + (f" [hint: {hints}]" if hints else "")


def _group_by_file(issues: list[dict], key: str) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    for issue in issues:
        if issue.get(key):
            result[issue[key]].append(issue)
    return result


def cross_correlator_node(state: GameReport) -> dict:
    """Correlate pipeline_check + code_review findings; detect refuted-assertion patterns.

    Refinement A:
      1. Groups refuted findings (final_severity="drop") by (assertion.type, entity).
         When ≥3 files share the same refuted assertion, emits an index_gap meta_finding.
      2. Cross-correlates pipeline_issues + non-refuted code findings by file.

    The cross_correlator sees ALL findings (including refuted) so it can detect patterns.
    filter_refuted_node removes them afterwards.
    """
    t0 = time.perf_counter()
    console.log("[bold cyan]▶ Cross Correlator[/bold cyan]")

    all_findings = state.get("findings", [])
    pipeline_issues = state.get("pipeline_issues", [])

    # ── Refinement A.1: detect refuted-assertion patterns ─────────────────────
    from review import _detect_refuted_patterns
    meta_findings = _detect_refuted_patterns(all_findings)
    if meta_findings:
        console.log(
            f"[yellow]  {len(meta_findings)} index_gap pattern(s) detected[/yellow]"
        )

    # ── Refinement A.2: normal cross-correlation on non-refuted findings ──────
    # Only correlate active (non-dropped) code findings with pipeline issues
    active_findings = [f for f in all_findings if f.get("final_severity") != "drop"]

    pipe_by_file = _group_by_file(pipeline_issues, "path")
    code_by_file = _group_by_file(active_findings, "file")
    correlated: list[dict] = []

    # Files with issues from BOTH agents → merged root-cause finding
    for f in sorted(set(pipe_by_file) & set(code_by_file)):
        p_issues, c_issues = pipe_by_file[f], code_by_file[f]
        all_sevs = [i.get("severity", "info") for i in p_issues + c_issues]
        top_sev = max(all_sevs, key=lambda s: _SEV_RANK.get(s, 0), default="info")
        correlated.append({
            "root_file": f,
            "root_cause": _infer_root_cause(p_issues, c_issues, ""),
            "severity": top_sev,
            "sources": ["pipeline_check", "code_review"],
            "component_issues": p_issues + c_issues,
        })

    # Pipeline-only files (single-agent, carried through individually)
    for f in sorted(set(pipe_by_file) - set(code_by_file)):
        for issue in pipe_by_file[f]:
            correlated.append({
                "root_file": f,
                "root_cause": issue.get("message", ""),
                "severity": "error",
                "sources": ["pipeline_check"],
                "component_issues": [issue],
            })

    console.log(
        f"[cyan]  {len(correlated)} correlated group(s), "
        f"{len(meta_findings)} meta finding(s) ({_elapsed(t0)})[/cyan]"
    )
    return {
        "correlated_findings": correlated,
        # runtime_verifier may already have contributed coverage notes upstream
        "meta_findings": state.get("meta_findings", []) + meta_findings,
        "agent_status": {"cross_correlator": "ok"},
    }


# ── Node: filter_refuted ──────────────────────────────────────────────────────

def filter_refuted_node(state: GameReport) -> dict:
    """Remove refuted findings (final_severity="drop") from the main findings list.

    This runs AFTER cross_correlator so it saw the full set for pattern detection.
    meta_findings (index_gap diagnostics) are kept — they are info-level and always shown.
    """
    t0 = time.perf_counter()
    findings = state.get("findings", [])
    non_refuted = [f for f in findings if f.get("final_severity") != "drop"]
    n_dropped = len(findings) - len(non_refuted)
    console.log(
        f"[dim]▶ Filter Refuted: {n_dropped} dropped, "
        f"{len(non_refuted)} remain ({_elapsed(t0)})[/dim]"
    )
    return {
        "findings": non_refuted,
        "agent_status": {"filter_refuted": f"ok:{n_dropped}_dropped"},
    }


# ── Node: findings_dedup ──────────────────────────────────────────────────────

def findings_dedup_node(state: GameReport) -> dict:
    """Cascading dedup on non-refuted findings + normalised pipeline_issues.

    Uses review.deduplicate_cascading for cascading dedup (root kept, cascade demoted).
    """
    t0 = time.perf_counter()
    console.log("[bold cyan]▶ Findings Dedup[/bold cyan]")

    # Unified list: code review findings (already enriched) + normalised pipeline issues
    all_raw: list[dict] = list(state.get("findings", []))

    # Normalise pipeline_issues into the enriched schema
    for issue in state.get("pipeline_issues", []):
        all_raw.append({
            "file": issue.get("path", ""),
            "line": None,
            "severity": "error",
            "rule": issue.get("type", "pipeline"),
            "message": issue.get("message", ""),
            "check_method": issue.get("check_method", "static"),
            "assertion": {"type": "unverifiable"},
            "verified": "unverifiable",
            "final_severity": "error",
            "dormant_bug": False,
            "trigger_condition": None,
            "_verifier": {"confidence": 0.0, "details": "pipeline check",
                          "fix_suggestion": None},
        })

    # Dedup by (file, line, rule), keep highest severity
    dedup: dict[tuple, dict] = {}
    for issue in all_raw:
        key = (issue.get("file", ""), issue.get("line"), issue.get("rule", ""))
        if key not in dedup:
            dedup[key] = dict(issue)
        else:
            existing_rank = _SEV_RANK.get(dedup[key].get("severity", "info"), 0)
            new_rank = _SEV_RANK.get(issue.get("severity", "info"), 0)
            if new_rank > existing_rank:
                dedup[key] = dict(issue)

    pre_dedup = list(dedup.values())

    # Cascading dedup on the deduplicated set
    from review import deduplicate_cascading
    deduplicated = deduplicate_cascading(pre_dedup)

    console.log(
        f"[cyan]  {len(all_raw)} raw → {len(deduplicated)} deduplicated ({_elapsed(t0)})[/cyan]"
    )
    return {
        "deduplicated_findings": deduplicated,
        "agent_status": {"findings_dedup": f"ok:{len(deduplicated)}_findings"},
    }


# ── Node: report_merger ────────────────────────────────────────────────────────

def _build_report_lines(state: GameReport, overall_health: str) -> list[str]:
    pipeline_issues    = state.get("pipeline_issues", [])
    code_review_errors = state.get("code_review_errors", [])
    skipped_files      = state.get("skipped_files", [])
    meta_findings      = state.get("meta_findings", [])
    agent_status       = state.get("agent_status", {})
    correlated         = state.get("correlated_findings", [])
    deduplicated       = state.get("deduplicated_findings", [])
    scope              = state.get("scope", [])
    fallback_mode      = state.get("fallback_mode")
    degradation_reason = state.get("degradation_reason")

    mode_label = "static_only" if fallback_mode == "static_only" else "full"
    errors   = [i for i in deduplicated if i.get("final_severity") == "error"]
    warnings = [i for i in deduplicated if i.get("final_severity") == "warning"]
    dormant  = [i for i in deduplicated if i.get("dormant_bug")]

    lines = [
        "# Harvester Game Agent Report",
        f"Generated: {state.get('timestamp', '')}",
        f"Project: {state.get('project_path', '')}",
        f"Mode: **{mode_label}**",
        f"Overall health: **{overall_health.upper()}**",
        f"Scope: {'full project' if not scope else f'{len(scope)} files'}",
    ]

    if fallback_mode == "static_only":
        lines += [
            "",
            "> ⚠ Running in static-only mode"
            + (f" ({degradation_reason})" if degradation_reason else ""),
            "> LLM-powered deep analysis was skipped. Results include syntax and",
            "> structure checks only. Configure the correct API key to enable full analysis.",
        ]

    lines += [
        "", "## Agent Status",
        *[f"- {a}: {s}" for a, s in agent_status.items()],
        "", "## Pipeline Check",
        f"{len(pipeline_issues)} issue(s)",
        *[f"- [{i.get('type','')}] {i.get('path','')} — {i.get('message','')}"
          for i in pipeline_issues],
    ]

    playtest_results = state.get("playtest_results", [])
    if playtest_results or state.get("playtest_summary"):
        observed = state.get("observed_signals", {})
        lines += [
            "", "## Automated Playtest",
            state.get("playtest_summary", ""),
            *[f"- [{'PASS' if r.get('passed') else 'FAIL'}] {r.get('scenario','')}"
              + (f" — {r.get('error') or r.get('message','')}"
                 if not r.get('passed', True) else "")
              for r in playtest_results],
        ]
        if observed:
            fired = {k: v for k, v in observed.items() if v > 0}
            lines.append(
                f"Signals observed: {len(fired)}/{len(observed)} fired; "
                f"{len(state.get('runtime_coverage', []))} script(s) exercised"
            )

    lines += [
        "", "## Code Review (Deduplicated)",
        (f"{len(deduplicated)} finding(s) — "
         f"{len(errors)} error(s), {len(dormant)} dormant, {len(warnings)} warning(s)"),
        *[f"- [{i.get('final_severity', i.get('severity',''))}] "
          f"[{i.get('check_method','?')}] "
          f"{i.get('file','')}:{i.get('line','?')} — {i.get('message','')}"
          for i in deduplicated if i.get("final_severity") in ("error", "warning")],
    ]

    if code_review_errors:
        lines += ["", "### Infrastructure Errors (files not reviewed)"]
        lines += [f"- [{e.get('error_type','')}] {e.get('file','')} — {e.get('message','')}"
                  for e in code_review_errors]

    if skipped_files:
        lines += ["", "### Skipped Files (failed after 3 retries)"]
        lines += [f"- {f}" for f in skipped_files]

    if correlated:
        lines += ["", "## Correlated Findings"]
        for c in correlated:
            lines.append(
                f"- [{c.get('severity','')}] {c.get('root_file','')} — "
                f"{c.get('root_cause','')} "
                f"(sources: {', '.join(c.get('sources', []))})"
            )

    if meta_findings:
        lines += ["", "## Meta Findings (Index Gap Diagnostics)"]
        for m in meta_findings:
            lines.append(f"- [info] {m.get('message', '')}")

    return lines


def report_merger_node(state: GameReport) -> dict:
    t0 = time.perf_counter()
    console.log("[bold cyan]▶ Merging report…[/bold cyan]")

    pipeline_issues = state.get("pipeline_issues", [])
    deduplicated    = state.get("deduplicated_findings", [])
    correlated      = state.get("correlated_findings", [])

    # Failed playtest scenarios are runtime-observed failures — strong enough
    # evidence to gate red. Infra failures (r["infra"]) mean "couldn't run",
    # not "game broken", and never affect health.
    playtest_failures = [
        r for r in state.get("playtest_results", [])
        if not r.get("passed", True) and not r.get("infra")
    ]

    has_hard_errors = (
        pipeline_issues or
        playtest_failures or
        any(i.get("final_severity") in ("critical", "error") for i in deduplicated + correlated)
    )
    has_warnings = any(i.get("final_severity") == "warning" for i in deduplicated)

    # Files that failed review after retries were never analysed — claiming
    # "green" would be a false pass for the quality gate, so cap at yellow.
    skipped = state.get("skipped_files", [])

    if has_hard_errors:
        overall_health = "red"
    elif has_warnings or skipped:
        overall_health = "yellow"
    else:
        overall_health = "green"

    lines = _build_report_lines(state, overall_health)
    console.log(f"[dim]  ✓ report ready ({_elapsed(t0)})[/dim]")
    return {"summary": "\n".join(lines), "overall_health": overall_health}


# ── Nodes: human_gate + auto_fix (--auto-fix only) ────────────────────────────

_ERROR_SEVERITIES = ("critical", "error")


def _error_findings(state: GameReport) -> list[dict]:
    return [f for f in state.get("deduplicated_findings", [])
            if f.get("final_severity") in _ERROR_SEVERITIES]


def human_gate_node(state: GameReport) -> dict:
    """Pause the graph for human approval before auto_fix rewrites files.

    Uses LangGraph interrupt(): the run suspends here, the checkpoint holds the
    full state, and the caller resumes with Command(resume=True|False).
    """
    errors = _error_findings(state)
    if not errors:
        console.log("[dim]▶ Human Gate: no error findings — auto-fix skipped[/dim]")
        return {
            "auto_fix_approved": False,
            "agent_status": {"human_gate": "skipped:no_errors"},
        }

    files = sorted({f.get("file", "") for f in errors})
    approved = interrupt({
        "question": "Apply auto-fix to these error findings?",
        "error_count": len(errors),
        "files": files,
    })
    console.log(f"[cyan]▶ Human Gate: {'approved' if approved else 'rejected'}[/cyan]")
    return {
        "auto_fix_approved": bool(approved),
        "agent_status": {"human_gate": "approved" if approved else "rejected"},
    }


def auto_fix_node(state: GameReport) -> dict:
    """Apply LLM fixes for error findings. Only reachable through human_gate."""
    console.log("[bold cyan]▶ Auto Fix[/bold cyan]")

    if not state.get("auto_fix_approved"):
        no_errors = not _error_findings(state)
        return {
            "fix_results": [],
            "fix_summary": ("No error-severity findings to fix." if no_errors
                            else "Auto-fix rejected at the human gate."),
            "agent_status": {"auto_fix": "skipped:no_errors" if no_errors else "rejected"},
        }

    code_issues = [
        {"file": f.get("file"), "line": f.get("line"), "severity": "error",
         "rule": f.get("rule", ""), "message": f.get("message", "")}
        for f in _error_findings(state)
    ]
    from agents.auto_fix import run as auto_fix_run
    result = auto_fix_run(
        {"project_path": state["project_path"], "code_issues": code_issues},
        apply=True,
    )
    console.log(f"[cyan]  {result['fix_summary']}[/cyan]")
    return {
        "fix_results": result["fix_results"],
        "fix_summary": result["fix_summary"],
        "agent_status": result["agent_status"],
    }


# ── Node: fix_verify (--auto-fix only) ────────────────────────────────────────

def fix_verify_node(state: GameReport) -> dict:
    """Re-run the playtest after auto_fix and compare against the pre-fix baseline.

    The fix-verify loop closes the cycle: a fix that breaks the game is worse
    than no fix. Regression = fewer passing scenarios than the baseline → an
    error-level fix_verify result is appended to the report.
    """
    console.log("[bold cyan]▶ Fix Verify (re-running playtest)[/bold cyan]")

    if not state.get("auto_fix_approved") or not state.get("fix_results"):
        return {
            "fix_verify_results": [],
            "agent_status": {"fix_verify": "skipped:no_fix_applied"},
        }

    baseline_passed = {r.get("scenario") for r in state.get("playtest_results", [])
                       if r.get("passed") and not r.get("infra")}

    try:
        from agents.playtest import run as playtest_run
        rerun = playtest_run(state)
    except Exception as e:
        return {
            "fix_verify_results": [{
                "scenario": "fix_verify", "passed": False,
                "message": f"playtest re-run failed after fix: {e}",
            }],
            "agent_status": {"fix_verify": "infra_error"},
        }

    if rerun.get("playtest_infra_error"):
        return {
            "fix_verify_results": [{
                "scenario": "fix_verify", "passed": False,
                "message": "playtest infrastructure failed after fix — cannot verify",
            }],
            "agent_status": {"fix_verify": "infra_error"},
        }

    after_passed = {r.get("scenario") for r in rerun.get("playtest_results", [])
                    if r.get("passed") and not r.get("infra")}
    regressions = sorted(baseline_passed - after_passed)

    results: list[dict] = []
    if regressions:
        results.append({
            "scenario": "fix_verify", "passed": False,
            "message": f"fix introduced regression(s): {', '.join(regressions)} "
                       f"passed before but fail now",
        })
        status = "regression"
        console.log(f"[red]  Regression: {', '.join(regressions)}[/red]")
    else:
        results.append({
            "scenario": "fix_verify", "passed": True,
            "message": f"no regression — {len(after_passed)} scenario(s) pass "
                       f"(baseline {len(baseline_passed)})",
        })
        status = "ok"
        console.log("[green]  No regression after fix ✓[/green]")

    summary_add = (
        "\n\n## Fix Verify (playtest re-run after auto-fix)\n"
        + "\n".join(
            f"- [{'PASS' if r['passed'] else 'FAIL'}] {r['scenario']}: {r['message']}"
            for r in results
        )
    )
    return {
        "fix_verify_results": results,
        "summary": state.get("summary", "") + summary_add,
        "agent_status": {"fix_verify": status},
    }


# ── Build LangGraph ────────────────────────────────────────────────────────────

def build_graph(checkpointer=None, with_auto_fix: bool = False):
    """Compile the LangGraph pipeline.

    Node sequence:
      preflight_check → build_index → resolve_scope → route_files →
      [pipeline_check ‖ code_review] →
      playtest → runtime_verifier →
      cross_correlator → filter_refuted → findings_dedup →
      report_merger → [human_gate → auto_fix → fix_verify]  (--auto-fix only)

    Args:
        checkpointer:  e.g. SqliteSaver — persists state at every node boundary
                       so a crashed/interrupted run can resume (requires one for
                       the human_gate interrupt).
        with_auto_fix: append the human_gate → auto_fix → fix_verify tail after
                       report_merger.

    Visualise with:
        app = build_graph()
        print(app.get_graph().draw_mermaid())
    """
    graph = StateGraph(GameReport)

    graph.add_node("preflight_check",  preflight_check_node)
    graph.add_node("build_index",      build_index_node)
    graph.add_node("resolve_scope",    resolve_scope_node)
    graph.add_node("route_files",      route_files_node)
    graph.add_node("pipeline_check",   pipeline_check_node,
                   retry_policy=RetryPolicy(
                       max_attempts=3, initial_interval=1.0,
                       backoff_factor=2.0, jitter=True,
                       # default_retry_on excludes OSError as "deterministic";
                       # for us a vanished/crashed Godot binary is transient.
                       retry_on=_TRANSIENT_ERRORS,
                   ))
    graph.add_node("code_review",      code_review_node)
    graph.add_node("playtest",         playtest_node)
    graph.add_node("runtime_verifier", runtime_verifier_node)
    graph.add_node("cross_correlator", cross_correlator_node)
    graph.add_node("filter_refuted",   filter_refuted_node)
    graph.add_node("findings_dedup",   findings_dedup_node)
    graph.add_node("report_merger",    report_merger_node)

    # START → preflight
    graph.add_edge(START, "preflight_check")

    # preflight: abort or continue
    graph.add_conditional_edges("preflight_check", preflight_route, {
        "abort":    END,
        "continue": "build_index",
    })

    # build_index → resolve_scope → route_files
    graph.add_edge("build_index", "resolve_scope")
    graph.add_edge("resolve_scope", "route_files")

    # route_files → parallel fan-out
    graph.add_edge("route_files", "pipeline_check")
    graph.add_edge("route_files", "code_review")

    # parallel join → runtime layer → cross_correlator
    graph.add_edge("pipeline_check", "playtest")
    graph.add_edge("code_review",    "playtest")
    graph.add_edge("playtest",         "runtime_verifier")
    graph.add_edge("runtime_verifier", "cross_correlator")

    # cross_correlator → filter_refuted → findings_dedup → report_merger
    graph.add_edge("cross_correlator", "filter_refuted")
    graph.add_edge("filter_refuted",   "findings_dedup")
    graph.add_edge("findings_dedup",   "report_merger")

    if with_auto_fix:
        graph.add_node("human_gate", human_gate_node)
        graph.add_node("auto_fix",   auto_fix_node)
        graph.add_node("fix_verify", fix_verify_node)
        graph.add_edge("report_merger", "human_gate")
        graph.add_edge("human_gate",    "auto_fix")
        graph.add_edge("auto_fix",      "fix_verify")
        graph.add_edge("fix_verify",    END)
    else:
        graph.add_edge("report_merger", END)

    return graph.compile(checkpointer=checkpointer)


# ── Save outputs ───────────────────────────────────────────────────────────────

def save_report(state: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "final_report.md").write_text(
        state.get("summary", ""), encoding="utf-8"
    )

    fallback_mode = state.get("fallback_mode")
    metadata: dict = {
        "timestamp": state.get("timestamp", ""),
        "project_path": state.get("project_path", ""),
        "index_path": state.get("index_path", ""),
        "mode": "static_only" if fallback_mode == "static_only" else "full",
        "overall_health": state.get("overall_health", "unknown"),
        "memory": state.get("memory_stats", {}),
    }
    if fallback_mode == "static_only":
        metadata["degradation_reason"] = state.get("degradation_reason") or "unknown"
    if state.get("fix_summary"):
        metadata["auto_fix"] = {
            "approved": state.get("auto_fix_approved", False),
            "summary": state["fix_summary"],
        }
    if state.get("playtest_summary"):
        metadata["playtest"] = {
            "summary": state["playtest_summary"],
            "signals_observed": sum(state.get("observed_signals", {}).values()),
            "scripts_exercised": len(state.get("runtime_coverage", [])),
            "infra_error": state.get("playtest_infra_error", False),
        }
    routing_stats = state.get("routing_stats") or {}
    if routing_stats:
        metadata["routing"] = routing_stats
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    for key, filename in [
        ("findings",              "findings.json"),
        ("meta_findings",         "meta_findings.json"),
        ("skipped_files",         "skipped_files.json"),
        ("pipeline_issues",       "pipeline_issues.json"),
        ("correlated_findings",   "correlated_findings.json"),
        ("deduplicated_findings", "deduplicated_findings.json"),
        ("code_review_errors",    "code_review_errors.json"),
        ("agent_status",          "agent_status.json"),
        ("fix_results",           "fix_results.json"),
        ("playtest_results",      "playtest_results.json"),
        ("fix_verify_results",    "fix_verify_results.json"),
        ("observed_signals",      "observed_signals.json"),
        ("runtime_coverage",      "runtime_coverage.json"),
        ("routing_plan",          "routing_plan.json"),
    ]:
        (output_dir / filename).write_text(
            json.dumps(state.get(key, []), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    console.print(f"\n[bold green]Reports saved to:[/bold green] {output_dir}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _thread_id_for(project_path: str) -> str:
    """Stable per-project thread id, so --resume can find the checkpoint."""
    digest = hashlib.sha1(str(Path(project_path).resolve()).encode()).hexdigest()
    return f"pipeline-{digest[:12]}"


def _make_checkpointer():
    """SqliteSaver over .kimi/checkpoints/pipeline.sqlite (gitignored)."""
    from langgraph.checkpoint.sqlite import SqliteSaver
    db_dir = _PIPELINE_ROOT / ".kimi" / "checkpoints"
    db_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_dir / "pipeline.sqlite"), check_same_thread=False)
    return SqliteSaver(conn)


def _health_exit_code(state: dict) -> int:
    if state.get("abort"):
        return 2  # preflight abort — pipeline failure, not a quality verdict
    return 1 if state.get("overall_health") == "red" else 0


def _resolve_interrupts(app, config, result: dict, args, output_dir: Path) -> dict:
    """Handle human-gate interrupts until the graph completes.

    The report is saved BEFORE prompting so the human can inspect it while
    deciding. Headless runs without --approve/--reject exit here with the
    health verdict (the gate must not be masked by the pause) and print the
    exact resume command.
    """
    while result.get("__interrupt__"):
        save_report(result, output_dir)
        payload = result["__interrupt__"][0].value
        console.print(Panel(
            f"[bold]{payload.get('question', 'Approval required')}[/bold]\n"
            f"{payload.get('error_count', 0)} error finding(s) in "
            f"{len(payload.get('files', []))} file(s):\n"
            + "\n".join(f"  - {f}" for f in payload.get("files", []))
            + f"\n\nReport saved to: {output_dir}",
            title="Human Gate",
        ))
        if args.approve or args.reject:
            approved = args.approve
        elif sys.stdin.isatty():
            answer = console.input("[bold]Apply auto-fix? [y/N][/bold] ")
            approved = answer.strip().lower() in ("y", "yes")
        else:
            console.print(
                "[yellow]Paused at the human gate (headless). "
                "Resume with:[/yellow]\n"
                f"  python game_agent_pipeline.py --project {args.project} "
                "--resume --approve   # or --reject"
            )
            sys.exit(_health_exit_code(result))
        result = app.invoke(Command(resume=approved), config)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvester Game Agent Pipeline")
    parser.add_argument("--project", default="../godot-2d-space-game/project",
                        help="Path to Godot project directory")
    parser.add_argument("--no-display", action="store_true",
                        help="Skip visual testing (CI/headless environments)")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: reports/<timestamp>/)")
    parser.add_argument("--print-graph", action="store_true",
                        help="Print Mermaid graph diagram and exit")
    parser.add_argument("--auto-fix", action="store_true",
                        help="Append human_gate + auto_fix nodes (pauses for approval)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the last checkpoint instead of starting fresh")
    parser.add_argument("--approve", action="store_true",
                        help="With --resume (or at the pause): approve auto-fix")
    parser.add_argument("--reject", action="store_true",
                        help="With --resume (or at the pause): reject auto-fix")
    parser.add_argument("--no-router", action="store_true",
                        help="Disable bandit model routing (A/B baseline: default model for all files)")
    args = parser.parse_args()

    if args.print_graph:
        print(build_graph(with_auto_fix=args.auto_fix).get_graph().draw_mermaid())
        return

    if not Path(args.project).exists():
        console.print(f"[red]Project path not found: {args.project}[/red]")
        sys.exit(2)

    checkpointer = _make_checkpointer()
    app = build_graph(checkpointer=checkpointer, with_auto_fix=args.auto_fix)
    config = {"configurable": {"thread_id": _thread_id_for(args.project)}}

    if args.resume:
        snapshot = app.get_state(config)
        if not snapshot.values:
            console.print("[red]No checkpoint for this project — nothing to resume.[/red]")
            sys.exit(2)
        output_dir = Path(snapshot.values.get("output_dir") or
                          args.output or "reports/resumed")
        interrupts = [i for task in snapshot.tasks for i in task.interrupts]
        if interrupts:
            # Paused at the human gate: answer it.
            result = _resolve_interrupts(
                app, config, {"__interrupt__": interrupts, **snapshot.values},
                args, output_dir,
            )
        elif snapshot.next:
            console.print(f"[cyan]Resuming from checkpoint — next: {list(snapshot.next)}[/cyan]")
            result = app.invoke(None, config)
        else:
            console.print("[yellow]Checkpoint is already complete — nothing to resume.[/yellow]")
            sys.exit(_health_exit_code(snapshot.values))
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output) if args.output else Path("reports") / timestamp

        console.print(Panel(
            f"[bold]Harvester Game Agent Pipeline[/bold]\n"
            f"Project: {args.project}\n"
            f"Display: {'disabled' if args.no_display else 'enabled'}\n"
            f"Output:  {output_dir}",
            title="Starting",
        ))

        result = app.invoke(
            _default_state(args.project, args.no_display, str(output_dir),
                           no_router=args.no_router),
            config,
        )
        result = _resolve_interrupts(app, config, result, args, output_dir)

    health = result.get("overall_health", "unknown")
    color = {"green": "green", "yellow": "yellow", "red": "red"}.get(health, "white")
    console.print(f"\n[bold {color}]Overall health: {health.upper()}[/bold {color}]")
    if result.get("fix_summary"):
        console.print(f"[bold]Auto-fix: {result['fix_summary']}[/bold]")

    save_report(result, output_dir)
    sys.exit(_health_exit_code(result))


if __name__ == "__main__":
    main()
