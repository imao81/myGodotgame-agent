"""tests/test_mcp_server.py — MCP packaging layer over the review pipeline.

Covers:
  - project path validation (missing path → ToolError)
  - per-project single-flight lock conflict
  - review task state machine (running / done / failed, unknown id)
  - get_latest_report parsing against a fake reports/ tree + no_reports
  - lookup_symbol / ask_codebase against a synthetic index (no LLM, no API)
  - health() provider detection via monkeypatched env vars
  - in-memory MCP integration: list_tools + call health (no stdio, no API)

No real API keys, no LLM calls, no network. Run:
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_mcp_server.py -v
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from pathlib import Path

import pytest
from mcp.server.fastmcp.exceptions import ToolError

sys.path.insert(0, str(Path(__file__).parent.parent))

import mcp_server


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clean_task_table():
    """Isolate the module-level task table and project locks between tests."""
    with mcp_server._TASKS_LOCK:
        saved_tasks, mcp_server._TASKS = mcp_server._TASKS, {}
        saved_locks, mcp_server._PROJECT_LOCKS = mcp_server._PROJECT_LOCKS, {}
    yield
    with mcp_server._TASKS_LOCK:
        mcp_server._TASKS = saved_tasks
        mcp_server._PROJECT_LOCKS = saved_locks


@pytest.fixture
def project(tmp_path):
    (tmp_path / "project.godot").write_text("; fake\n", encoding="utf-8")
    return tmp_path


def _wait_status(task_id: str, want: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = mcp_server.get_review_result(task_id)
        if result["status"] == want:
            return result
        time.sleep(0.02)
    raise AssertionError(f"task {task_id} never reached {want}: {result}")


_SYNTHETIC_INDEX = {
    "scripts": {
        "res://ships/mining_ship.gd": {
            "class_name": "MiningShip",
            "extends": "CharacterBody2D",
            "methods": {"_ready": 5, "collect_ore": 12},
            "signals": ["ore_collected"],
        },
    },
    "scenes": {},
    "class_names": {"MiningShip": "res://ships/mining_ship.gd"},
    "autoloads": {"Events": "res://autoload/events.gd"},
}


# ── start_review / get_review_result ─────────────────────────────────────────

def test_start_review_missing_project_raises():
    with pytest.raises(ToolError, match="project path not found"):
        mcp_server.start_review(project="Z:/definitely/not/here")


def test_start_review_returns_task_id_and_completes(project, tmp_path, monkeypatch):
    report_dir = tmp_path / "reports" / "REVIEW_20260101_000000"
    report_dir.mkdir(parents=True)
    summary = {"total": 2, "verified_errors": 1, "dormant_bugs": 0,
               "warnings": 1, "infos": 0, "refuted_drops": 0}
    (report_dir / "verified_report.json").write_text(
        json.dumps({"generated_at": "t", "metadata": {}, "summary": summary,
                    "findings": []}), encoding="utf-8")
    md_path = report_dir / "verified_report.md"
    md_path.write_text("# report\n", encoding="utf-8")

    monkeypatch.setattr(mcp_server, "run_pipeline",
                        lambda root, target=".": ([], report_dir / "verified_report.json", md_path))

    started = mcp_server.start_review(project=str(project))
    assert started["status"] == "started"
    result = _wait_status(started["task_id"], "done")
    assert result["summary"] == summary
    assert result["report_json"].endswith("verified_report.json")
    assert result["report_md"].endswith("verified_report.md")


def test_single_flight_conflict(project, tmp_path, monkeypatch):
    gate = threading.Event()
    report_json = tmp_path / "verified_report.json"
    report_json.write_text(json.dumps({"summary": {"total": 0}}),
                           encoding="utf-8")

    def _blocking(root, target="."):
        gate.wait(5)
        return ([], report_json, tmp_path / "verified_report.md")

    monkeypatch.setattr(mcp_server, "run_pipeline", _blocking)
    try:
        first = mcp_server.start_review(project=str(project))
        assert first["status"] == "started"
        with pytest.raises(ToolError, match="already running"):
            mcp_server.start_review(project=str(project))
        running = mcp_server.get_review_result(first["task_id"])
        assert running["status"] == "running"
        assert running["elapsed_s"] >= 0
    finally:
        gate.set()
    assert _wait_status(first["task_id"], "done")["summary"] == {"total": 0}


def test_task_failure_records_error(project, monkeypatch):
    def _boom(root, target="."):
        raise RuntimeError("index build exploded")

    monkeypatch.setattr(mcp_server, "run_pipeline", _boom)
    started = mcp_server.start_review(project=str(project))
    result = _wait_status(started["task_id"], "failed")
    assert "index build exploded" in result["error"]


def test_get_review_result_unknown_task():
    with pytest.raises(ToolError, match="unknown task_id"):
        mcp_server.get_review_result("deadbeefdead")


# ── get_latest_report ─────────────────────────────────────────────────────────

def _fake_report(reports_base: Path, ts: str, project: Path, findings: list[dict]):
    report_dir = reports_base / f"REVIEW_{ts}"
    report_dir.mkdir(parents=True)
    (report_dir / "verified_report.json").write_text(json.dumps({
        "generated_at": f"2026-01-01T{ts.split('_')[1][:2]}:00:00+00:00",
        "metadata": {"project_path": str(project)},
        "summary": {"total": len(findings), "verified_errors": 1,
                    "dormant_bugs": 0, "warnings": 0, "infos": 0,
                    "refuted_drops": 0},
        "findings": findings,
    }), encoding="utf-8")


def test_get_latest_report_parses_newest(project, tmp_path, monkeypatch):
    reports_base = tmp_path / "reports"
    monkeypatch.setattr(mcp_server, "_REPORTS_BASE", reports_base)
    _fake_report(reports_base, "20260101_010000", project, [])
    findings = [{"file": f"res://f{i}.gd", "line": i, "severity": "warning",
                 "final_severity": "error", "message": f"bug {i}"}
                for i in range(7)]
    _fake_report(reports_base, "20260101_020000", project, findings)

    result = mcp_server.get_latest_report(project=str(project))
    assert result["generated_at"] == "2026-01-01T02:00:00+00:00"
    assert result["summary"]["total"] == 7
    assert len(result["top_findings"]) == 5
    assert result["top_findings"][0] == {
        "file": "res://f0.gd", "line": 0, "severity": "error", "message": "bug 0"}


def test_get_latest_report_no_reports(project, tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_server, "_REPORTS_BASE", tmp_path / "empty")
    assert mcp_server.get_latest_report(project=str(project)) == {"status": "no_reports"}


def test_get_latest_report_skips_other_projects(project, tmp_path, monkeypatch):
    reports_base = tmp_path / "reports"
    monkeypatch.setattr(mcp_server, "_REPORTS_BASE", reports_base)
    other = tmp_path / "other_project"
    other.mkdir()
    _fake_report(reports_base, "20260101_020000", other, [])  # newer, wrong project
    _fake_report(reports_base, "20260101_010000", project, [])  # older, ours

    result = mcp_server.get_latest_report(project=str(project))
    assert result["generated_at"] == "2026-01-01T01:00:00+00:00"


# ── lookup_symbol / ask_codebase ──────────────────────────────────────────────

@pytest.fixture
def patched_index(project, monkeypatch):
    monkeypatch.setattr(mcp_server, "load_or_build_index",
                        lambda root, **kw: (_SYNTHETIC_INDEX, Path("fake.json")))
    return project


def test_lookup_symbol_hits(patched_index):
    result = mcp_server.lookup_symbol(str(patched_index), "MiningShip")
    kinds = {h["kind"] for h in result["hits"]}
    assert "class_name" in kinds
    assert "script" in kinds


def test_lookup_symbol_no_hits(patched_index):
    result = mcp_server.lookup_symbol(str(patched_index), "NoSuchThing")
    assert result["hits"] == []


def test_ask_codebase_degrades_without_key(patched_index, monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "MOONSHOT_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    result = mcp_server.ask_codebase(str(patched_index), "MiningShip 是什么")
    assert result["llm_used"] is False
    assert result["sources"]  # retrieval still ran


def test_ask_codebase_uses_client_when_available(patched_index, monkeypatch):
    class _FakeClient:
        def chat(self, messages, max_tokens):
            return "MiningShip 是采矿船 [res://ships/mining_ship.gd:1]"

    monkeypatch.setattr(mcp_server, "_make_client", lambda: _FakeClient())
    result = mcp_server.ask_codebase(str(patched_index), "MiningShip 是什么")
    assert result["llm_used"] is True
    assert "采矿船" in result["answer"]


# ── health ────────────────────────────────────────────────────────────────────

def test_health_no_keys(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "MOONSHOT_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    result = mcp_server.health()
    assert result["providers"] == {"kimi": False, "deepseek": False,
                                   "claude": False}
    assert set(result) == {"providers", "default_project_exists", "index_cached"}


def test_health_with_kimi_key(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "MOONSHOT_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test")
    assert mcp_server.health()["providers"]["kimi"] is True


# ── In-memory MCP integration (no stdio, no network) ─────────────────────────

def test_mcp_integration_list_tools_and_health(monkeypatch):
    from mcp.shared.memory import create_connected_server_and_client_session

    for var in ("ANTHROPIC_API_KEY", "MOONSHOT_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    async def _run():
        async with create_connected_server_and_client_session(
                mcp_server.mcp) as session:
            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            assert names == {"start_review", "get_review_result",
                             "get_latest_report", "lookup_symbol",
                             "ask_codebase", "health"}
            result = await session.call_tool("health", {})
            assert not result.isError
            payload = json.loads(result.content[0].text)
            assert payload["providers"]["kimi"] is False

    asyncio.run(_run())
