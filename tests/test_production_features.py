"""tests/test_production_features.py — production-grade LangGraph features.

Covers the ① production-hardening work:
  - retry_policy: declarative node retry (flaky node recovers; exhaustion raises)
  - checkpointer: node-boundary persistence + `invoke(None)` resume after failure
  - human_gate/auto_fix: interrupt pauses the graph; Command(resume=...) answers it
  - pipeline_check_node: transient errors re-raise (for retry_policy),
    permanent errors degrade to the static fallback

No LLM, no Godot binary, no real game project required.

Run:
    python -m pytest tests/test_production_features.py -v
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock
from typing_extensions import TypedDict

sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy

from game_agent_pipeline import (
    GameReport,
    _default_state,
    _pipeline_check_fallback,
    auto_fix_node,
    build_graph,
    human_gate_node,
    pipeline_check_node,
)


class TestRetryPolicy(unittest.TestCase):
    """Declarative retry: framework re-runs a failed node, then gives up."""

    @staticmethod
    def _flaky_graph(side_effects: list[Exception | None], calls: list[int]):
        class S(TypedDict):
            x: int

        def node(_: S) -> dict:
            calls.append(1)
            if side_effects:
                exc = side_effects.pop(0)
                if exc is not None:
                    raise exc
            return {"x": 1}

        g = StateGraph(S)
        # Explicit retry_on: langgraph's default_retry_on treats ValueError /
        # OSError / RuntimeError as deterministic bugs and does NOT retry them.
        g.add_node("flaky", node, retry_policy=RetryPolicy(
            max_attempts=3, initial_interval=0.01, backoff_factor=1.0,
            jitter=False, retry_on=(ValueError, OSError),
        ))
        g.add_edge(START, "flaky")
        g.add_edge("flaky", END)
        return g.compile()

    def test_recovers_after_transient_failures(self):
        calls: list[int] = []
        app = self._flaky_graph([ValueError("boom"), ValueError("boom"), None], calls)
        result = app.invoke({"x": 0})
        self.assertEqual(result["x"], 1)
        self.assertEqual(len(calls), 3)

    def test_exhaustion_propagates(self):
        calls: list[int] = []

        class S(TypedDict):
            x: int

        def always_fails(_: S) -> dict:
            calls.append(1)
            raise OSError("godot crashed")

        g = StateGraph(S)
        g.add_node("flaky", always_fails, retry_policy=RetryPolicy(
            max_attempts=3, initial_interval=0.01, backoff_factor=1.0,
            jitter=False, retry_on=(OSError,),
        ))
        g.add_edge(START, "flaky")
        g.add_edge("flaky", END)
        app = g.compile()
        with self.assertRaises(OSError):
            app.invoke({"x": 0})
        self.assertEqual(len(calls), 3)


class TestCheckpointerResume(unittest.TestCase):
    """Node-boundary checkpoints: a failed run resumes where it stopped."""

    def test_resume_after_mid_graph_failure(self):
        class S(TypedDict):
            x: int
            y: int

        calls = {"a": 0, "b": 0}
        fail_b = {"on": True}

        def a(_: S) -> dict:
            calls["a"] += 1
            return {"x": 1}

        def b(_: S) -> dict:
            calls["b"] += 1
            if fail_b["on"]:
                raise RuntimeError("crash inside b")
            return {"y": 2}

        g = StateGraph(S)
        g.add_node("a", a)
        g.add_node("b", b)
        g.add_edge(START, "a")
        g.add_edge("a", "b")
        g.add_edge("b", END)
        app = g.compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "t-resume"}}

        with self.assertRaises(RuntimeError):
            app.invoke({"x": 0, "y": 0}, config)
        self.assertEqual(calls, {"a": 1, "b": 1})

        fail_b["on"] = False
        result = app.invoke(None, config)  # resume from checkpoint
        self.assertEqual((result["x"], result["y"]), (1, 2))
        # 'a' was checkpointed before the crash — it must NOT re-run
        self.assertEqual(calls, {"a": 1, "b": 2})

    def test_build_graph_accepts_checkpointer(self):
        app = build_graph(checkpointer=MemorySaver(), with_auto_fix=True)
        self.assertIsNotNone(app)


class TestHumanGateAutoFix(unittest.TestCase):
    """interrupt() pauses before auto_fix; Command(resume=...) answers the gate."""

    @staticmethod
    def _gate_graph():
        g = StateGraph(GameReport)
        g.add_node("human_gate", human_gate_node)
        g.add_node("auto_fix", auto_fix_node)
        g.add_edge(START, "human_gate")
        g.add_edge("human_gate", "auto_fix")
        g.add_edge("auto_fix", END)
        return g.compile(checkpointer=MemorySaver())

    @staticmethod
    def _state_with_error() -> dict:
        return {
            **_default_state(".", True),
            "deduplicated_findings": [{
                "file": "a.gd", "line": 3, "rule": "syntax_error",
                "message": "broken", "severity": "error",
                "final_severity": "error",
            }],
        }

    def test_interrupt_then_approve(self):
        app = self._gate_graph()
        config = {"configurable": {"thread_id": "t-approve"}}

        result = app.invoke(self._state_with_error(), config)
        interrupts = result["__interrupt__"]
        self.assertEqual(len(interrupts), 1)
        payload = interrupts[0].value
        self.assertEqual(payload["error_count"], 1)
        self.assertEqual(payload["files"], ["a.gd"])

        fake_fix = {
            "fix_results": [{"file": "a.gd", "status": "fixed", "issues_fixed": 1}],
            "fix_summary": "1 file(s) fixed, 0 skipped, 0 failed.",
            "agent_status": {"auto_fix": "ok"},
        }
        with mock.patch("agents.auto_fix.run", return_value=fake_fix) as m:
            final = app.invoke(Command(resume=True), config)

        m.assert_called_once()
        call_state, = m.call_args.args[:1]
        self.assertEqual(call_state["code_issues"][0]["file"], "a.gd")
        self.assertEqual(call_state["code_issues"][0]["severity"], "error")
        self.assertEqual(m.call_args.kwargs.get("apply"), True)
        self.assertEqual(final["fix_results"][0]["status"], "fixed")
        self.assertEqual(final["agent_status"]["auto_fix"], "ok")

    def test_interrupt_then_reject(self):
        app = self._gate_graph()
        config = {"configurable": {"thread_id": "t-reject"}}
        app.invoke(self._state_with_error(), config)

        with mock.patch("agents.auto_fix.run") as m:
            final = app.invoke(Command(resume=False), config)

        m.assert_not_called()
        self.assertEqual(final["agent_status"]["human_gate"], "rejected")
        self.assertEqual(final["agent_status"]["auto_fix"], "rejected")
        self.assertEqual(final["fix_results"], [])

    def test_no_errors_skips_gate_without_interrupt(self):
        app = self._gate_graph()
        config = {"configurable": {"thread_id": "t-clean"}}
        result = app.invoke(_default_state(".", True), config)
        self.assertNotIn("__interrupt__", result)
        self.assertEqual(result["agent_status"]["human_gate"], "skipped:no_errors")
        self.assertEqual(result["agent_status"]["auto_fix"], "skipped:no_errors")


class TestPipelineCheckNodeErrors(unittest.TestCase):
    """Transient → re-raise for retry_policy; permanent → static fallback."""

    def setUp(self):
        self.state = _default_state(".", True)

    def test_transient_error_propagates(self):
        with mock.patch("agents.pipeline_check.run",
                        side_effect=subprocess.TimeoutExpired("godot", 10)):
            with self.assertRaises(subprocess.TimeoutExpired):
                pipeline_check_node(self.state)

    def test_os_error_propagates(self):
        with mock.patch("agents.pipeline_check.run",
                        side_effect=OSError("godot binary vanished")):
            with self.assertRaises(OSError):
                pipeline_check_node(self.state)

    def test_permanent_error_uses_fallback(self):
        with mock.patch("agents.pipeline_check.run",
                        side_effect=ValueError("corrupt parser state")):
            result = pipeline_check_node(self.state)
        self.assertEqual(result, _pipeline_check_fallback(self.state))
        self.assertEqual(result["agent_status"]["pipeline_check"], "ok:static_fallback")


if __name__ == "__main__":
    unittest.main()
