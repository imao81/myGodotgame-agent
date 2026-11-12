"""tests/test_tracer.py — trajectory-level tracing.

No real LLM calls: LLMClient internals are monkeypatched.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import tracer


class TracerCase(unittest.TestCase):
    def tearDown(self):
        tracer.end_run()  # never leak an active tracer across tests

    def _events(self, path: Path) -> list[dict]:
        return [json.loads(l) for l in
                path.read_text(encoding="utf-8").splitlines() if l.strip()]


class TestRunLifecycle(TracerCase):
    def test_inactive_by_default(self):
        self.assertIsNone(tracer.get())
        tracer.record_llm("kimi", "k3", None, 0.1, True)  # no-op, no crash

    def test_start_record_end(self):
        import tempfile
        out = Path(tempfile.mkdtemp())
        tracer.start_run(out)
        tracer.record_llm("kimi", "kimi-k3",
                          {"prompt_tokens": 10, "completion_tokens": 5},
                          0.25, True)
        tracer.record_llm("deepseek", "v4-flash", None, 0.1, False, error="boom")
        summary = tracer.end_run()

        self.assertEqual(summary["llm_calls"], 2)
        self.assertEqual(summary["llm_failed"], 1)
        self.assertEqual(summary["tokens_by_provider"], {"kimi": 15, "deepseek": 0})
        events = self._events(out / "trace.jsonl")
        self.assertEqual([e["type"] for e in events], ["llm_call", "llm_call"])
        self.assertEqual(events[1]["error"], "boom")
        self.assertIn("caller", events[0])

    def test_end_without_start_returns_empty(self):
        self.assertEqual(tracer.end_run(), {})


class TestNodeSpan(TracerCase):
    def test_traced_wrapper_records_span(self):
        import tempfile
        out = Path(tempfile.mkdtemp())
        tracer.start_run(out)

        wrapped = tracer.traced("demo_node", lambda s: {"ok": True})
        self.assertEqual(wrapped({"x": 1}), {"ok": True})
        tracer.end_run()

        spans = [e for e in self._events(out / "trace.jsonl")
                 if e["type"] == "node_span"]
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["node"], "demo_node")
        self.assertEqual(spans[0]["status"], "ok")
        self.assertIn("duration_s", spans[0])

    def test_traced_wrapper_records_error_and_reraises(self):
        import tempfile
        out = Path(tempfile.mkdtemp())
        tracer.start_run(out)

        def boom(s):
            raise ValueError("nope")

        with self.assertRaises(ValueError):
            tracer.traced("bad_node", boom)({})
        tracer.end_run()
        spans = self._events(out / "trace.jsonl")
        self.assertEqual(spans[0]["status"], "error")

    def test_traced_passthrough_when_inactive(self):
        wrapped = tracer.traced("n", lambda s: 42)
        self.assertEqual(wrapped({}), 42)  # no tracer, no overhead


class TestLLMClientIntegration(TracerCase):
    def _client(self, monkey_enabled=False):
        from agents.llm_client import LLMClient
        with mock.patch.dict("os.environ", {"MOONSHOT_API_KEY": "sk-test"},
                             clear=False):
            return LLMClient(provider="kimi")

    def test_chat_records_success(self):
        import tempfile
        out = Path(tempfile.mkdtemp())
        tracer.start_run(out)
        client = self._client()
        with mock.patch.object(
            client, "_chat_kimi",
            return_value="hello",
        ) as m:
            client.last_usage = {"prompt_tokens": 3, "completion_tokens": 2}
            # last_usage is set by the real _chat_kimi; emulate
            def fake(messages, max_tokens):
                client.last_usage = {"prompt_tokens": 3, "completion_tokens": 2}
                return "hello"
            m.side_effect = fake
            self.assertEqual(client.chat([{"role": "user", "content": "hi"}]), "hello")
        tracer.end_run()

        events = self._events(out / "trace.jsonl")
        self.assertEqual(len(events), 1)
        e = events[0]
        self.assertEqual(e["provider"], "kimi")
        self.assertEqual(e["model"], "kimi-k3")
        self.assertTrue(e["ok"])
        self.assertEqual(e["usage"], {"prompt_tokens": 3, "completion_tokens": 2})
        self.assertIn("caller", e)
        self.assertNotIn("prompt", e)  # TRACE_FULL off by default
        self.assertEqual(e["prompt_chars"], 2)

    def test_chat_records_failure_then_fallback(self):
        import tempfile
        out = Path(tempfile.mkdtemp())
        tracer.start_run(out)
        client = self._client()
        client.enable_fallback = True
        # patch the candidate list directly — env-based resolution is polluted
        # by other tests in the full suite
        with mock.patch("agents.llm_client._fallback_candidates",
                        return_value=["deepseek"]), \
             mock.patch.object(client, "_chat_kimi",
                               side_effect=RuntimeError("kimi down")), \
             mock.patch.object(client, "_chat_deepseek", return_value="ok"):
            self.assertEqual(client.chat([{"role": "user", "content": "hi"}]), "ok")
        tracer.end_run()

        events = self._events(out / "trace.jsonl")
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["provider"], "kimi")
        self.assertFalse(events[0]["ok"])
        self.assertEqual(events[1]["provider"], "deepseek")
        self.assertTrue(events[1]["ok"])

    def test_trace_full_captures_text(self):
        import tempfile
        out = Path(tempfile.mkdtemp())
        with mock.patch.dict("os.environ", {"TRACE_FULL": "1"}):
            tracer.start_run(out)
            client = self._client()
            with mock.patch.object(client, "_chat_kimi", return_value="answer"):
                client.chat([{"role": "user", "content": "question"}])
            tracer.end_run()
        e = self._events(out / "trace.jsonl")[0]
        self.assertEqual(e["prompt"], "question")
        self.assertEqual(e["completion"], "answer")


if __name__ == "__main__":
    unittest.main()
