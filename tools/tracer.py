"""Trajectory-level tracing for pipeline runs.

Records every LLM call (provider, model, usage, latency, outcome) and every
graph node span to `<output_dir>/trace.jsonl`. Inactive (zero overhead) unless
start_run() was called — standalone tools are never traced accidentally.

Set TRACE_FULL=1 to also persist prompt/completion text (large files).
"""

from __future__ import annotations

import inspect
import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

_lock = threading.Lock()
_active: "Tracer | None" = None

_TRACE_SKIP_MODULES = {"llm_client", "tracer"}


class Tracer:
    def __init__(self, path: Path):
        self.path = path
        self._fh = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self._llm_calls = 0
        self._llm_failed = 0
        self._tokens: dict[str, int] = {}
        self._spans = 0

    def record(self, event: dict) -> None:
        event = {"ts": round(time.time(), 3), **event}
        with self._lock:
            self._fh.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._fh.flush()
            if event.get("type") == "llm_call":
                self._llm_calls += 1
                if not event.get("ok"):
                    self._llm_failed += 1
                usage = event.get("usage") or {}
                provider = event.get("provider", "?")
                tok = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
                self._tokens[provider] = self._tokens.get(provider, 0) + tok
            elif event.get("type") == "node_span":
                self._spans += 1

    def summary(self) -> dict:
        return {
            "trace_path": str(self.path),
            "llm_calls": self._llm_calls,
            "llm_failed": self._llm_failed,
            "tokens_by_provider": dict(self._tokens),
            "node_spans": self._spans,
        }

    def close(self) -> None:
        with self._lock:
            self._fh.close()


def start_run(output_dir: str | Path) -> Tracer:
    """Begin a traced run; trace.jsonl lands next to the other reports."""
    global _active
    path = Path(output_dir) / "trace.jsonl"
    with _lock:
        _active = Tracer(path)
    return _active


def end_run() -> dict:
    global _active
    with _lock:
        tracer, _active = _active, None
    if tracer is None:
        return {}
    summary = tracer.summary()
    tracer.close()
    return summary


def get() -> "Tracer | None":
    return _active


def _caller() -> str:
    """First stack frame outside llm_client/tracer — who made this LLM call."""
    frame = inspect.currentframe()
    try:
        frame = frame.f_back.f_back  # skip _caller + record_llm
        while frame:
            module = frame.f_globals.get("__name__", "")
            stem = module.rsplit(".", 1)[-1]
            if stem not in _TRACE_SKIP_MODULES:
                return f"{stem}.{frame.f_code.co_name}"
            frame = frame.f_back
        return "unknown"
    finally:
        del frame


def record_llm(provider: str, model: str, usage: dict | None, latency_s: float,
               ok: bool, error: str | None = None,
               prompt_text: str | None = None, completion_text: str | None = None) -> None:
    tracer = get()
    if tracer is None:
        return
    event = {
        "type": "llm_call",
        "provider": provider,
        "model": model,
        "caller": _caller(),
        "usage": usage,
        "latency_s": round(latency_s, 3),
        "ok": ok,
    }
    if error:
        event["error"] = error
    if os.environ.get("TRACE_FULL") == "1":
        event["prompt"] = prompt_text
        event["completion"] = completion_text
    else:
        event["prompt_chars"] = len(prompt_text) if prompt_text else None
        event["completion_chars"] = len(completion_text) if completion_text else None
    tracer.record(event)


def traced(name: str, fn):
    """Wrap a graph node so its execution is recorded as a node_span."""
    def wrapper(*args, **kwargs):
        tracer = get()
        if tracer is None:
            return fn(*args, **kwargs)
        t0 = time.perf_counter()
        status = "ok"
        try:
            return fn(*args, **kwargs)
        except Exception:
            status = "error"
            raise
        finally:
            tracer.record({
                "type": "node_span",
                "node": name,
                "duration_s": round(time.perf_counter() - t0, 3),
                "status": status,
            })
    wrapper.__name__ = getattr(fn, "__name__", name)
    return wrapper


@contextmanager
def span(name: str):
    """Ad-hoc span for code outside graph nodes (e.g. per-file review loop)."""
    tracer = get()
    if tracer is None:
        yield
        return
    t0 = time.perf_counter()
    status = "ok"
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        tracer.record({"type": "node_span", "node": name,
                       "duration_s": round(time.perf_counter() - t0, 3),
                       "status": status})
