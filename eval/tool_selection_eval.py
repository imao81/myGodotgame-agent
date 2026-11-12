"""Tool-selection eval for the MCP server descriptions.

Simulates an MCP host: given the tool manifest (name + description) and a
user request, an LLM picks the one tool to call (or null). Measures exact-
match accuracy on a hand-written fixture, A/B between description versions.

    .venv/Scripts/python.exe eval/tool_selection_eval.py --manifest v0
    .venv/Scripts/python.exe eval/tool_selection_eval.py --manifest v1 --arm k3

Real host selection stats would come from trace.jsonl; this offline proxy is
what lets us A/B description rewrites before shipping them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

_AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from dotenv import load_dotenv
load_dotenv(_AGENT_ROOT / ".env")

_EVAL_DIR = Path(__file__).resolve().parent
_CASES = _EVAL_DIR / "tool_selection_set.jsonl"
_V0 = _EVAL_DIR / "fixtures" / "tool_manifest_v0.json"

_PROMPT = """\
You are an AI IDE host routing a user request to MCP tools. Choose the ONE
tool to call, or null if none of the tools fits the request.

Available tools:
{tool_list}

User request: "{request}"

Answer with JSON only: {{"tool": "<name>"}} or {{"tool": null}}"""


def load_cases(path: Path = _CASES) -> list[dict]:
    return [json.loads(l) for l in
            path.read_text(encoding="utf-8").splitlines() if l.strip()]


def live_manifest() -> list[dict]:
    """Extract name+description from the running FastMCP instance (v1)."""
    import mcp_server

    manager = mcp_server.mcp._tool_manager  # noqa: SLF001 — FastMCP public-ish
    tools = manager.list_tools()
    return [{"name": t.name, "description": t.description or ""} for t in tools]


def load_manifest(which: str) -> list[dict]:
    if which == "v0":
        return json.loads(_V0.read_text(encoding="utf-8"))["tools"]
    if which == "v1":
        return live_manifest()
    return json.loads(Path(which).read_text(encoding="utf-8"))["tools"]


def _extract_tool_name(text: str) -> str | None:
    """Robustly parse {"tool": ...} from LLM output (fences/chatter tolerant)."""
    import re
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            tool = data.get("tool")
            return tool if tool else None
        except json.JSONDecodeError:
            pass
    m = re.search(r'"tool"\s*:\s*"([^"]+)"', text)
    return m.group(1) if m else None


def select_tool(client, manifest: list[dict], request: str) -> str | None:
    tool_list = "\n".join(
        f"- {t['name']}: {t['description']}" for t in manifest)
    out = client.chat(
        [{"role": "user", "content": _PROMPT.format(
            tool_list=tool_list, request=request)}],
        max_tokens=128,
    )
    return _extract_tool_name(out)


def run_eval(manifest: list[dict], cases: list[dict], client) -> dict:
    valid_names = {t["name"] for t in manifest}
    correct, rows = 0, []
    for case in cases:
        picked = select_tool(client, manifest, case["request"])
        if picked not in valid_names:
            picked = None if picked is None else f"INVALID:{picked}"
        expect = case["expect"]
        ok = picked == expect or (picked is None and expect is None)
        correct += ok
        rows.append({"request": case["request"], "expect": expect,
                     "picked": picked, "ok": ok})
    return {
        "accuracy": round(correct / len(cases), 4) if cases else 0.0,
        "n": len(cases),
        "rows": rows,
    }


def _confusions(rows: list[dict]) -> list[str]:
    return [f"  {r['request']!r}: expect={r['expect']} picked={r['picked']}"
            for r in rows if not r["ok"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="v1", help="v0 | v1 | path to manifest json")
    parser.add_argument("--arm", default="flash", choices=["flash", "k3"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from agents.llm_client import LLMClient
    client = (LLMClient(provider="deepseek", model="deepseek-v4-flash")
              if args.arm == "flash"
              else LLMClient(provider="kimi", model="kimi-k3"))

    manifest = load_manifest(args.manifest)
    cases = load_cases()
    result = run_eval(manifest, cases, client)

    lines = [
        f"# Tool-Selection Eval — manifest={args.manifest} arm={args.arm}",
        "",
        f"accuracy: {result['accuracy'] * 100:.1f}% ({round(result['accuracy'] * result['n'])}/{result['n']})",
        "",
    ]
    misses = _confusions(result["rows"])
    lines.append("confusions:" if misses else "no confusions — perfect run")
    lines.extend(misses)
    report = "\n".join(lines)
    print(report)

    out = Path(args.out) if args.out else (
        _EVAL_DIR / "results" / f"tool_selection_{args.manifest}_{args.arm}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
