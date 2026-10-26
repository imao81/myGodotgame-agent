#!/usr/bin/env python3
"""chat.py — Product-style natural-language entry to the harvester-agent.

Classifies each message (agents/intent_router.py) and dispatches:

    full_review    → review.run_pipeline (whole project)
    diff_review    → subprocess: game_agent_pipeline.py (git-diff scope)
    symbol_lookup  → agents.qa_engine.lookup_symbol (pure index, no LLM)
    code_qa        → agents.qa_engine.answer_question (index RAG + LLM)
    unknown        → usage help

The existing review pipeline is untouched — chat.py only calls into it.
Everything except full_review / LLM-enhanced code_qa works headless:
--no-llm forces rule-based intents and retrieval-only answers.

Usage
-----
  python chat.py                                  # REPL
  python chat.py --ask "MiningShip 的信号在哪声明"   # single shot
  python chat.py --no-llm --ask "MiningShip 是什么"  # headless demo
  python chat.py --project ../other/project --show-intent

Options
-------
  --project PATH   Godot project root (default: ../godot-2d-space-game/project)
  --ask TEXT       Single question mode (no REPL)
  --no-llm         No LLM calls: rule intents + retrieval-only answers
  --show-intent    Print the classification result before dispatching
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ── Bootstrap path so sibling modules are importable regardless of CWD ────────
_AGENT_ROOT = Path(__file__).parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

# ── Auto-load .env if present ─────────────────────────────────────────────────
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

from agents.intent_router import classify_intent
from agents.qa_engine import answer_question, lookup_symbol

_DEFAULT_PROJECT = "../godot-2d-space-game/project"

_HELP_TEXT = """支持的问法示例：
  符号查询   MiningShip 是什么 / ore_collected 信号在哪个文件声明
  代码问答   采矿船怎么收集矿石？/ 这个场景为什么连不上信号？
  全面审查   帮我审查一下整个项目 / 跑一遍完整检查
  增量审查   审查这次提交的改动 / 看看 diff 里有没有问题
  退出       quit / exit
"""


# ── Handler context ───────────────────────────────────────────────────────────

class ChatContext:
    """Shared per-session state passed to every intent handler."""

    def __init__(self, project_root: Path, index: dict, client=None) -> None:
        self.project_root = project_root
        self.index = index
        self.client = client


# ── Intent handlers ───────────────────────────────────────────────────────────

def handle_symbol_lookup(ctx: ChatContext, text: str, intent_info: dict) -> int:
    entities = intent_info.get("entities", [])
    if not entities:
        print("想查哪个符号？请给出类名 / 信号名 / 方法名（例如：MiningShip）。")
        return 0
    for name in entities:
        hits = lookup_symbol(ctx.index, name)["hits"]
        if not hits:
            print(f"「{name}」: 索引中没有找到。")
            continue
        print(f"「{name}」:")
        for h in hits:
            print(f"  [{h['kind']}] {h['detail']}")
    return 0


def handle_code_qa(ctx: ChatContext, text: str, intent_info: dict) -> int:
    result = answer_question(ctx.index, ctx.project_root, text,
                             client=ctx.client)
    print(result["answer"])
    if result["sources"]:
        print("\n来源:")
        for s in result["sources"]:
            print(f"  - [{s['kind']}] {s['path']} (score {s['score']})")
    if not result["llm_used"]:
        print("\n(未调用 LLM — 以上为索引检索结果)")
    return 0


def handle_full_review(ctx: ChatContext, text: str, intent_info: dict) -> int:
    if ctx.client is None:
        print("全面审查需要 LLM API key（当前未配置或使用了 --no-llm）。")
        print("可改用增量审查，或配置 key 后重试。")
        return 2
    import review as review_mod
    try:
        enriched, json_p, md_p = review_mod.run_pipeline(
            project_root=ctx.project_root)
    except Exception as exc:
        print(f"审查流水线执行失败: {exc}")
        return 2
    counts = review_mod._summarise(enriched)
    print(f"报告: {md_p}")
    print(f"JSON: {json_p}")
    print(f"{counts['verified_errors']} error(s) | "
          f"{counts['dormant_bugs']} dormant | "
          f"{counts['warnings']} warning(s) | "
          f"{counts['refuted_drops']} refuted")
    return 1 if counts["verified_errors"] > 0 else 0


def handle_diff_review(ctx: ChatContext, text: str, intent_info: dict) -> int:
    # git-diff scope is decided by the pipeline itself (resolve_scope node)
    cmd = [sys.executable, str(_AGENT_ROOT / "game_agent_pipeline.py"),
           "--project", str(ctx.project_root)]
    print(f"启动增量审查（scope 由 git diff 自动决定）: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(_AGENT_ROOT)).returncode


def handle_unknown(ctx: ChatContext, text: str, intent_info: dict) -> int:
    print("没听懂这个问题。")
    print(_HELP_TEXT)
    return 0


# Data-driven dispatch: intent → handler(ctx, text, intent_info) -> exit code
HANDLERS = {
    "full_review":   handle_full_review,
    "diff_review":   handle_diff_review,
    "symbol_lookup": handle_symbol_lookup,
    "code_qa":       handle_code_qa,
    "unknown":       handle_unknown,
}


def dispatch(intent_info: dict, ctx: ChatContext, text: str,
             handlers: dict | None = None) -> int:
    """Route a classified message to its handler. `handlers` is injectable
    so tests can substitute mocks without touching the real handlers."""
    table = handlers or HANDLERS
    handler = table.get(intent_info["intent"]) or table.get("unknown",
                                                            handle_unknown)
    return handler(ctx, text, intent_info)


# ── Session setup ─────────────────────────────────────────────────────────────

def _has_any_api_key() -> bool:
    return any(os.environ.get(k) for k in
               ("ANTHROPIC_API_KEY", "MOONSHOT_API_KEY", "DEEPSEEK_API_KEY"))


def _make_client(no_llm: bool):
    """LLMClient when usable, else None. Never raises."""
    if no_llm or not _has_any_api_key():
        return None
    try:
        from agents.llm_client import LLMClient
        return LLMClient()
    except Exception as exc:
        print(f"(LLM 初始化失败，降级为无 LLM 模式: {exc})",
              file=sys.stderr)
        return None


def process_message(text: str, ctx: ChatContext, show_intent: bool = False,
                    handlers: dict | None = None) -> int:
    intent_info = classify_intent(text, client=ctx.client)
    if show_intent:
        print(f"[intent] {intent_info['intent']} "
              f"(confidence={intent_info['confidence']}, "
              f"source={intent_info['source']}, "
              f"entities={intent_info['entities']})")
    return dispatch(intent_info, ctx, text, handlers=handlers)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chat.py",
        description="Natural-language entry: symbol lookup, code Q&A, "
                    "and review-pipeline dispatch.",
    )
    p.add_argument("--project", default=_DEFAULT_PROJECT, metavar="PATH",
                   help=f"Godot project root (default: {_DEFAULT_PROJECT})")
    p.add_argument("--ask", metavar="TEXT",
                   help="Single question mode (no REPL)")
    p.add_argument("--no-llm", action="store_true",
                   help="No LLM calls: rule intents + retrieval-only answers")
    p.add_argument("--show-intent", action="store_true",
                   help="Print the intent classification before dispatching")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    project_root = Path(args.project).resolve()
    if not project_root.is_dir():
        print(f"ERROR: project path not found: {project_root}", file=sys.stderr)
        return 2
    if not (project_root / "project.godot").exists():
        print(f"ERROR: no project.godot in {project_root}", file=sys.stderr)
        return 2

    from review import load_or_build_index
    try:
        index, index_path = load_or_build_index(project_root)
    except Exception as exc:
        print(f"ERROR: index build failed: {exc}", file=sys.stderr)
        return 2

    client = _make_client(args.no_llm)
    ctx = ChatContext(project_root, index, client)
    mode = "no-llm" if client is None else "llm"
    print(f"[chat] 项目: {project_root} | 索引: {index_path} | 模式: {mode}",
          file=sys.stderr)

    if args.ask:
        return process_message(args.ask, ctx, show_intent=args.show_intent)

    print("harvester-agent chat — 输入问题，quit/exit 退出。")
    print(_HELP_TEXT)
    while True:
        try:
            text = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in ("quit", "exit"):
            break
        process_message(text, ctx, show_intent=args.show_intent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
