"""agents/intent_router.py — Natural-language intent classification for chat.py.

Maps a free-form user message (Chinese or English) to one of five intents:

    full_review    run the whole review pipeline on the project
    diff_review    review only the current git diff (game_agent_pipeline.py)
    symbol_lookup  "where is X declared / what is X" — pure index lookup
    code_qa        "how does X work / is there a problem" — retrieval + LLM answer
    unknown        nothing matched

Two paths: a single small LLM call (when a client is available), with a
robust JSON extractor; or a keyword-rule path that works fully headless
(no API key, CI, demos). Any LLM failure silently degrades to the rules.

Usage:
    from agents.intent_router import classify_intent

    info = classify_intent("MiningShip 的信号在哪声明", client)
    # {"intent": "symbol_lookup", "confidence": 0.9,
    #  "entities": ["MiningShip"], "source": "rules"}
"""

from __future__ import annotations

import json
import re

INTENTS: tuple[str, ...] = (
    "full_review", "diff_review", "symbol_lookup", "code_qa", "unknown",
)

# ── Entity extraction ─────────────────────────────────────────────────────────

# CamelCase or snake_case identifiers — the shapes GDScript symbols take.
# A bare lowercase word ("review", "ship") is NOT an entity.
_ENTITY_RE = re.compile(r"\b(?:[A-Za-z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*"
                        r"|[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+)\b")

# English stopwords that happen to contain capitals/underscores are rare in
# practice; the noise floor is acceptable for v1.


def extract_entities(text: str) -> list[str]:
    """Pull likely symbol names (CamelCase / snake_case) out of free text."""
    seen: set[str] = set()
    out: list[str] = []
    for m in _ENTITY_RE.finditer(text):
        name = m.group()
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


# ── Rule path ─────────────────────────────────────────────────────────────────

# Checked in order — first matching group wins. diff_review precedes
# full_review because "审查这次改动" should mean diff scope, not full project.
_RULE_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("diff_review",   ("diff", "改动", "增量", "提交了", "提交")),
    ("full_review",   ("审查", "检查", "review", "跑一遍")),
    ("symbol_lookup", ("在哪", "哪个文件", "是什么", "声明在", "定义在",
                       "where is", "what is")),
    ("code_qa",       ("怎么", "为什么", "如何", "会不会", "有没有",
                       "how does", "why ")),
]


def _classify_with_rules(text: str) -> dict:
    lowered = text.lower()
    intent = "unknown"
    for name, keywords in _RULE_KEYWORDS:
        if any(kw in lowered for kw in keywords):
            intent = name
            break
    confidence = 0.9 if intent != "unknown" else 0.0
    return {"intent": intent, "confidence": confidence,
            "entities": extract_entities(text), "source": "rules"}


# ── LLM path ─────────────────────────────────────────────────────────────────

_LLM_PROMPT = """Classify the user message for a Godot code-review assistant.

Intents:
  full_review   — review the whole project ("全面审查", "跑一遍检查")
  diff_review   — review only uncommitted/committed changes ("审查改动", "diff")
  symbol_lookup — locate or describe a class/signal/method/scene/autoload
  code_qa       — explain how code works, or judge whether it has problems
  unknown       — anything else

User message:
\"\"\"{text}\"\"\"

Reply with ONLY a JSON object:
{{"intent": "<one of the intents>", "confidence": <0.0-1.0>,
  "entities": ["<symbol names mentioned, if any>"]}}"""

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_llm_json(raw: str) -> dict | None:
    """Extract the intent JSON from raw LLM output.

    Tolerates ```json fences and surrounding chatter. Returns None when no
    usable object is found or the intent value is not recognised.
    """
    m = _JSON_OBJ_RE.search(raw)
    if not m:
        return None
    try:
        data = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    intent = str(data.get("intent", "")).strip()
    if intent not in INTENTS:
        return None
    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    entities = data.get("entities", [])
    if not isinstance(entities, list):
        entities = []
    return {"intent": intent, "confidence": confidence,
            "entities": [str(e) for e in entities], "source": "llm"}


def _classify_with_llm(text: str, client) -> dict | None:
    """One small LLM call; None on any failure (caller falls back to rules)."""
    raw = client.chat(
        [{"role": "user", "content": _LLM_PROMPT.format(text=text)}],
        max_tokens=256,
    )
    return _parse_llm_json(raw)


# ── Public API ────────────────────────────────────────────────────────────────

def classify_intent(text: str, client=None) -> dict:
    """Classify a user message. Never raises; degrades to the rule path.

    Returns {"intent", "confidence", "entities", "source": "llm"|"rules"}.
    Entities from the LLM are merged with regex-extracted ones so a symbol
    the model missed is still available to the lookup handler.
    """
    result: dict | None = None
    if client is not None:
        try:
            result = _classify_with_llm(text, client)
        except Exception:
            result = None
    if result is None:
        return _classify_with_rules(text)

    regex_entities = extract_entities(text)
    merged = list(result["entities"])
    for e in regex_entities:
        if e not in merged:
            merged.append(e)
    result["entities"] = merged
    return result
