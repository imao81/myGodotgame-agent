"""tests/test_intent_chat.py — Intent router, index RAG, and chat dispatch.

Covers:
  - rule-based intent classification: 2+ positive + 1 negative per intent
  - entity extraction (CamelCase / snake_case)
  - LLM path: robust JSON parsing (```json fences, surrounding chatter,
    bad JSON → rule fallback) via a chat-compatible fake client
  - lookup_symbol on a synthetic index (class_name/signal/method/autoload/miss)
  - IndexRetriever ranking (exact name hit first) and k truncation
  - answer_question degraded path (no client): honest summary, no fabrication
  - chat dispatch table: intent → handler mapping with mocks

No real LLM calls, no network, no Godot binary.

Run:
    python -m pytest tests/test_intent_chat.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.intent_router import classify_intent, extract_entities
from agents.qa_engine import (IndexRetriever, answer_question, lookup_symbol,
                              retrieve)
import chat


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _synthetic_index() -> dict:
    return {
        "scripts": {
            "res://ships/mining_ship.gd": {
                "class_name": "MiningShip", "extends": "Ship",
                "is_tool": False,
                "signals": ["ore_collected", "docked"],
                "methods": {
                    "_ready": {"async": False, "args": [], "line": 5},
                    "mine_asteroid": {"async": True, "args": ["target"],
                                      "line": 12},
                },
                "preloads": [], "onready_vars": ["drill"],
                "has_async_methods": True,
            },
            "res://ui/hud.gd": {
                "class_name": "Hud", "extends": "CanvasLayer",
                "is_tool": False, "signals": [],
                "methods": {"update_score": {"async": False, "args": ["n"],
                                             "line": 8}},
                "preloads": [], "onready_vars": [], "has_async_methods": False,
            },
        },
        "scenes": {
            "res://ships/mining_ship.tscn": {
                "root_class": "MiningShip",
                "nodes": {"MiningShip": {"type": "CharacterBody2D"},
                          "MiningShip/Drill": {"type": "Node2D"}},
                "script_path": "res://ships/mining_ship.gd",
            },
        },
        "autoloads": {"Events": "res://autoload/events.gd"},
        "class_names": {"MiningShip": "res://ships/mining_ship.gd",
                        "Hud": "res://ui/hud.gd"},
        "signal_signatures": {},
        "files": [],
    }


class FakeClient:
    """chat()-compatible test double returning a canned response."""

    def __init__(self, response: str = "", raises: bool = False):
        self._response = response
        self._raises = raises
        self.calls: list[dict] = []
        self.last_usage = None

    def chat(self, messages: list, max_tokens: int = 1024) -> str:
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        if self._raises:
            raise RuntimeError("fake LLM failure")
        return self._response


# ── Rule-path intent classification ──────────────────────────────────────────

class TestRuleIntents(unittest.TestCase):

    def _intent(self, text: str) -> str:
        return classify_intent(text, client=None)["intent"]

    def test_full_review_positive(self):
        self.assertEqual(self._intent("帮我审查一下整个项目"), "full_review")
        self.assertEqual(self._intent("跑一遍完整检查"), "full_review")

    def test_full_review_negative(self):
        self.assertNotEqual(self._intent("MiningShip 在哪个文件"),
                            "full_review")

    def test_diff_review_positive(self):
        self.assertEqual(self._intent("审查这次提交的改动"), "diff_review")
        self.assertEqual(self._intent("看看 diff 里有没有问题"), "diff_review")

    def test_diff_review_negative(self):
        self.assertNotEqual(self._intent("帮我全面检查一遍"), "diff_review")

    def test_symbol_lookup_positive(self):
        self.assertEqual(self._intent("MiningShip 在哪个文件声明"),
                         "symbol_lookup")
        self.assertEqual(self._intent("ore_collected 是什么"), "symbol_lookup")

    def test_symbol_lookup_negative(self):
        self.assertNotEqual(self._intent("采矿船怎么收集矿石"),
                            "symbol_lookup")

    def test_code_qa_positive(self):
        self.assertEqual(self._intent("采矿船怎么收集矿石？"), "code_qa")
        self.assertEqual(self._intent("这个场景为什么连不上信号"), "code_qa")

    def test_code_qa_negative(self):
        self.assertNotEqual(self._intent("MiningShip 声明在哪"), "code_qa")

    def test_unknown(self):
        info = classify_intent("今天天气不错", client=None)
        self.assertEqual(info["intent"], "unknown")
        self.assertEqual(info["confidence"], 0.0)
        self.assertEqual(info["source"], "rules")


# ── Entity extraction ─────────────────────────────────────────────────────────

class TestEntityExtraction(unittest.TestCase):

    def test_camel_case(self):
        self.assertEqual(extract_entities("MiningShip 是什么"), ["MiningShip"])

    def test_snake_case(self):
        self.assertEqual(extract_entities("ore_collected 在哪声明"),
                         ["ore_collected"])

    def test_mixed_and_dedup(self):
        text = "MiningShip 和 mine_asteroid，再看下 MiningShip"
        self.assertEqual(extract_entities(text),
                         ["MiningShip", "mine_asteroid"])

    def test_plain_words_are_not_entities(self):
        self.assertEqual(extract_entities("审查一下整个项目"), [])


# ── LLM path ─────────────────────────────────────────────────────────────────

class TestLLMPath(unittest.TestCase):

    def test_clean_json(self):
        client = FakeClient('{"intent": "code_qa", "confidence": 0.95,'
                            ' "entities": ["MiningShip"]}')
        info = classify_intent("采矿船怎么工作", client=client)
        self.assertEqual(info["intent"], "code_qa")
        self.assertEqual(info["source"], "llm")
        self.assertIn("MiningShip", info["entities"])
        self.assertEqual(client.calls[0]["max_tokens"], 256)

    def test_fenced_json_with_chatter(self):
        client = FakeClient(
            '好的，分类结果如下：\n```json\n'
            '{"intent": "full_review", "confidence": 0.8, "entities": []}\n'
            '```\n希望对你有帮助！')
        info = classify_intent("审查一下", client=client)
        self.assertEqual(info["intent"], "full_review")
        self.assertEqual(info["source"], "llm")

    def test_bad_json_falls_back_to_rules(self):
        client = FakeClient("{intent: code_qa, not valid json")
        info = classify_intent("MiningShip 在哪", client=client)
        self.assertEqual(info["source"], "rules")
        self.assertEqual(info["intent"], "symbol_lookup")

    def test_unknown_intent_value_falls_back(self):
        client = FakeClient('{"intent": "poetry", "confidence": 1.0}')
        info = classify_intent("审查一遍", client=client)
        self.assertEqual(info["source"], "rules")
        self.assertEqual(info["intent"], "full_review")

    def test_llm_exception_falls_back(self):
        client = FakeClient(raises=True)
        info = classify_intent("MiningShip 是什么", client=client)
        self.assertEqual(info["source"], "rules")
        self.assertEqual(info["intent"], "symbol_lookup")

    def test_regex_entities_merged_into_llm_result(self):
        client = FakeClient('{"intent": "symbol_lookup", "confidence": 0.9,'
                            ' "entities": []}')
        info = classify_intent("MiningShip 在哪", client=client)
        self.assertEqual(info["source"], "llm")
        self.assertIn("MiningShip", info["entities"])


# ── lookup_symbol ─────────────────────────────────────────────────────────────

class TestLookupSymbol(unittest.TestCase):

    def setUp(self):
        self.index = _synthetic_index()

    def _kinds(self, name: str) -> list[str]:
        return [h["kind"] for h in lookup_symbol(self.index, name)["hits"]]

    def test_class_name_hit(self):
        hits = lookup_symbol(self.index, "MiningShip")["hits"]
        kinds = [h["kind"] for h in hits]
        self.assertIn("class_name", kinds)
        self.assertIn("script", kinds)
        for h in hits:
            self.assertTrue(h["path"].startswith("res://"))

    def test_signal_hit(self):
        hits = lookup_symbol(self.index, "ore_collected")["hits"]
        self.assertTrue(any(h["kind"] == "signal" and
                            h["path"] == "res://ships/mining_ship.gd"
                            for h in hits))

    def test_method_hit_with_line(self):
        hits = lookup_symbol(self.index, "mine_asteroid")["hits"]
        method = next(h for h in hits if h["kind"] == "method")
        self.assertIn(":12", method["detail"])

    def test_autoload_hit(self):
        hits = lookup_symbol(self.index, "Events")["hits"]
        self.assertTrue(any(h["kind"] == "autoload" for h in hits))

    def test_unknown_symbol_no_hits(self):
        result = lookup_symbol(self.index, "NonExistentThing")
        self.assertEqual(result["hits"], [])
        self.assertEqual(result["query"], "NonExistentThing")


# ── IndexRetriever ────────────────────────────────────────────────────────────

class TestIndexRetriever(unittest.TestCase):

    def setUp(self):
        self.index = _synthetic_index()

    def test_exact_name_ranks_first(self):
        results = retrieve(self.index, "MiningShip 的信号", k=5)
        self.assertTrue(results)
        self.assertEqual(results[0]["path"], "res://ships/mining_ship.gd")

    def test_result_shape(self):
        for r in retrieve(self.index, "mining ship", k=5):
            self.assertIn(r["kind"],
                          ("script", "scene", "class_name", "autoload"))
            self.assertTrue(r["path"].startswith("res://"))
            self.assertGreater(r["score"], 0)
            self.assertTrue(r["summary"])

    def test_k_truncation(self):
        results = retrieve(self.index, "mining ship signal", k=1)
        self.assertEqual(len(results), 1)

    def test_no_match_returns_empty(self):
        self.assertEqual(retrieve(self.index, "zzzzqqqq"), [])


# ── answer_question ───────────────────────────────────────────────────────────

class TestAnswerQuestion(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        ship = self.root / "ships"
        ship.mkdir()
        (ship / "mining_ship.gd").write_text(
            "class_name MiningShip\nextends Ship\n\n"
            "signal ore_collected\n\nfunc mine_asteroid(target):\n\tpass\n",
            encoding="utf-8")
        self.index = _synthetic_index()

    def tearDown(self):
        self._tmp.cleanup()

    def test_degraded_without_client(self):
        result = answer_question(self.index, self.root,
                                 "MiningShip 怎么采矿", client=None)
        self.assertFalse(result["llm_used"])
        self.assertTrue(result["sources"])
        # honest degradation: says no LLM, lists retrieval results, no invention
        self.assertIn("未调用 LLM", result["answer"])
        self.assertIn("mining_ship.gd", result["answer"])

    def test_llm_failure_degrades(self):
        result = answer_question(self.index, self.root,
                                 "MiningShip 怎么采矿",
                                 client=FakeClient(raises=True))
        self.assertFalse(result["llm_used"])
        self.assertTrue(result["sources"])

    def test_llm_answer_path(self):
        client = FakeClient("mine_asteroid 在 ships/mining_ship.gd:12 定义。")
        result = answer_question(self.index, self.root,
                                 "MiningShip 怎么采矿", client=client)
        self.assertTrue(result["llm_used"])
        self.assertIn("mining_ship.gd:12", result["answer"])
        # prompt carried the retrieved source body
        prompt = client.calls[0]["messages"][0]["content"]
        self.assertIn("mine_asteroid", prompt)

    def test_no_results_without_client(self):
        result = answer_question(self.index, self.root, "zzzzqqqq",
                                 client=None)
        self.assertFalse(result["llm_used"])
        self.assertEqual(result["sources"], [])


# ── chat dispatch ─────────────────────────────────────────────────────────────

class TestDispatch(unittest.TestCase):

    def test_intent_to_handler_mapping(self):
        ctx = chat.ChatContext(Path("/nonexistent"), {}, None)
        calls: list[str] = []
        fake_handlers = {}
        for intent in ("full_review", "diff_review", "symbol_lookup",
                       "code_qa", "unknown"):
            def make(name):
                def h(c, t, i):
                    calls.append(name)
                    return 0
                return h
            fake_handlers[intent] = make(intent)

        for intent in fake_handlers:
            rc = chat.dispatch({"intent": intent, "entities": []}, ctx, "msg",
                               handlers=fake_handlers)
            self.assertEqual(rc, 0)
        self.assertEqual(sorted(calls), sorted(fake_handlers))

    def test_unlisted_intent_falls_to_unknown(self):
        ctx = chat.ChatContext(Path("/nonexistent"), {}, None)
        unknown = mock.Mock(return_value=0)
        chat.dispatch({"intent": "not_a_real_intent", "entities": []}, ctx,
                      "msg", handlers={"unknown": unknown})
        unknown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
