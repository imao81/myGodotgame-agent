"""Unit tests for agents/llm_client.py.

Covers:
  - Provider resolution (_resolve_provider)
  - LLMClient construction and provider selection
  - Missing API key error (RuntimeError for Kimi)
  - Fallback behavior (LLM_FALLBACK=1)
  - Correct model names routed to each provider

Note: anthropic and openai are injected via sys.modules so this test file does
not require either package to be installed in the test environment.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from agents.llm_client import _resolve_provider, LLMClient, _KIMI_TEXT_MODEL, _CLAUDE_TEXT_MODEL


# ── sys.modules helpers ───────────────────────────────────────────────────────

def _make_openai_mod(kimi_client=None):
    """Minimal fake openai module. OpenAI() returns kimi_client."""
    mod = MagicMock()
    mod.OpenAI.return_value = kimi_client or MagicMock()
    return mod


def _make_anthropic_mod(response_text="ok"):
    """Minimal fake anthropic module with a pre-wired text response."""
    mod = MagicMock()

    class TextBlock:  # real class — _chat_claude filters blocks with isinstance()
        def __init__(self, text: str):
            self.text = text

    mod.types.TextBlock = TextBlock
    msg = MagicMock()
    msg.content = [TextBlock(text=response_text)]
    mod.Anthropic.return_value.messages.create.return_value = msg
    return mod


# ── Provider resolution ────────────────────────────────────────────────────────

class TestResolveProvider(unittest.TestCase):

    def test_claude_key_only(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}, clear=True):
            self.assertEqual(_resolve_provider(), "claude")

    def test_kimi_key_only(self):
        with patch.dict(os.environ, {"MOONSHOT_API_KEY": "sk-test"}, clear=True):
            self.assertEqual(_resolve_provider(), "kimi")

    def test_both_keys_claude_wins(self):
        with patch.dict(os.environ,
                        {"ANTHROPIC_API_KEY": "sk-a", "MOONSHOT_API_KEY": "sk-k"},
                        clear=True):
            self.assertEqual(_resolve_provider(), "claude")

    def test_neither_key_defaults_to_claude(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_resolve_provider(), "claude")

    def test_explicit_llm_provider_claude(self):
        with patch.dict(os.environ,
                        {"LLM_PROVIDER": "claude", "MOONSHOT_API_KEY": "sk-k"},
                        clear=True):
            self.assertEqual(_resolve_provider(), "claude")

    def test_explicit_llm_provider_kimi(self):
        with patch.dict(os.environ,
                        {"LLM_PROVIDER": "kimi", "ANTHROPIC_API_KEY": "sk-a"},
                        clear=True):
            self.assertEqual(_resolve_provider(), "kimi")

    def test_explicit_llm_provider_case_insensitive(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "CLAUDE"}, clear=True):
            self.assertEqual(_resolve_provider(), "claude")

    def test_unknown_explicit_provider_falls_through_to_key_check(self):
        with patch.dict(os.environ,
                        {"LLM_PROVIDER": "bogus", "MOONSHOT_API_KEY": "sk-k"},
                        clear=True):
            self.assertEqual(_resolve_provider(), "kimi")


# ── LLMClient construction ────────────────────────────────────────────────────

class TestLLMClientConstruction(unittest.TestCase):

    def test_claude_provider_selected_from_env(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-a"}, clear=True):
            client = LLMClient()
        self.assertEqual(client.provider, "claude")

    def test_kimi_provider_selected_from_env(self):
        fake_kimi = MagicMock()
        with patch.dict(sys.modules, {"openai": _make_openai_mod(fake_kimi)}), \
             patch.dict(os.environ, {"MOONSHOT_API_KEY": "sk-k"}, clear=True):
            client = LLMClient()
        self.assertEqual(client.provider, "kimi")
        self.assertIs(client._kimi_client, fake_kimi)

    def test_kimi_missing_key_raises_runtime_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                LLMClient(provider="kimi")
        self.assertIn("MOONSHOT_API_KEY", str(ctx.exception))

    def test_fallback_enabled_via_env(self):
        with patch.dict(os.environ,
                        {"ANTHROPIC_API_KEY": "sk-a", "LLM_FALLBACK": "1"},
                        clear=True):
            client = LLMClient()
        self.assertTrue(client.enable_fallback)

    def test_fallback_disabled_by_default(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-a"}, clear=True):
            client = LLMClient()
        self.assertFalse(client.enable_fallback)

    def test_repr(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-a"}, clear=True):
            client = LLMClient()
        self.assertIn("claude", repr(client))
        self.assertIn("disabled", repr(client))


# ── Model names ───────────────────────────────────────────────────────────────

class TestModelNames(unittest.TestCase):

    def test_kimi_text_model_is_correct(self):
        self.assertEqual(_KIMI_TEXT_MODEL, "kimi-k2.7-code")

    def test_claude_text_model_is_correct(self):
        self.assertEqual(_CLAUDE_TEXT_MODEL, "claude-sonnet-4-6")

    def test_claude_chat_routes_correct_model(self):
        fake_anthropic = _make_anthropic_mod("Hello")
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-a"}, clear=True):
            client = LLMClient(provider="claude")
        with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
            client.chat([{"role": "user", "content": "Hi"}])
        call_kwargs = fake_anthropic.Anthropic.return_value.messages.create.call_args
        self.assertEqual(call_kwargs.kwargs["model"], "claude-sonnet-4-6")


# ── Fallback behavior ─────────────────────────────────────────────────────────

class TestFallback(unittest.TestCase):

    def test_chat_falls_back_to_kimi_on_claude_failure(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Kimi reply"))]
        fake_kimi_client = MagicMock()
        fake_kimi_client.chat.completions.create.return_value = mock_response

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-a"}, clear=True):
            client = LLMClient(provider="claude", enable_fallback=True)

        # Simulate fallback: kimi client created lazily via _get_kimi_client
        with patch.dict(os.environ, {"MOONSHOT_API_KEY": "sk-k"}), \
             patch.dict(sys.modules, {"openai": _make_openai_mod(fake_kimi_client)}), \
             patch.object(client, "_chat_claude", side_effect=Exception("Claude down")):
            result = client.chat([{"role": "user", "content": "Hi"}])

        self.assertEqual(result, "Kimi reply")

    def test_chat_raises_when_fallback_disabled(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-a"}, clear=True):
            client = LLMClient(provider="claude", enable_fallback=False)

        with patch.object(client, "_chat_claude", side_effect=Exception("Claude down")):
            with self.assertRaises(Exception):
                client.chat([{"role": "user", "content": "Hi"}])


if __name__ == "__main__":
    unittest.main()
