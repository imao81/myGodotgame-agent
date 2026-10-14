"""Unified LLM client for harvester-agent.

Supports two providers:
  - claude : Anthropic Claude       (requires ANTHROPIC_API_KEY)
  - kimi   : Moonshot AI Kimi       (requires MOONSHOT_API_KEY)
             Uses the OpenAI-compatible API at https://api.moonshot.cn/v1

Configuration via environment variables:
  LLM_PROVIDER=claude|kimi   — select provider explicitly
                               (auto-detected from available API keys if unset)
  LLM_FALLBACK=1             — on API failure, automatically retry with
                               the other provider

Auto-detection order (when LLM_PROVIDER is unset):
  1. ANTHROPIC_API_KEY present → claude
  2. MOONSHOT_API_KEY present  → kimi
  3. Neither present           → claude (fails with clear auth error at call time)

Default models (env-overridable via KIMI_TEXT_MODEL / KIMI_VISION_MODEL):
  Claude text  : claude-sonnet-4-6
  Claude vision: claude-sonnet-4-6
  Kimi text    : kimi-k2.7-code
  Kimi vision  : moonshot-v1-8k-vision-preview

Usage:
    from agents.llm_client import LLMClient

    # Auto-detect provider from env
    client = LLMClient()

    # Force a specific provider
    client = LLMClient(provider="kimi")
    client = LLMClient(provider="claude")

    # Enable fallback to the other provider on failure
    client = LLMClient(enable_fallback=True)

    # Text chat
    text = client.chat([{"role": "user", "content": "Hello"}], max_tokens=512)

    # Vision (image + text)
    text = client.chat_vision(image_b64, "image/png", "What do you see?", max_tokens=512)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, cast

if TYPE_CHECKING:
    from anthropic.types import (
        ImageBlockParam,
        MessageParam,
        TextBlockParam,
    )
    from openai.types.chat import (
        ChatCompletionContentPartImageParam,
        ChatCompletionContentPartTextParam,
        ChatCompletionMessageParam,
    )

# ── Default models ─────────────────────────────────────────────────────────────

_CLAUDE_TEXT_MODEL = "claude-sonnet-4-6"
_CLAUDE_VISION_MODEL = "claude-sonnet-4-6"
# Model names change over time — env vars override the defaults
_KIMI_TEXT_MODEL = os.environ.get("KIMI_TEXT_MODEL", "kimi-k2.7-code")
_KIMI_VISION_MODEL = os.environ.get("KIMI_VISION_MODEL", "moonshot-v1-8k-vision-preview")


# ── Provider resolution ────────────────────────────────────────────────────────


def _resolve_provider() -> str:
    """Pick a provider based on LLM_PROVIDER env var or available API keys."""
    explicit = os.environ.get("LLM_PROVIDER", "").lower().strip()
    if explicit in ("claude", "kimi"):
        return explicit
    # Auto-detect: Claude takes priority when both keys are present
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude"
    if os.environ.get("MOONSHOT_API_KEY"):
        return "kimi"
    return "claude"  # Will fail with a clear auth error at call time


def _other_provider(provider: str) -> str:
    return "kimi" if provider == "claude" else "claude"


# ── LLMClient ─────────────────────────────────────────────────────────────────


class LLMClient:
    """Provider-agnostic LLM wrapper for text and vision requests."""

    def __init__(
        self,
        provider: Optional[str] = None,
        enable_fallback: bool = False,
    ):
        self.provider = (provider or _resolve_provider()).lower()
        self.enable_fallback = (
            enable_fallback or os.environ.get("LLM_FALLBACK", "").strip() == "1"
        )
        # Kimi client is created eagerly when provider=kimi, lazily on first
        # fallback use when provider=claude. See _get_kimi_client().
        self._kimi_client = None
        if self.provider == "kimi":
            self._get_kimi_client()  # validate key + create client at construction

    # ── Internal: Claude ───────────────────────────────────────────────────────

    def _chat_claude(self, messages: list[MessageParam], max_tokens: int) -> str:
        import anthropic

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=_CLAUDE_TEXT_MODEL,
            max_tokens=max_tokens,
            messages=messages,
        )
        block = next(
            b for b in response.content if isinstance(b, anthropic.types.TextBlock)
        )
        return block.text or ""

    def _vision_claude(
        self, image_b64: str, media_type: str, prompt: str, max_tokens: int
    ) -> str:
        import anthropic

        client = anthropic.Anthropic()
        image_block = cast(
            "ImageBlockParam",
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_b64,
                },
            },
        )
        text_block = cast("TextBlockParam", {"type": "text", "text": prompt})
        vision_messages: list[MessageParam] = [
            {
                "role": "user",
                "content": [image_block, text_block],
            }
        ]
        response = client.messages.create(
            model=_CLAUDE_VISION_MODEL,
            max_tokens=max_tokens,
            messages=vision_messages,
        )
        block = next(
            b for b in response.content if isinstance(b, anthropic.types.TextBlock)
        )
        return block.text or ""

    # ── Internal: Kimi (OpenAI-compatible) ────────────────────────────────────

    def _get_kimi_client(self):
        """Return the cached Kimi client, creating it lazily if needed (e.g. fallback)."""
        if self._kimi_client is None:
            api_key = os.environ.get("MOONSHOT_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "MOONSHOT_API_KEY is not set. "
                    "Set it or switch to LLM_PROVIDER=claude."
                )
            from openai import OpenAI

            self._kimi_client = OpenAI(
                api_key=api_key,
                base_url="https://api.moonshot.cn/v1",
            )
        return self._kimi_client

    def _chat_kimi(self, messages: list[ChatCompletionMessageParam], max_tokens: int) -> str:
        response = self._get_kimi_client().chat.completions.create(
            model=_KIMI_TEXT_MODEL,
            max_tokens=max_tokens,
            messages=messages,
        )
        return response.choices[0].message.content or ""

    def _vision_kimi(
        self, image_b64: str, media_type: str, prompt: str, max_tokens: int
    ) -> str:
        image_part = cast(
            "ChatCompletionContentPartImageParam",
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{image_b64}",
                },
            },
        )
        text_part = cast(
            "ChatCompletionContentPartTextParam", {"type": "text", "text": prompt}
        )
        vision_messages: list[ChatCompletionMessageParam] = [
            {
                "role": "user",
                "content": [image_part, text_part],
            }
        ]
        response = self._get_kimi_client().chat.completions.create(
            model=_KIMI_VISION_MODEL,
            max_tokens=max_tokens,
            messages=vision_messages,
        )
        return response.choices[0].message.content or ""

    # ── Public API ─────────────────────────────────────────────────────────────

    def chat(self, messages: list[ChatCompletionMessageParam], max_tokens: int = 1024) -> str:
        """Send a text chat and return the response as a string."""
        dispatch = {"claude": self._chat_claude, "kimi": self._chat_kimi}
        if self.provider not in dispatch:
            raise ValueError(
                f"Unknown LLM provider: {self.provider!r}. Use 'claude' or 'kimi'."
            )
        try:
            return dispatch[self.provider](messages, max_tokens)
        except Exception:
            if self.enable_fallback:
                fallback = _other_provider(self.provider)
                if fallback in dispatch:
                    return dispatch[fallback](messages, max_tokens)
            raise

    def chat_vision(
        self,
        image_b64: str,
        media_type: str,
        prompt: str,
        max_tokens: int = 512,
    ) -> str:
        """Send an image + text prompt and return the response as a string."""
        dispatch = {"claude": self._vision_claude, "kimi": self._vision_kimi}
        if self.provider not in dispatch:
            raise ValueError(
                f"Unknown LLM provider: {self.provider!r}. Use 'claude' or 'kimi'."
            )
        try:
            return dispatch[self.provider](image_b64, media_type, prompt, max_tokens)
        except Exception:
            if self.enable_fallback:
                fallback = _other_provider(self.provider)
                if fallback in dispatch:
                    return dispatch[fallback](image_b64, media_type, prompt, max_tokens)
            raise

    def __repr__(self) -> str:
        fallback_str = "enabled" if self.enable_fallback else "disabled"
        return f"LLMClient(provider={self.provider!r}, fallback={fallback_str})"
