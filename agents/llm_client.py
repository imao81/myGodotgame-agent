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
_DEEPSEEK_TEXT_MODEL = os.environ.get("DEEPSEEK_TEXT_MODEL", "deepseek-v4-pro")

# OpenAI-compatible provider endpoints
_OPENAI_COMPAT = {
    "kimi": ("MOONSHOT_API_KEY", "https://api.moonshot.cn/v1"),
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com"),
}


# ── Provider resolution ────────────────────────────────────────────────────────


def _resolve_provider() -> str:
    """Pick a provider based on LLM_PROVIDER env var or available API keys."""
    explicit = os.environ.get("LLM_PROVIDER", "").lower().strip()
    if explicit in ("claude", "kimi", "deepseek"):
        return explicit
    # Auto-detect: Claude takes priority when both keys are present
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude"
    if os.environ.get("MOONSHOT_API_KEY"):
        return "kimi"
    if os.environ.get("DEEPSEEK_API_KEY"):
        return "deepseek"
    return "claude"  # Will fail with a clear auth error at call time


def _fallback_candidates(provider: str) -> list[str]:
    """Other providers with a configured key, in preference order."""
    order = ["claude", "kimi", "deepseek"]
    key_of = {"claude": "ANTHROPIC_API_KEY", "kimi": "MOONSHOT_API_KEY",
              "deepseek": "DEEPSEEK_API_KEY"}
    return [p for p in order if p != provider and os.environ.get(key_of[p])]


# ── LLMClient ─────────────────────────────────────────────────────────────────


class LLMClient:
    """Provider-agnostic LLM wrapper for text and vision requests.

    Args:
        provider:        "claude" | "kimi" (default: env auto-detection)
        model:           override the default text model — used by the router
                         to pin a file's review to a specific model tier
        enable_fallback: also try the other provider on failure

    After each successful chat() call, `last_usage` holds token counts
    ({"prompt_tokens": int, "completion_tokens": int}); None after a failure.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        enable_fallback: bool = False,
    ):
        self.provider = (provider or _resolve_provider()).lower()
        self._model = model
        self.last_usage: Optional[dict] = None
        self.enable_fallback = (
            enable_fallback or os.environ.get("LLM_FALLBACK", "").strip() == "1"
        )
        # OpenAI-compatible clients are created eagerly when selected, lazily
        # on first fallback use. Kimi: see _get_kimi_client().
        self._kimi_client = None
        self._deepseek_client = None
        if self.provider in ("kimi", "deepseek"):
            self._get_compat_client(self.provider)  # validate key at construction

    # ── Internal: Claude ───────────────────────────────────────────────────────

    def _chat_claude(self, messages: list[MessageParam], max_tokens: int) -> str:
        import anthropic

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=self._model or _CLAUDE_TEXT_MODEL,
            max_tokens=max_tokens,
            messages=messages,
        )
        u = getattr(response, "usage", None)
        self.last_usage = (
            {"prompt_tokens": u.input_tokens, "completion_tokens": u.output_tokens}
            if u else None
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

    def _get_compat_client(self, provider: str):
        """Return the cached OpenAI-compatible client, creating it lazily."""
        env_var, base_url = _OPENAI_COMPAT[provider]
        attr = f"_{provider}_client"
        client = getattr(self, attr)
        if client is None:
            api_key = os.environ.get(env_var)
            if not api_key:
                raise RuntimeError(
                    f"{env_var} is not set. "
                    f"Set it or switch LLM_PROVIDER to another provider."
                )
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url)
            setattr(self, attr, client)
        return client

    def _get_kimi_client(self):
        """Return the cached Kimi client (kept for backward compatibility)."""
        return self._get_compat_client("kimi")

    def _chat_compat(self, provider: str, default_model: str,
                     messages: list[ChatCompletionMessageParam], max_tokens: int) -> str:
        extra: dict = {}
        # DeepSeek V4 enables thinking mode by default — the reasoning chain
        # eats the entire token budget and content comes back EMPTY (observed:
        # 8192/8192 completion tokens, zero content). Opt out per-request
        # unless DEEPSEEK_THINKING=1.
        if provider == "deepseek" and os.environ.get("DEEPSEEK_THINKING", "0") != "1":
            extra["extra_body"] = {"thinking": {"type": "disabled"}}
        response = self._get_compat_client(provider).chat.completions.create(
            model=self._model or default_model,
            max_tokens=max_tokens,
            messages=messages,
            **extra,
        )
        u = getattr(response, "usage", None)
        self.last_usage = (
            {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens}
            if u else None
        )
        return response.choices[0].message.content or ""

    def _chat_kimi(self, messages: list[ChatCompletionMessageParam], max_tokens: int) -> str:
        return self._chat_compat("kimi", _KIMI_TEXT_MODEL, messages, max_tokens)

    def _chat_deepseek(self, messages: list[ChatCompletionMessageParam], max_tokens: int) -> str:
        return self._chat_compat("deepseek", _DEEPSEEK_TEXT_MODEL, messages, max_tokens)

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
        dispatch = {"claude": self._chat_claude, "kimi": self._chat_kimi,
                    "deepseek": self._chat_deepseek}
        if self.provider not in dispatch:
            raise ValueError(
                f"Unknown LLM provider: {self.provider!r}. "
                f"Use one of: {sorted(dispatch)}."
            )
        try:
            return dispatch[self.provider](messages, max_tokens)
        except Exception:
            if self.enable_fallback:
                for candidate in _fallback_candidates(self.provider):
                    if candidate in dispatch:
                        try:
                            return dispatch[candidate](messages, max_tokens)
                        except Exception:
                            continue
            self.last_usage = None
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
                f"Provider {self.provider!r} has no vision route configured "
                f"(supported: {sorted(dispatch)})."
            )
        try:
            return dispatch[self.provider](image_b64, media_type, prompt, max_tokens)
        except Exception:
            if self.enable_fallback:
                for candidate in _fallback_candidates(self.provider):
                    if candidate in dispatch:
                        try:
                            return dispatch[candidate](image_b64, media_type,
                                                       prompt, max_tokens)
                        except Exception:
                            continue
            raise

    def __repr__(self) -> str:
        fallback_str = "enabled" if self.enable_fallback else "disabled"
        return f"LLMClient(provider={self.provider!r}, fallback={fallback_str})"
