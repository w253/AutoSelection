from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ReasoningResponse:
    """Structured response from a reasoning/thinking model."""
    thinking: str       # The model's chain-of-thought reasoning
    answer: str         # The final answer after reasoning
    raw: str            # Full raw response content


def _split_thinking(raw: str) -> Tuple[str, str]:
    """Split a response into (thinking, answer) parts.

    Handles two formats:
      1. ``<think>...</think>`` tags in content (DeepSeek-R1 style)
      2. Plain text without thinking tags (returns empty thinking)
    """
    if not raw:
        return "", ""

    # Pattern: <think>...</think> followed by answer
    m = re.search(r"<think>(.*?)</think>(.*)", raw, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    # No thinking tags — entire content is the answer
    return "", raw.strip()


class LLMClient:
    """Minimal OpenAI-compatible chat-completion client with reasoning support."""

    DEFAULT_TIMEOUT = 1000  # seconds — prevents hanging on dead connections

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o",
        timeout: float = 1000,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        try:
            import openai  # noqa: F811
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for Recipe Sandbox agents. "
                "Install it with: pip install openai"
            ) from exc

        kwargs: Dict[str, Any] = {"timeout": timeout, "max_retries": max_retries}
        if api_key is not None:
            kwargs["api_key"] = api_key
        if base_url is not None:
            kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**kwargs)

    def chat(self, prompt: str, temperature: float = 0.6) -> str:
        """Send a chat request and return the answer (thinking stripped)."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        raw = response.choices[0].message.content or ""

        # If the model has a separate reasoning_content field (Volcengine Ark),
        # the main content is already the answer.
        reasoning_content = getattr(response.choices[0].message, "reasoning_content", None)
        if reasoning_content:
            return raw.strip()

        # Otherwise strip <think>...</think> tags if present
        _, answer = _split_thinking(raw)
        return answer

    def chat_with_reasoning(self, prompt: str, temperature: float = 0.6) -> ReasoningResponse:
        """Send a chat request and return both thinking and answer parts.

        Use this when you need to log or analyze the model's reasoning process.
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        raw = response.choices[0].message.content or ""

        # Check for dedicated reasoning_content field first
        reasoning_content = getattr(response.choices[0].message, "reasoning_content", None)
        if reasoning_content:
            return ReasoningResponse(
                thinking=reasoning_content.strip(),
                answer=raw.strip(),
                raw=raw,
            )

        # Fall back to <think> tag parsing
        thinking, answer = _split_thinking(raw)
        return ReasoningResponse(thinking=thinking, answer=answer, raw=raw)
