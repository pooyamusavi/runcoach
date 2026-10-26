"""Thin LLM client abstraction.

Production code talks to Claude through the Anthropic SDK. Tests and offline runs use
``ScriptedLLM``, which returns pre-recorded responses with the same shape, so the agent
and extractor logic are fully testable without network access or an API key.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Block:
    """Minimal stand-in for an Anthropic content block."""

    type: str
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class Response:
    content: list[Block]
    stop_reason: str


class LLMClient(Protocol):
    def create(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        tool_choice: dict[str, Any] | None = None,
        max_tokens: int = 1024,
    ) -> Any: ...


class AnthropicLLM:
    """Production client backed by the Anthropic Messages API."""

    def __init__(self, api_key: str, model: str) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def create(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        tool_choice: dict[str, Any] | None = None,
        max_tokens: int = 1024,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "system": system,
            "messages": messages,
            "tools": list(tools),
            "max_tokens": max_tokens,
        }
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return self._client.messages.create(**kwargs)


class ScriptedLLM:
    """Deterministic client that replays queued responses; records every request."""

    def __init__(self, responses: Sequence[Response]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def create(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        tool_choice: dict[str, Any] | None = None,
        max_tokens: int = 1024,
    ) -> Response:
        self.requests.append(
            {"system": system, "messages": [dict(m) for m in messages], "tools": list(tools)}
        )
        if not self._responses:
            raise RuntimeError("ScriptedLLM ran out of responses")
        return self._responses.pop(0)
