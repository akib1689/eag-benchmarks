from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Type

from pydantic import BaseModel


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class ChatResponse:
    content: str | None
    tool_calls: list[ToolCall] | None
    usage: dict
    finish_reason: str
    reasoning_content: str | None = None


class LLMInterface(ABC):
    """Abstract interface for all LLM providers.

    Every adapter must implement `generate` and `get_usage` so that the
    benchmark harness can swap Groq <-> GLM (or any future provider)
    with zero code changes — only config.
    """

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_completion_tokens: int = 4096,
        response_format: Type[BaseModel] | None = None,
    ) -> str:
        """Generate a text completion. Must handle retry logic internally."""
        ...

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        max_completion_tokens: int | None = None,
        response_format: Type[BaseModel] | None = None,
    ) -> ChatResponse:
        """Generate a chat completion with optional native tool calling.

        Args:
            messages: OpenAI-format message list (system, user, assistant, tool).
            tools: OpenAI-format tool definitions (function calling).
            tool_choice: "auto" | "none" | "required" | dict, see OpenAI docs.
            max_completion_tokens: Override default max completion tokens.

        Returns:
            ChatResponse with content, tool_calls, usage, and finish_reason.
        """
        ...

    @abstractmethod
    def get_usage(self) -> dict:
        """Return {prompt_tokens, completion_tokens, total_tokens} for the last call."""
        ...
