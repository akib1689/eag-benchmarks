from abc import ABC, abstractmethod
from typing import Type

from pydantic import BaseModel


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
    def get_usage(self) -> dict:
        """Return {prompt_tokens, completion_tokens, total_tokens} for the last call."""
        ...
