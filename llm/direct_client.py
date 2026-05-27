import os
from pathlib import Path
from typing import Type

import litellm
import yaml
from pydantic import BaseModel

from .provider import ChatResponse, LLMInterface, ToolCall

litellm.enable_json_schema_validation = True

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def _load_direct_config(name: str) -> dict:
    path = CONFIGS_DIR / "models.yaml"
    with open(path) as f:
        configs = yaml.safe_load(f)
    direct = configs.get("direct")
    if not direct or name not in direct:
        available = list(direct.keys()) if direct else []
        raise ValueError(
            f"Direct model config '{name}' not found under 'direct' in {path}. "
            f"Available: {', '.join(available)}"
        )
    return direct[name]


class DirectLLMClient(LLMInterface):
    """LLM adapter that calls providers directly via the litellm SDK (no proxy).

    Configuration is loaded from the ``direct`` section of
    ``configs/models.yaml`` by name.  The model name must include the
    litellm provider prefix (e.g., ``groq/...``, ``openrouter/...``) so
    litellm can resolve the correct endpoint automatically.

    Required YAML keys per direct entry:
        model: str           — full litellm model name with provider prefix

    Optional YAML keys:
        api_key_env: str     — env var holding the provider API key (default: GROQ_API_KEY)
        temperature: float   — default sampling temperature (default: 0.0)
        max_completion_tokens: int — default max completion tokens (default: 4096)
        num_retries: int     — retry count for transient / 429 errors (default: 8)
    """

    def __init__(self, config_name: str):
        cfg = _load_direct_config(config_name)

        api_key_env = cfg.get("api_key_env", "GROQ_API_KEY")
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(
                f"{api_key_env} environment variable is not set "
                f"(required by direct model config '{config_name}')"
            )

        self.api_key = api_key
        self.model = cfg["model"]
        self.default_temperature = cfg.get("temperature", 0.0)
        self.default_max_completion_tokens = cfg.get("max_completion_tokens", 4096)
        self.num_retries = cfg.get("num_retries", 8)
        self._last_usage: dict = {}

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_completion_tokens: int | None = None,
        response_format: Type[BaseModel] | None = None,
    ) -> str:
        max_completion_tokens = (
            max_completion_tokens if max_completion_tokens is not None
            else self.default_max_completion_tokens
        )

        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=max_completion_tokens,
            num_retries=self.num_retries,
            api_key=self.api_key,
        )
        if response_format is not None:
            kwargs["response_format"] = response_format

        res = litellm.completion(**kwargs)
        choice = res.choices[0]
        msg = choice.message
        content = msg.content.strip() if msg.content else ""
        reasoning_tokens = (
            res.usage.completion_tokens_details.reasoning_tokens
            if res.usage and res.usage.completion_tokens_details
            else 0
        )
        self._last_usage = {
            "prompt_tokens": res.usage.prompt_tokens,
            "completion_tokens": res.usage.completion_tokens,
            "total_tokens": res.usage.total_tokens,
            "reasoning_tokens": reasoning_tokens or 0,
        }

        return content

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        max_completion_tokens: int | None = None,
        response_format: Type[BaseModel] | None = None,
    ) -> ChatResponse:
        max_completion_tokens = (
            max_completion_tokens if max_completion_tokens is not None
            else self.default_max_completion_tokens
        )

        kwargs = dict(
            model=self.model,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            num_retries=self.num_retries,
            api_key=self.api_key,
        )
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if response_format is not None:
            kwargs["response_format"] = response_format

        res = litellm.completion(**kwargs)
        choice = res.choices[0]
        msg = choice.message

        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                )
                for tc in msg.tool_calls
            ]

        reasoning_tokens = (
            res.usage.completion_tokens_details.reasoning_tokens
            if res.usage and res.usage.completion_tokens_details
            else 0
        )
        usage = {
            "prompt_tokens": res.usage.prompt_tokens,
            "completion_tokens": res.usage.completion_tokens,
            "total_tokens": res.usage.total_tokens,
            "reasoning_tokens": reasoning_tokens or 0,
        }
        self._last_usage = usage

        return ChatResponse(
            content=msg.content,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=choice.finish_reason or "stop",
            reasoning_content=getattr(msg, "reasoning_content", None),
        )

    def get_usage(self) -> dict:
        return self._last_usage.copy()
