import os
from pathlib import Path
from typing import Type

import litellm
import yaml
from pydantic import BaseModel

from .provider import LLMInterface

litellm.enable_json_schema_validation = True
# litellm._turn_on_debug()

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_model_config(name: str) -> dict:
    """Load a named model config from configs/models.yaml.

    Each entry must include at least ``model`` and may include
    ``api_key_env``, ``base_url``, ``temperature``, ``max_tokens``.
    """
    path = CONFIGS_DIR / "models.yaml"
    with open(path) as f:
        configs = yaml.safe_load(f)
    if name not in configs:
        raise ValueError(
            f"Model config '{name}' not found in {path}. "
            f"Available: {', '.join(configs.keys())}"
        )
    return configs[name]


class LiteLLMClient(LLMInterface):
    """Unified LLM adapter that talks to a LiteLLM proxy via the litellm SDK.

    Configuration is loaded from ``configs/models.yaml`` by name.  The
    proxy handles provider routing, key rotation across multiple Groq
    accounts, and rate-limit retries.

    The model name is prefixed with ``litellm_proxy/`` so the SDK routes
    the request through the proxy instead of calling providers directly.

    Required YAML keys per model entry:
        model: str           — model alias as registered on the proxy

    Optional YAML keys:
        base_url: str        — proxy endpoint (env: LITELLM_BASE_URL, default: http://localhost:4000)
        api_key_env: str     — env var holding the virtual key (default: LITELLM_API_KEY)
        temperature: float   — default sampling temperature (default: 0.0)
        max_tokens: int      — default max completion tokens (default: 1024)
        num_retries: int     — retry count for transient / 429 errors (default: 8)
    """

    def __init__(self, config_name: str):
        cfg = load_model_config(config_name)

        api_key_env = cfg.get("api_key_env", "LITELLM_API_KEY")
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(
                f"{api_key_env} environment variable is not set "
                f"(required by model config '{config_name}')"
            )

        self.api_key = api_key
        self.api_base = cfg.get("base_url", os.getenv("LITELLM_BASE_URL", "http://localhost:4000"))
        self.model = f"litellm_proxy/{cfg['model']}"
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
            api_base=self.api_base,
        )
        if response_format is not None:
            kwargs["response_format"] = response_format

        res = litellm.completion(**kwargs)
        choice = res.choices[0]
        msg = choice.message
        content = msg.content.strip()
        self._last_usage = {
            "prompt_tokens": res.usage.prompt_tokens,
            "completion_tokens": res.usage.completion_tokens,
            "total_tokens": res.usage.total_tokens,
        }

        return content

    def get_usage(self) -> dict:
        return self._last_usage.copy()
