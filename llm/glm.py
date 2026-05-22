import os
import time

from openai import OpenAI

from .provider import LLMInterface


class GLMClient(LLMInterface):
    """GLM-5.1 adapter using the OpenAI-compatible API.

    This is a stub — fill in the correct base_url and model name once
    the GLM endpoint is available.
    """

    def __init__(self, model: str = "glm-5.1"):
        api_key = os.getenv("GLM_API_KEY")
        if not api_key:
            raise ValueError("GLM_API_KEY environment variable is not set")

        self.client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
        )
        self.model = model
        self._last_usage: dict = {}

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                res = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                self._last_usage = {
                    "prompt_tokens": res.usage.prompt_tokens,
                    "completion_tokens": res.usage.completion_tokens,
                    "total_tokens": res.usage.total_tokens,
                }
                return res.choices[0].message.content.strip()
            except Exception:
                if attempt == max_retries - 1:
                    raise
                wait = 2**attempt
                time.sleep(wait)

    def get_usage(self) -> dict:
        return self._last_usage.copy()
