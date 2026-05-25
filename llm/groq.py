import os
import time

from openai import OpenAI, RateLimitError

from .provider import LLMInterface


class GroqClient(LLMInterface):
    """Groq adapter using the OpenAI-compatible API.

    Defaults to gpt-oss-120b but accepts any model name available on Groq.
    """

    def __init__(self, model: str = "gpt-oss-120b"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = model
        self._last_usage: dict = {}

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stop: list[str] | None = None,
    ) -> str:
        max_retries = 3
        rate_limit_retries = 6
        rate_limit_attempt = 0
        for attempt in range(max_retries):
            try:
                kwargs = dict(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if stop:
                    kwargs["stop"] = stop
                res = self.client.chat.completions.create(**kwargs)
                self._last_usage = {
                    "prompt_tokens": res.usage.prompt_tokens,
                    "completion_tokens": res.usage.completion_tokens,
                    "total_tokens": res.usage.total_tokens,
                }
                return res.choices[0].message.content.strip()
            except RateLimitError:
                if rate_limit_attempt >= rate_limit_retries - 1:
                    raise
                wait = min(30, 5 * 2**rate_limit_attempt)
                time.sleep(wait)
                rate_limit_attempt += 1
            except Exception:
                if attempt == max_retries - 1:
                    raise
                wait = 2**attempt
                time.sleep(wait)

    def get_usage(self) -> dict:
        return self._last_usage.copy()
