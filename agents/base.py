from abc import ABC, abstractmethod
from typing import Any, Dict

from llm.provider import LLMInterface


class AgentABC(ABC):
    """Base class for all benchmark agents.

    Every agent receives a task dict with the BIRD question context and
    returns a result dict with the predicted answer and metadata.

    The key EAG differentiator: agents should only send schema information
    to the LLM, never raw data rows. Data sovereignty is the core principle.

    Agents produce answers, not SQL. The evaluation pipeline compares
    agent answers against gold answers (from executing gold SQL).
    """

    def __init__(self, llm: LLMInterface):
        self.llm = llm

    @abstractmethod
    def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent on a single task.

        Args:
            task: dict with keys: db_id, question, evidence, SQL (gold), difficulty

        Returns:
            dict with keys:
                - answer: str — the agent's answer to the question
                - raw_output: str — full LLM output for debugging
                - usage: dict — token usage from the LLM
                - latency_ms: float — time taken in milliseconds
                - error: str | None — if something went wrong
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable agent name for logging."""
        ...


def extract_sql(raw: str) -> str:
    """Extract SQL from an LLM response that may contain markdown fences.

    This is an internal utility for agents that need to parse SQL from their
    own LLM output (e.g., for Execute[sql] tool calls). It is NOT used by
    the evaluation pipeline.
    """
    if "```sql" in raw:
        return raw.split("```sql")[-1].split("```")[0].strip()
    if "```" in raw:
        return raw.split("```")[-2].split("```")[0].strip()
    return raw.strip()
