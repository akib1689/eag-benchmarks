from typing import Any, Dict

from llm.provider import LLMInterface

from .base import AgentABC


class EAGAgent(AgentABC):
    """EAG (Execution-Augmented Generation) agent — stub.

    The core EAG idea: the LLM generates an executable *plan* (not raw SQL)
    against a schema-only contract. A local executor runs the plan against
    the real database. A verifier validates the plan before execution.
    Raw data never leaves the local environment.

    This stub returns the architecture skeleton. Full implementation is
    planned for a future task once the verifier module is designed.
    """

    def __init__(self, llm: LLMInterface):
        super().__init__(llm)

    @property
    def name(self) -> str:
        return "eag"

    def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError(
            "EAG agent is not yet implemented. "
            "Implement the plan-generator, local executor, and verifier first."
        )
