import time
from typing import Any, Dict

from datasets.bird.loader import get_schema
from llm.provider import LLMInterface

from .base import AgentABC, extract_sql

SYSTEM_PROMPT = (
    "You are a SQLite expert using the Program-of-Thoughts (PoT) paradigm.\n"
    "Express your reasoning as a step-by-step computational program, then "
    "extract the final SQL query.\n\n"
    "Format:\n"
    "# Step 1: Understand what we need\n"
    "# Step 2: Identify relevant tables and columns\n"
    "# Step 3: Build the query\n"
    "FINAL_SQL: SELECT ..."
)


class POTAgent(AgentABC):
    """PoT (Program-of-Thoughts) agent stub.

    The LLM expresses its reasoning as a structured program, then outputs
    the final SQL. Similar to PAL but emphasizes declarative reasoning
    over imperative code generation.
    """

    def __init__(self, llm: LLMInterface):
        super().__init__(llm)

    @property
    def name(self) -> str:
        return "pot"

    def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        db_id = task["db_id"]
        question = task["question"]
        evidence = task.get("evidence", "")

        try:
            schema = get_schema(db_id)
        except FileNotFoundError as e:
            return {
                "answer": "", "raw_output": "", "usage": {}, "latency_ms": 0,
                "error": str(e),
            }

        user_prompt = f"Schema:\n{schema}\n\nQuestion: {question}"
        if evidence:
            user_prompt += f"\nEvidence: {evidence}"
        user_prompt += "\n\nExpress your reasoning as a program, then output FINAL_SQL:"

        start = time.perf_counter()
        try:
            raw = self.llm.generate(SYSTEM_PROMPT, user_prompt)
        except Exception as e:
            return {
                "answer": "", "raw_output": "", "usage": {}, "latency_ms": 0,
                "error": str(e),
            }
        latency_ms = (time.perf_counter() - start) * 1000

        if "FINAL_SQL:" in raw:
            answer = raw.split("FINAL_SQL:")[-1].strip().split("\n")[0].strip()
        else:
            answer = extract_sql(raw)

        return {
            "answer": answer,
            "raw_output": raw,
            "usage": self.llm.get_usage(),
            "latency_ms": latency_ms,
            "error": None,
        }
