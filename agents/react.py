import time
from typing import Any, Dict

from datasets.bird.loader import get_schema
from llm.provider import LLMInterface

from .base import AgentABC, extract_sql

SYSTEM_PROMPT = (
    "You are a SQLite expert. Given a database schema and a natural language question, "
    "output ONLY a valid SQLite query. No explanations. No markdown. "
    "If evidence/hints are provided, use them to interpret column values."
)

REACT_SYSTEM_PROMPT = (
    "You are a SQLite expert using the ReAct (Reasoning + Acting) paradigm.\n"
    "For each question:\n"
    "1. Thought: Analyze the question and schema to plan your approach.\n"
    "2. Action: Write the SQL query.\n"
    "3. Observation: (simulated — the query would return results)\n\n"
    "After your reasoning chain, output the final SQL query on a line that "
    "starts with 'FINAL_SQL:'. Only output valid SQLite syntax.\n"
    "Example format:\n"
    "Thought: I need to find...\n"
    "Action: SELECT ...\n"
    "FINAL_SQL: SELECT ..."
)


class ReActAgent(AgentABC):
    """ReAct (Reasoning + Acting) agent for text-to-SQL.

    Uses a multi-step prompt that encourages the LLM to reason about the
    schema before generating SQL. The 'Acting' phase is the SQL generation
    itself — no actual tool calls in this baseline.
    """

    def __init__(self, llm: LLMInterface, max_steps: int = 3):
        super().__init__(llm)
        self.max_steps = max_steps

    @property
    def name(self) -> str:
        return "react"

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

        user_prompt = self._build_prompt(schema, question, evidence)

        start = time.perf_counter()
        try:
            raw = self.llm.generate(REACT_SYSTEM_PROMPT, user_prompt)
        except Exception as e:
            return {
                "answer": "", "raw_output": "", "usage": {}, "latency_ms": 0,
                "error": str(e),
            }
        latency_ms = (time.perf_counter() - start) * 1000

        sql = self._extract_final_sql(raw)

        return {
            "answer": sql,
            "raw_output": raw,
            "usage": self.llm.get_usage(),
            "latency_ms": latency_ms,
            "error": None,
        }

    def _build_prompt(self, schema: str, question: str, evidence: str) -> str:
        parts = [f"Schema:\n{schema}", f"\nQuestion: {question}"]
        if evidence:
            parts.append(f"Evidence: {evidence}")
        parts.append(
            "\nReason step by step, then output your final SQL query "
            "on a line starting with FINAL_SQL:"
        )
        return "\n".join(parts)

    def _extract_final_sql(self, raw: str) -> str:
        if "FINAL_SQL:" in raw:
            sql = raw.split("FINAL_SQL:")[-1].strip()
            sql = sql.split("\n")[0].strip()
            return sql
        return extract_sql(raw)
