import time
from typing import Any, Dict

from datasets.bird.loader import get_schema
from llm.provider import LLMInterface

from .base import AgentABC, extract_sql

SYSTEM_PROMPT = (
    "You are a SQLite expert using the PAL (Program-Aided Language) paradigm.\n"
    "Given a database schema and a question, write a Python program that:\n"
    "1. Constructs the SQL query as a string variable `sql`\n"
    "2. The program should show your reasoning, but the final `sql` variable "
    "must contain ONLY valid SQLite.\n\n"
    "Output format:\n"
    "```python\n"
    "# Your reasoning here\n"
    "sql = \"SELECT ...\"\n"
    "```\n\n"
    "After the code block, output the SQL on a line starting with FINAL_SQL:"
)


class PALAgent(AgentABC):
    """PAL (Program-Aided Language) agent stub.

    Generates a Python program that constructs the SQL query. The program
    is not actually executed in this baseline — only the extracted SQL is used.
    Full execution sandbox is planned for a future iteration.
    """

    def __init__(self, llm: LLMInterface):
        super().__init__(llm)

    @property
    def name(self) -> str:
        return "pal"

    def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        db_id = task["db_id"]
        question = task["question"]
        evidence = task.get("evidence", "")

        try:
            schema = get_schema(db_id)
        except FileNotFoundError as e:
            return {"sql": "", "usage": {}, "latency_ms": 0, "error": str(e)}

        user_prompt = f"Schema:\n{schema}\n\nQuestion: {question}"
        if evidence:
            user_prompt += f"\nEvidence: {evidence}"
        user_prompt += "\n\nWrite a Python program that builds the SQL query."

        start = time.perf_counter()
        try:
            raw = self.llm.generate(SYSTEM_PROMPT, user_prompt)
        except Exception as e:
            return {"sql": "", "usage": {}, "latency_ms": 0, "error": str(e)}
        latency_ms = (time.perf_counter() - start) * 1000

        if "FINAL_SQL:" in raw:
            sql = raw.split("FINAL_SQL:")[-1].strip().split("\n")[0].strip()
        else:
            sql = extract_sql(raw)

        return {
            "sql": sql,
            "usage": self.llm.get_usage(),
            "latency_ms": latency_ms,
            "error": None,
        }
