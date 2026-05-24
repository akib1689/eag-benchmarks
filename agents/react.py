import json
import re
import sys
import time
from typing import Any, Dict

from llm.provider import LLMInterface
from tools.execute_sql import ExecuteSQLTool
from tools.get_schema import GetSchemaTool

from .base import AgentABC

MAX_ITERATIONS = 10
STOP_SEQUENCES = ["\nObservation:", "\nObservation: "]

SYSTEM_PROMPT = """\
You are a data analyst that answers questions about databases using the ReAct \
(Reasoning + Acting) paradigm. You MUST use tools to explore the database and \
execute queries. NEVER guess or fabricate answers — every answer must come from \
query results.

Available tools:
- GetSchema: Get the enriched database schema with table structures, column \
descriptions, primary keys, and foreign key relationships. \
Params: {{"db_id": "<database id>"}}
- Execute: Execute a SQL query against the database and see the results. \
Params: {{"sql": "<SQL query>", "db_id": "<database id>"}}

You must respond in EXACTLY this format for each step:

Thought: <your reasoning about what to do next>
Action: <tool name: GetSchema or Execute>
Action Input: <JSON object with the parameters>

When you have the final answer from query results, respond with:

Thought: I now know the final answer.
Action: Finish
Action Input: <the final answer as a plain value>

Rules:
- You MUST call GetSchema first to understand the database structure.
- You MUST execute at least one SQL query via Execute before answering.
- Do NOT guess or fabricate answer values. Every answer must be grounded in \
query results you have observed.
- Write only valid SQLite syntax.
- The final answer must be the actual answer value, NOT a SQL query.
- For numeric answers, output the number only.
- For Yes/No questions, output Yes or No.
- For list answers, output comma-separated values.
- Pay close attention to column descriptions, value descriptions, and foreign \
key relationships in the schema — they tell you what each column means.
- When a SQL Hint is provided, use it to construct your query.
- If a query returns an error, fix the SQL and try again.
- For questions about totals or aggregates, use GROUP BY with SUM, AVG, COUNT \
etc. — do not query individual rows.
"""

FEW_SHOT_1 = """\
Example 1 (simple lookup):

Question: How many employees are in the Engineering department?
Thought: I need to see the database schema first.
Action: GetSchema
Action Input: {{"db_id": "company_db"}}
Observation: Database: company_db

Table: employee
  id (integer) [PK] — employee identifier
  name (text) — employee name
  department (text) — department name
  salary (real) — annual salary in USD

Thought: I can count employees in the Engineering department.
Action: Execute
Action Input: {{"sql": "SELECT COUNT(*) FROM employee WHERE department = \
'Engineering'", "db_id": "company_db"}}
Observation: COUNT(*)
-----------
200

Total rows: 1

Thought: I now know the final answer.
Action: Finish
Action Input: 200
"""

FEW_SHOT_2 = """\
Example 2 (aggregation with GROUP BY):

Question: Which department has the highest total salary?
Thought: I need the schema first.
Action: GetSchema
Action Input: {{"db_id": "company_db"}}
Observation: Database: company_db

Table: employee
  id (integer) [PK] — employee identifier
  name (text) — employee name
  department (text) — department name
  salary (real) — annual salary in USD

Thought: I need to sum salaries per department and find the max.
Action: Execute
Action Input: {{"sql": "SELECT department, SUM(salary) as total FROM employee \
GROUP BY department ORDER BY total DESC LIMIT 1", "db_id": "company_db"}}
Observation: department  total
------------ -----------
Engineering   15000000

Total rows: 1

Thought: I now know the final answer.
Action: Finish
Action Input: Engineering
"""

CORRECTION_PROMPT = (
    "Your previous response did not follow the required format. "
    "You must respond with:\n"
    "Thought: <reasoning>\n"
    "Action: <GetSchema, Execute, or Finish>\n"
    "Action Input: <JSON params or final answer>\n\n"
    "Please try again."
)

_indent = "    "


def _log_trace(msg: str) -> None:
    print(f"{_indent}{msg}")


def _extract_thought(response: str) -> str:
    match = re.search(r"Thought:\s*(.+?)(?:\n|$)", response, re.DOTALL)
    return match.group(1).strip() if match else ""


class ReActAgent(AgentABC):
    """ReAct (Reasoning + Acting) agent for text-to-SQL.

    Uses an iterative Thought -> Action -> Observation loop with real tool
    execution. The agent explores the schema via GetSchema, writes and
    executes SQL via Execute, and terminates with Finish[answer].
    """

    def __init__(self, llm: LLMInterface, max_steps: int = MAX_ITERATIONS):
        super().__init__(llm)
        self.max_steps = max_steps
        self._sql_tool = ExecuteSQLTool()
        self._schema_tool = GetSchemaTool()
        self.trace: bool = False

    @property
    def name(self) -> str:
        return "react"

    def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        db_id = task["db_id"]
        question = task["question"]
        evidence = task.get("evidence", "")

        system = SYSTEM_PROMPT + "\n" + FEW_SHOT_1 + "\n" + FEW_SHOT_2
        user = self._build_user_prompt(question, evidence, db_id)

        scratchpad = ""
        full_trace_parts: list[str] = []
        steps: list[Dict[str, Any]] = []
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        start = time.perf_counter()

        for iteration in range(self.max_steps):
            current_prompt = user + scratchpad
            try:
                response = self.llm.generate(
                    system, current_prompt,
                    max_tokens=2048,
                    stop=STOP_SEQUENCES,
                )
            except Exception as e:
                full_trace = "\n".join(full_trace_parts) if full_trace_parts else scratchpad
                latency_ms = (time.perf_counter() - start) * 1000
                return {
                    "answer": "",
                    "raw_output": full_trace,
                    "usage": total_usage,
                    "latency_ms": latency_ms,
                    "error": str(e),
                    "steps": steps,
                }

            self._accumulate_usage(total_usage)

            parsed = self._parse_response(response)
            full_trace_parts.append(response)

            action = parsed["action"]
            action_input = parsed["action_input"]
            thought = _extract_thought(response)

            step_record: Dict[str, Any] = {
                "iteration": iteration + 1,
                "thought": thought,
                "action": action,
                "action_input": action_input,
            }

            if self.trace:
                _log_trace(f"--- Step {iteration + 1} ---")
                if thought:
                    _log_trace(f"Thought: {thought}")

            if action == "Finish":
                if self.trace:
                    _log_trace(f"Finish: {action_input}")
                step_record["observation"] = None
                steps.append(step_record)
                latency_ms = (time.perf_counter() - start) * 1000
                return {
                    "answer": action_input,
                    "raw_output": "\n".join(full_trace_parts),
                    "usage": total_usage,
                    "latency_ms": latency_ms,
                    "error": None,
                    "steps": steps,
                }

            if action in ("GetSchema", "Execute"):
                if self.trace:
                    _log_trace(f"Action: {action}")
                    display_input = action_input
                    if len(display_input) > 120:
                        display_input = display_input[:120] + "..."
                    _log_trace(f"Action Input: {display_input}")

                observation = self._execute_tool(action, action_input, db_id)
                scratchpad += f"\n{response}\nObservation: {observation}\n"
                full_trace_parts.append(f"Observation: {observation}")

                if self.trace:
                    obs_display = observation
                    if len(obs_display) > 200:
                        obs_display = obs_display[:200] + "..."
                    _log_trace(f"Observation: {obs_display}")

                step_record["observation"] = observation
            else:
                correction_obs = (
                    f"Error: Unknown action '{action}'. "
                    "Available actions: GetSchema, Execute, Finish."
                )
                if self.trace:
                    _log_trace(f"Action: {action} (INVALID)")
                    _log_trace(f"Observation: {correction_obs}")
                scratchpad += f"\n{response}\nObservation: {correction_obs}\n"
                full_trace_parts.append(f"Observation: {correction_obs}")
                step_record["observation"] = correction_obs

            if not parsed["valid"]:
                if self.trace:
                    _log_trace("Format error — injecting correction prompt")
                scratchpad += f"\n{CORRECTION_PROMPT}\n"
                full_trace_parts.append(CORRECTION_PROMPT)

            steps.append(step_record)

            if self.trace:
                sys.stdout.flush()

        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "answer": "",
            "raw_output": "\n".join(full_trace_parts),
            "usage": total_usage,
            "latency_ms": latency_ms,
            "error": f"Agent exceeded {self.max_steps} iterations without finishing",
            "steps": steps,
        }

    def _build_user_prompt(self, question: str, evidence: str, db_id: str) -> str:
        parts = [f"Database ID: {db_id}", f"Question: {question}"]
        if evidence:
            parts.append(f"SQL Hint: {evidence}")
        parts.append("\nStart by calling GetSchema to explore the database structure.")
        return "\n".join(parts)

    def _parse_response(self, response: str) -> Dict[str, Any]:
        action_match = re.search(
            r"Action:\s*(GetSchema|Execute|Finish)\s*\n",
            response,
            re.IGNORECASE,
        )
        input_match = re.search(
            r"Action Input:\s*(.+?)(?:\n|$)",
            response,
            re.IGNORECASE | re.DOTALL,
        )

        if not action_match:
            return {
                "action": None,
                "action_input": "",
                "valid": False,
            }

        action = action_match.group(1).strip()
        action_input = input_match.group(1).strip() if input_match else ""

        return {
            "action": action,
            "action_input": action_input,
            "valid": True,
        }

    def _execute_tool(self, action: str, action_input: str, db_id: str) -> str:
        params = self._parse_action_input(action_input, db_id)

        if action == "GetSchema":
            return self._schema_tool.run(params)
        if action == "Execute":
            return self._sql_tool.run(params)
        return f"Error: Unknown action '{action}'"

    def _parse_action_input(self, raw: str, db_id: str) -> Dict[str, Any]:
        try:
            params = json.loads(raw)
            if not isinstance(params, dict):
                params = {}
        except (json.JSONDecodeError, ValueError):
            params = {}

        params.setdefault("db_id", db_id)

        if "sql" not in params and "SQL" in raw:
            sql_match = re.search(r"(?:sql|SQL)[:\s]+(.+?)(?:\}|$)", raw, re.DOTALL)
            if sql_match:
                params["sql"] = sql_match.group(1).strip().rstrip("}")

        return params

    def _accumulate_usage(self, total: Dict[str, int]) -> None:
        usage = self.llm.get_usage()
        total["prompt_tokens"] += usage.get("prompt_tokens", 0)
        total["completion_tokens"] += usage.get("completion_tokens", 0)
        total["total_tokens"] += usage.get("total_tokens", 0)
