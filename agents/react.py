import json
import sys
import time
from typing import Any

from llm.provider import LLMInterface
from tools.execute_sql import ExecuteSQLTool
from tools.get_schema import GetSchemaTool

from .base import AgentABC

MAX_ITERATIONS = 10

SYSTEM_PROMPT = """\
You are a data analyst that answers questions about databases. You MUST use the \
provided tools to explore the database and execute queries. NEVER guess or \
fabricate answers — every answer must come from query results.

You have two ways to submit your final answer:
1. Call the "finish" tool with your answer.
2. Respond with just the answer value (no tool calls).

## Rules

### Workflow
- You MUST call get_schema first to understand the database structure.
- You MUST execute at least one SQL query via execute_sql before answering.
- When a SQL Hint is provided, use it to construct your query.
- If a query returns an error, fix the SQL and try again.
- Write only valid SQLite syntax.

### SQL does the math
- Always write SQL that produces the FINAL computed answer directly.
- NEVER compute arithmetic from query results in your head. Have SQL do it.
- If you need a ratio, write: SELECT a * 1.0 / b — not two separate columns.
- If you need a difference, write: SELECT x - y — not two queries.
- If you need a percentage, write: SELECT (x - y) * 100.0 / y — not raw values.
- NEVER output unevaluated expressions like 2002/30459. SQL must compute the \
final number.

### Answer format
- The final answer must be the actual value the question asks for, NOT a SQL \
query and NOT a JSON object wrapping the answer.
- Return ONLY the specific value:
  - "Which year?" → just the year number (e.g., 2013, not {"year": 2013})
  - "Who?" → just the ID (e.g., 47273, not {"CustomerID": 47273, ...})
  - "How much?" → just the number (e.g., 1224.96)
  - "How many?" → just the count (e.g., 176)
  - "Which month?" → just the month number (e.g., 4 or "04")
  - "Yes/No" → "Yes" or "No"
- Do NOT wrap the answer in a JSON object or return extra columns from the \
result set alongside the answer.
- Do NOT return the full result row when the question asks for a single value.

### Complex multi-part questions
- Compute ALL parts in a single SQL query when possible.
- Return results as a JSON array in the order the question specifies.
- For percentages, differences, or comparisons between groups, compute them in \
SQL using CASE/IIF expressions or CTEs — never manually.

### Database awareness
- Pay close attention to column descriptions, value descriptions, and foreign \
key relationships in the schema — they tell you what each column means.
- For questions about totals or aggregates, use GROUP BY with SUM, AVG, COUNT \
etc. — do not query individual rows.
- For date columns stored as text (e.g., YYYYMM format), use SUBSTR or BETWEEN \
to filter by year or month.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_schema",
            "description": (
                "Get the enriched database schema with table structures, "
                "column descriptions, primary keys, and foreign key relationships."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "db_id": {
                        "type": "string",
                        "description": "Database identifier",
                    },
                },
                "required": ["db_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": (
                "Execute a SQL query against the database and return the results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "SQL query to execute (SQLite syntax)",
                    },
                    "db_id": {
                        "type": "string",
                        "description": "Database identifier",
                    },
                },
                "required": ["sql", "db_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Submit the final answer to the question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "description": (
                            "The final answer. Use a number for numeric "
                            'answers, "Yes"/"No" for boolean, a JSON array '
                            "for lists, or a plain string for text."
                        ),
                    },
                },
                "required": ["answer"],
            },
        },
    },
]

_indent = "    "


def _log_trace(msg: str) -> None:
    print(f"{_indent}{msg}")


class ReActAgent(AgentABC):
    """ReAct (Reasoning + Acting) agent for text-to-SQL.

    Uses native LLM tool calling with an iterative loop. The agent explores
    the schema via get_schema, writes and executes SQL via execute_sql, and
    terminates via either:
      - Path A: responding with structured JSON {"answer": ...} (no tool calls)
      - Path B: calling the finish tool with {answer: ...}
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

    MAX_EMPTY_RETRIES = 2

    def run(self, task: dict) -> dict:
        db_id = task["db_id"]
        question = task["question"]
        evidence = task.get("evidence", "")

        messages = self._build_messages(db_id, question, evidence)
        full_trace_parts: list[str] = []
        steps: list[dict] = []
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        start = time.perf_counter()

        iteration = 0
        empty_retries = 0
        while iteration < self.max_steps:
            try:
                response = self.llm.chat(
                    messages=messages,
                    tools=TOOLS,
                    max_completion_tokens=4096,
                )
            except Exception as exc:
                latency_ms = (time.perf_counter() - start) * 1000
                return {
                    "answer": "",
                    "raw_output": "\n".join(full_trace_parts),
                    "usage": total_usage,
                    "latency_ms": latency_ms,
                    "error": f"LLM generation failed: {exc}",
                    "steps": steps,
                }

            self._accumulate_usage(total_usage, response.usage)

            if response.content:
                full_trace_parts.append(f"[Assistant]: {response.content}")

            if not response.tool_calls:
                answer = response.content or ""
                if not answer.strip() and empty_retries < self.MAX_EMPTY_RETRIES:
                    empty_retries += 1
                    messages.append({"role": "assistant", "content": None})
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your previous response was empty. "
                            "Please provide the answer now using the finish tool "
                            "or respond with the final value."
                        ),
                    })
                    if self.trace:
                        _log_trace("Empty response — retrying")
                    iteration += 1
                    continue

                latency_ms = (time.perf_counter() - start) * 1000
                steps.append({
                    "step_type": "text_response",
                    "content": answer,
                })
                if self.trace:
                    _log_trace(f"Text response: {answer[:100]}")
                return {
                    "answer": answer,
                    "raw_output": "\n".join(full_trace_parts),
                    "usage": total_usage,
                    "latency_ms": latency_ms,
                    "error": None if answer.strip() else "Empty response after retries",
                    "steps": steps,
                }

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        },
                    }
                    for tc in response.tool_calls
                ],
            }
            messages.append(assistant_msg)

            full_trace_parts.append(
                "[Tool Calls]: "
                + json.dumps([
                    {"name": tc.name, "arguments": tc.arguments}
                    for tc in response.tool_calls
                ])
            )

            finished = False
            final_answer = None

            for tc in response.tool_calls:
                if tc.name == "finish":
                    try:
                        args = json.loads(tc.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    candidate = args.get("answer")

                    if candidate is not None and candidate != "":
                        final_answer = candidate
                        finished = True
                        steps.append({
                            "step_type": "action",
                            "thought": response.reasoning_content,
                            "action": "finish",
                            "action_input": args,
                            "observation": None,
                        })
                        if self.trace:
                            _log_trace(f"Finish: {final_answer}")
                        break

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": (
                            "Error: finish() requires a non-empty 'answer' "
                            "parameter. Please call finish again with your "
                            "final answer value."
                        ),
                    })
                    steps.append({
                        "step_type": "action",
                        "thought": response.reasoning_content,
                        "action": "finish",
                        "action_input": args,
                        "observation": "Error: missing answer parameter",
                    })
                    if self.trace:
                        _log_trace("Finish called without answer — retrying")
                    continue

                observation = self._execute_tool(tc.name, tc.arguments, db_id)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": observation,
                })

                full_trace_parts.append(f"[{tc.name} Result]: {observation}")

                step_record: dict[str, Any] = {
                    "step_type": "action",
                    "thought": response.reasoning_content,
                    "action": tc.name,
                    "action_input": tc.arguments,
                    "observation": observation,
                }
                steps.append(step_record)

                if self.trace:
                    _log_trace(f"--- Step {iteration + 1} ---")
                    if response.reasoning_content:
                        _log_trace(f"Thought: {response.reasoning_content}")
                    _log_trace(f"Action: {tc.name}")
                    obs_display = observation
                    if len(obs_display) > 200:
                        obs_display = obs_display[:200] + "..."
                    _log_trace(f"Observation: {obs_display}")
                    sys.stdout.flush()

            if finished:
                latency_ms = (time.perf_counter() - start) * 1000
                return {
                    "answer": final_answer,
                    "raw_output": "\n".join(full_trace_parts),
                    "usage": total_usage,
                    "latency_ms": latency_ms,
                    "error": None,
                    "steps": steps,
                }

            iteration += 1

        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "answer": "",
            "raw_output": "\n".join(full_trace_parts),
            "usage": total_usage,
            "latency_ms": latency_ms,
            "error": f"Agent exceeded {self.max_steps} iterations without finishing",
            "steps": steps,
        }

    def _build_messages(self, db_id: str, question: str, evidence: str) -> list[dict]:
        parts = [f"Database ID: {db_id}", f"Question: {question}"]
        if evidence:
            parts.append(f"SQL Hint: {evidence}")
        parts.append("\nStart by calling get_schema to explore the database structure.")
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(parts)},
        ]

    def _execute_tool(self, name: str, arguments_json: str, db_id: str) -> str:
        try:
            params = json.loads(arguments_json)
        except json.JSONDecodeError:
            return "Error: Invalid JSON in tool arguments."
        params.setdefault("db_id", db_id)

        if name == "get_schema":
            return self._schema_tool.run(params)
        if name == "execute_sql":
            return self._sql_tool.run(params)
        return f"Error: Unknown tool '{name}'"

    def _accumulate_usage(self, total: dict, usage: dict) -> None:
        total["prompt_tokens"] += usage.get("prompt_tokens", 0)
        total["completion_tokens"] += usage.get("completion_tokens", 0)
        total["total_tokens"] += usage.get("total_tokens", 0)
