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
VALID_ACTIONS = {"GetSchema", "Execute", "Finish"}

SYSTEM_PROMPT = """\
You are a data analyst that answers questions about databases using the ReAct \
(Reasoning + Acting) paradigm. You MUST use tools to explore the database and \
execute queries. NEVER guess or fabricate answers — every answer must come from \
query results.

Available tools:
- GetSchema: Get the enriched database schema with table structures, column \
descriptions, primary keys, and foreign key relationships.
- Execute: Execute a SQL query against the database and see the results.

You MUST respond with a single JSON object on every turn. No other text outside \
the JSON. The JSON schema is:

{
  "thought": "your reasoning about what to do next",
  "action": "GetSchema | Execute | Finish",
  "action_input": { ... }
}

action_input shapes by action:
- GetSchema: {"db_id": "<database id>"}
- Execute:   {"sql": "<SQL query>", "db_id": "<database id>"}
- Finish:    {"answer": <the final answer as a plain value>}

Rules:
- You MUST call GetSchema first to understand the database structure.
- You MUST execute at least one SQL query via Execute before answering.
- Do NOT guess or fabricate answer values. Every answer must be grounded in \
query results you have observed.
- Write only valid SQLite syntax.
- The final answer must be the actual answer value, NOT a SQL query.
- For numeric answers, use a number (not a string).
- For Yes/No questions, use "Yes" or "No".
- For list answers, use a JSON array.
- Pay close attention to column descriptions, value descriptions, and foreign \
key relationships in the schema — they tell you what each column means.
- When a SQL Hint is provided, use it to construct your query.
- If a query returns an error, fix the SQL and try again.
- For questions about totals or aggregates, use GROUP BY with SUM, AVG, COUNT \
etc. — do not query individual rows.
"""

FEW_SHOT_1 = """\
Example 1 (simple lookup):

Database ID: company_db
Question: How many employees are in the Engineering department?
SQL Hint: department contains the department name of each employee

{"thought": "I need to see the database schema first.", "action": "GetSchema", \
"action_input": {"db_id": "company_db"}}
Observation: Database: company_db

Table: employee
  id (integer) [PK] — employee identifier
  name (text) — employee name
  department (text) — department name
  salary (real) — annual salary in USD

{"thought": "I can count employees in the Engineering department.", "action": \
"Execute", "action_input": {"sql": "SELECT COUNT(*) FROM employee WHERE \
department = 'Engineering'", "db_id": "company_db"}}
Observation: COUNT(*)
-----------
200

Total rows: 1

{"thought": "I now know the final answer.", "action": "Finish", \
"action_input": {"answer": 200}}
"""

FEW_SHOT_2 = """\
Example 2 (aggregation with GROUP BY):

Database ID: company_db
Question: Which department has the highest total salary?
SQL Hint: none

{"thought": "I need the schema first.", "action": "GetSchema", \
"action_input": {"db_id": "company_db"}}
Observation: Database: company_db

Table: employee
  id (integer) [PK] — employee identifier
  name (text) — employee name
  department (text) — department name
  salary (real) — annual salary in USD

{"thought": "I need to sum salaries per department and find the max.", \
"action": "Execute", "action_input": {"sql": "SELECT department, SUM(salary) \
as total FROM employee GROUP BY department ORDER BY total DESC LIMIT 1", \
"db_id": "company_db"}}
Observation: department  total
------------ -----------
Engineering   15000000

Total rows: 1

{"thought": "I now know the final answer.", "action": "Finish", \
"action_input": {"answer": "Engineering"}}
"""

CORRECTION_PROMPT = (
    "Your previous response was not valid JSON or did not match the required "
    "schema. You MUST respond with a single JSON object:\n"
    '{"thought": "...", "action": "GetSchema|Execute|Finish", '
    '"action_input": { ... }}\n\n'
    "Try again."
)

_indent = "    "


def _log_trace(msg: str) -> None:
    print(f"{_indent}{msg}")


class ReActAgent(AgentABC):
    """ReAct (Reasoning + Acting) agent for text-to-SQL.

    Uses an iterative Thought -> Action -> Observation loop with real tool
    execution. The agent explores the schema via GetSchema, writes and
    executes SQL via Execute, and terminates with Finish[answer].

    LLM output is JSON-formatted for robust parsing.
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

        iteration = 0
        while iteration < self.max_steps:
            current_prompt = user + scratchpad

            response = self._generate_with_retry(
                system, current_prompt, total_usage, max_retries=6
            )
            if response is None:
                full_trace = (
                    "\n".join(full_trace_parts) if full_trace_parts else scratchpad
                )
                latency_ms = (time.perf_counter() - start) * 1000
                return {
                    "answer": "",
                    "raw_output": full_trace,
                    "usage": total_usage,
                    "latency_ms": latency_ms,
                    "error": "LLM returned empty responses after multiple retries",
                    "steps": steps,
                }

            full_trace_parts.append(response)

            parsed = self._parse_response(response)
            retry_response = None

            if not parsed["valid"]:
                retry_response = self._retry_with_correction(
                    system, current_prompt, response
                )
                if retry_response:
                    self._accumulate_usage(total_usage)
                    full_trace_parts.append(retry_response)
                    parsed = self._parse_response(retry_response)

            if not parsed["valid"]:
                parse_detail = self._diagnose_parse_failure(response)
                step_record = {
                    "step_type": "parse_error",
                    "thought": "",
                    "action": None,
                    "action_input": "",
                    "observation": "Failed to parse LLM output as valid JSON.",
                    "raw_response": response,
                    "parse_detail": parse_detail,
                }
                if retry_response:
                    step_record["retry_raw_response"] = retry_response
                    step_record["retry_parse_detail"] = self._diagnose_parse_failure(
                        retry_response
                    )
                if self.trace:
                    _log_trace(f"--- Step {iteration + 1} (PARSE ERROR) ---")
                    _log_trace(f"Raw: {response[:500]}...")
                    _log_trace(f"Detail: {parse_detail}")
                scratchpad += (
                    "\nThought: (parse error)\n"
                    "Action: None\n"
                    "Action Input: \n"
                    "Observation: Error: Could not parse response as JSON. "
                    "Respond with a valid JSON object.\n"
                )
                full_trace_parts.append(CORRECTION_PROMPT)
                steps.append(step_record)
                if self.trace:
                    sys.stdout.flush()
                iteration += 1
                continue

            thought = parsed["thought"]
            action = parsed["action"]
            action_input = parsed["action_input"]

            if self.trace:
                _log_trace(f"--- Step {iteration + 1} ---")
                if thought:
                    _log_trace(f"Thought: {thought}")

            if action == "Finish":
                raw_answer = (
                    action_input.get("answer", "")
                    if isinstance(action_input, dict)
                    else ""
                )
                answer = (
                    str(raw_answer)
                    if not isinstance(raw_answer, str)
                    else raw_answer
                )
                if self.trace:
                    _log_trace(f"Finish: {answer}")
                steps.append({
                    "step_type": "action",
                    "thought": thought,
                    "action": "Finish",
                    "action_input": action_input,
                    "observation": None,
                })
                latency_ms = (time.perf_counter() - start) * 1000
                return {
                    "answer": answer,
                    "raw_output": "\n".join(full_trace_parts),
                    "usage": total_usage,
                    "latency_ms": latency_ms,
                    "error": None,
                    "steps": steps,
                }

            if action in ("GetSchema", "Execute"):
                if self.trace:
                    _log_trace(f"Action: {action}")
                    display_input = str(action_input)
                    if len(display_input) > 120:
                        display_input = display_input[:120] + "..."
                    _log_trace(f"Action Input: {display_input}")

                observation = self._execute_tool(action, action_input, db_id)

                scratchpad += (
                    f"\nThought: {thought}\n"
                    f"Action: {action}\n"
                    f"Action Input: {json.dumps(action_input)}\n"
                    f"Observation: {observation}\n"
                )
                full_trace_parts.append(f"Observation: {observation}")

                if self.trace:
                    obs_display = observation
                    if len(obs_display) > 200:
                        obs_display = obs_display[:200] + "..."
                    _log_trace(f"Observation: {obs_display}")

                steps.append({
                    "step_type": "action",
                    "thought": thought,
                    "action": action,
                    "action_input": action_input,
                    "observation": observation,
                })
            else:
                correction_obs = (
                    f"Error: Unknown action '{action}'. "
                    f"Available actions: GetSchema, Execute, Finish."
                )
                if self.trace:
                    _log_trace(f"Action: {action} (INVALID)")
                    _log_trace(f"Observation: {correction_obs}")
                scratchpad += (
                    f"\nThought: {thought}\n"
                    f"Action: {action}\n"
                    f"Action Input: {json.dumps(action_input)}\n"
                    f"Observation: {correction_obs}\n"
                )
                full_trace_parts.append(f"Observation: {correction_obs}")
                steps.append({
                    "step_type": "action",
                    "thought": thought,
                    "action": action,
                    "action_input": action_input,
                    "observation": correction_obs,
                })

            if self.trace:
                sys.stdout.flush()

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

    def _build_user_prompt(self, question: str, evidence: str, db_id: str) -> str:
        parts = [f"Database ID: {db_id}", f"Question: {question}"]
        if evidence:
            parts.append(f"SQL Hint: {evidence}")
        parts.append("\nStart by calling GetSchema to explore the database structure.")
        return "\n".join(parts)

    def _generate_with_retry(
        self,
        system: str,
        prompt: str,
        total_usage: Dict[str, int],
        max_retries: int = 6,
    ) -> str | None:
        for _ in range(max_retries + 1):
            try:
                response = self.llm.generate(
                    system, prompt,
                    max_tokens=2048,
                    temperature=0.0,
                )
                self._accumulate_usage(total_usage)
                if response and response.strip():
                    return response
            except Exception:
                continue
        return None

    def _parse_response(self, response: str) -> Dict[str, Any]:
        cleaned = re.sub(r"```(?:json)?\s*", "", response).strip()

        try:
            obj = json.loads(cleaned)
            if isinstance(obj, dict):
                return self._validate_parsed_object(obj)
        except (json.JSONDecodeError, ValueError):
            pass

        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return {"valid": False}

        try:
            obj = json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            return {"valid": False}

        if not isinstance(obj, dict):
            return {"valid": False}

        return self._validate_parsed_object(obj)

    def _validate_parsed_object(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        thought = obj.get("thought", "")
        action = obj.get("action")

        if not action or action not in VALID_ACTIONS:
            return {"valid": False}

        action_input = obj.get("action_input", {})
        if isinstance(action_input, str):
            try:
                action_input = json.loads(action_input)
            except (json.JSONDecodeError, ValueError):
                action_input = {}

        if not isinstance(action_input, dict):
            action_input = {"value": action_input}

        return {
            "valid": True,
            "thought": thought,
            "action": action,
            "action_input": action_input,
        }

    def _diagnose_parse_failure(self, response: str) -> Dict[str, Any]:
        cleaned = re.sub(r"```(?:json)?\s*", "", response).strip()
        diagnosis: Dict[str, Any] = {
            "response_length": len(response),
            "cleaned_length": len(cleaned),
            "starts_with_brace": cleaned.startswith("{"),
            "ends_with_brace": cleaned.endswith("}"),
            "has_json_object": bool(re.search(r"\{.*\}", cleaned, re.DOTALL)),
            "has_markdown_fences": "```" in response,
        }

        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            candidate = match.group()
            diagnosis["candidate_length"] = len(candidate)
            try:
                obj = json.loads(candidate)
                diagnosis["json_parse_ok"] = True
                diagnosis["parsed_keys"] = (
                    list(obj.keys())
                    if isinstance(obj, dict)
                    else type(obj).__name__
                )
                if isinstance(obj, dict):
                    action = obj.get("action")
                    diagnosis["action_value"] = action
                    diagnosis["action_valid"] = action in VALID_ACTIONS if action else False
                    diagnosis["has_action_input"] = "action_input" in obj
                    diagnosis["has_thought"] = "thought" in obj
            except (json.JSONDecodeError, ValueError) as e:
                diagnosis["json_parse_ok"] = False
                diagnosis["json_error"] = str(e)
                diagnosis["json_error_pos"] = getattr(e, "pos", None)
        else:
            diagnosis["has_json_object"] = False

        return diagnosis

    def _retry_with_correction(
        self, system: str, current_prompt: str, original_response: str
    ) -> str | None:
        retry_prompt = (
            current_prompt
            + f"\n{original_response}\n\n"
            + CORRECTION_PROMPT
        )
        try:
            return self.llm.generate(
                system, retry_prompt,
                max_tokens=2048,
                temperature=0.0,
            )
        except Exception:
            return None

    def _execute_tool(
        self, action: str, action_input: Dict[str, Any], db_id: str
    ) -> str:
        params = dict(action_input) if isinstance(action_input, dict) else {}
        params.setdefault("db_id", db_id)

        if action == "GetSchema":
            return self._schema_tool.run(params)
        if action == "Execute":
            return self._sql_tool.run(params)
        return f"Error: Unknown action '{action}'"

    def _accumulate_usage(self, total: Dict[str, int]) -> None:
        usage = self.llm.get_usage()
        total["prompt_tokens"] += usage.get("prompt_tokens", 0)
        total["completion_tokens"] += usage.get("completion_tokens", 0)
        total["total_tokens"] += usage.get("total_tokens", 0)
