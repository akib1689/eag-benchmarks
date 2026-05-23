"""Answer extraction from agent raw output.

Tries multiple strategies to extract a structured answer from
the agent's raw text output. No SQL extraction — agents produce answers.
"""

import json
import re
from typing import Any, Dict, Optional


def _extract_finish(raw: str) -> Optional[Any]:
    """Extract answer from ReAct-style Finish[answer] pattern."""
    match = re.search(r"Finish\[(.+?)\]\s*$", raw, re.MULTILINE)
    if not match:
        return None
    answer_str = match.group(1).strip()
    return _parse_value(answer_str)


def _extract_json(raw: str) -> Optional[Any]:
    """Try to parse JSON from the output."""
    json_patterns = [
        r"```json\s*(.*?)\s*```",
        r"({.*})",
        r"(\[.*\])",
    ]
    for pattern in json_patterns:
        matches = re.findall(pattern, raw, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict) and "answer" in parsed:
                    return parsed["answer"]
                if isinstance(parsed, dict) and "result" in parsed:
                    return parsed["result"]
                return parsed
            except (json.JSONDecodeError, ValueError):
                continue
    return None


def _extract_scalar(raw: str) -> Optional[Any]:
    """Extract a numeric scalar from the output."""
    lines = [line.strip() for line in raw.strip().split("\n") if line.strip()]
    if not lines:
        return None

    last_lines = lines[-3:] if len(lines) >= 3 else lines
    for line in reversed(last_lines):
        line = line.strip()
        if not line:
            continue

        if line.startswith("Answer:") or line.startswith("answer:"):
            val_str = line.split(":", 1)[1].strip()
            return _parse_value(val_str)

        pct_match = re.match(r"^(-?[\d.]+)\s*%$", line)
        if pct_match:
            return float(pct_match.group(1))

        embedded_pct = re.search(r"(-?[\d.]+)\s*%", line)
        if embedded_pct:
            return float(embedded_pct.group(1))

        num_match = re.match(r"^-?[\d,]+\.?\d*$", line.replace(",", ""))
        if num_match:
            return _parse_value(line)

    return None


def _extract_last_line(raw: str) -> Optional[Any]:
    """Take the last meaningful line of output."""
    lines = [line.strip() for line in raw.strip().split("\n") if line.strip()]
    if not lines:
        return None
    last = lines[-1]
    return _parse_value(last)


def _parse_value(val_str: str) -> Any:
    """Parse a string value into the most appropriate Python type."""
    val_str = val_str.strip().strip("'\"")

    if val_str.lower() in ("null", "none", "n/a", "nan"):
        return None

    try:
        int_val = int(val_str)
        return int_val
    except ValueError:
        pass

    try:
        float_val = float(val_str)
        return float_val
    except ValueError:
        pass

    try:
        parsed = json.loads(val_str)
        return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    return val_str


def extract_answer(raw_output: str) -> Dict[str, Any]:
    """Extract a structured answer from the agent's raw output.

    Tries strategies in order:
      1. Finish[answer] (ReAct-style)
      2. JSON parse
      3. Numeric scalar extraction
      4. Last-line text extraction
      5. Fallback: raw output as-is

    Returns:
        {"success": bool, "answer": <extracted>, "method": str}
    """
    if not raw_output or not raw_output.strip():
        return {"success": False, "answer": None, "method": "empty"}

    strategies = [
        ("finish", _extract_finish),
        ("json", _extract_json),
        ("scalar", _extract_scalar),
        ("last_line", _extract_last_line),
    ]

    for method_name, strategy in strategies:
        result = strategy(raw_output)
        if result is not None:
            return {"success": True, "answer": result, "method": method_name}

    return {"success": False, "answer": raw_output.strip(), "method": "fallback"}
