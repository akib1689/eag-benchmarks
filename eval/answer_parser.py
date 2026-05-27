"""Agent answer parsing for the evaluation pipeline.

Takes raw agent output strings and extracts structured Python values
for deterministic comparison against gold answers. Tries multiple
parse strategies in order of specificity.
"""

import json
import re
from typing import Any, List, Optional, Union


def _try_json_parse(text: str) -> Optional[Any]:
    try:
        result = json.loads(text)
        return result
    except (json.JSONDecodeError, ValueError):
        return None


def _try_bracketed_list(text: str) -> Optional[List[str]]:
    text = text.strip()
    if not text.startswith("[") or not text.endswith("]"):
        return None
    inner = text[1:-1].strip()
    if not inner:
        return []
    items = re.split(r""",\s*(?=(?:[^"']|["'][^"']*["'])*$)""", inner)
    return [item.strip().strip("\"'") for item in items if item.strip()]


def _try_newline_split(text: str) -> Optional[List[str]]:
    if "\n" not in text:
        return None
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    if len(lines) < 2:
        return None
    return lines


def _try_comma_split(text: str) -> Optional[List[str]]:
    if "," not in text:
        return None
    items = [item.strip() for item in text.split(",")]
    items = [item for item in items if item]
    if len(items) < 2:
        return None
    return items


def _try_pipe_split(text: str) -> Optional[List[str]]:
    if "|" not in text:
        return None
    items = [item.strip() for item in text.split("|")]
    items = [item for item in items if item]
    if len(items) < 2:
        return None
    return items


def _parse_rows_of_json_arrays(text: str) -> Optional[List[list]]:
    pattern = re.compile(r"\[[^\]]+\]")
    matches = pattern.findall(text)
    if not matches:
        return None
    rows = []
    for m in matches:
        parsed = _try_json_parse(m)
        if isinstance(parsed, list):
            rows.append(parsed)
        else:
            return None
    return rows if rows else None


def parse_agent_answer(raw: str) -> Union[str, int, float, list]:
    """Parse a raw agent answer string into a structured Python value.

    Tries in order:
    1. JSON parse (handles arrays, objects, numbers)
    2. Rows of JSON arrays: [1,"a"],[2,"b"] -> [[1,"a"],[2,"b"]]
    3. Bracketed list: [a, b, c] -> ["a", "b", "c"]
    4. Newline-separated values
    5. Comma-separated values
    6. Pipe-separated values
    7. Fallback: single value

    Returns a string, int, float, or list of values.
    """
    if raw is None:
        return ""

    if not isinstance(raw, str):
        return raw

    text = raw.strip()
    if not text:
        return ""

    result = _try_json_parse(text)
    if result is not None:
        return result

    stripped = text.strip()
    if stripped.startswith("(") and stripped.endswith(")"):
        inner = stripped[1:-1].strip()
        items = [item.strip() for item in inner.split(",")]
        items = [item for item in items if item]
        if len(items) >= 2:
            parsed = [_coerce_scalar(item) for item in items]
            return parsed

    rows = _parse_rows_of_json_arrays(text)
    if rows is not None and len(rows) >= 2:
        return rows

    bracketed = _try_bracketed_list(text)
    if bracketed is not None:
        parsed = []
        for item in bracketed:
            v = _coerce_scalar(item)
            parsed.append(v)
        return parsed

    newlines = _try_newline_split(text)
    if newlines is not None:
        parsed = []
        for item in newlines:
            v = _coerce_scalar(item)
            parsed.append(v)
        return parsed

    commas = _try_comma_split(text)
    if commas is not None:
        parsed = []
        for item in commas:
            v = _coerce_scalar(item)
            parsed.append(v)
        return parsed

    pipes = _try_pipe_split(text)
    if pipes is not None:
        parsed = []
        for item in pipes:
            v = _coerce_scalar(item)
            parsed.append(v)
        return parsed

    return _coerce_scalar(text)


def _coerce_scalar(text: str) -> Union[str, int, float]:
    """Coerce a string to int, float, or keep as string."""
    text = text.strip()
    try:
        int_val = int(text)
        return int_val
    except (ValueError, TypeError):
        pass
    try:
        float_val = float(text)
        return float_val
    except (ValueError, TypeError):
        pass
    return text


def normalize_agent_parsed(parsed: Any) -> Any:
    """Normalize a parsed agent answer into a canonical form for comparison.

    Returns one of:
    - A scalar (int, float, str, None) for single-value answers
    - A tuple of scalars for row answers
    - A sorted list of tuples for table/column answers
    """
    if isinstance(parsed, (int, float, str)):
        return _normalize_scalar(parsed)
    if isinstance(parsed, bool):
        return _normalize_scalar(str(parsed))
    if parsed is None:
        return None

    if isinstance(parsed, dict):
        vals = list(parsed.values())
        if len(vals) == 1:
            return _normalize_scalar(vals[0])
        return tuple(_normalize_scalar(v) for v in vals)

    if isinstance(parsed, (list, tuple)):
        if len(parsed) == 0:
            return []

        first = parsed[0]
        if isinstance(first, (list, tuple)):
            norm_rows = [tuple(_normalize_scalar(v) for v in row) for row in parsed]
            if len(norm_rows) == 1 and len(norm_rows[0]) == 1:
                return norm_rows[0][0]
            return sorted(norm_rows)

        norm_vals = [_normalize_scalar(v) for v in parsed]
        if len(norm_vals) == 1:
            return norm_vals[0]
        return tuple(norm_vals)

    return parsed


def _normalize_scalar(val: Any) -> Any:
    """Normalize a single scalar value."""
    if val is None:
        return None
    if isinstance(val, bool):
        return str(val).lower()
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        if val == int(val) and not (val != val):
            return int(val)
        return round(val, 10)
    if isinstance(val, str):
        val = val.strip()
        if val.lower() in ("null", "none", "n/a", "nan", ""):
            return None
        if val.lower() in ("true", "yes"):
            return "true"
        if val.lower() in ("false", "no"):
            return "false"
        try:
            int_val = int(val)
            return int_val
        except ValueError:
            pass
        try:
            float_val = float(val)
            if float_val == int(float_val):
                return int(float_val)
            return round(float_val, 10)
        except ValueError:
            pass
        return val.lower().strip()
    return val
