"""Answer normalization utilities for the evaluation pipeline.

Canonicalizes both gold answers (from SQL execution) and agent answers
(parsed from raw output) into a uniform format for comparison.
"""

from typing import Any, List, Optional, Union

NULL_VALUES = {None, "NULL", "null", "None", "N/A", "N/A ", "NaN", "nan"}


def normalize_value(val: Any) -> Optional[Union[int, float, str]]:
    """Coerce a single value to its canonical form.

    - None / NULL-like strings -> None
    - int-valued floats -> int (3.0 -> 3)
    - Numeric strings -> int or float
    - Strings -> stripped, lowercased
    """
    if val in NULL_VALUES:
        return None

    if isinstance(val, int):
        return val

    if isinstance(val, float):
        if val == int(val) and not (val != val):
            return int(val)
        return round(val, 10)

    if isinstance(val, str):
        val = val.strip()
        if val in NULL_VALUES:
            return None
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
        return val.lower()

    return val


def normalize_row(row: tuple) -> tuple:
    """Normalize each element in a result row."""
    return tuple(normalize_value(v) for v in row)


def detect_shape(rows: list, num_cols: int) -> str:
    """Classify the answer shape.

    Returns: 'scalar', 'row', 'column', or 'table'
    """
    if len(rows) == 0:
        return "empty"
    if len(rows) == 1 and num_cols == 1:
        return "scalar"
    if len(rows) == 1 and num_cols > 1:
        return "row"
    if len(rows) > 1 and num_cols == 1:
        return "column"
    return "table"


def normalize_result(
    rows: List[tuple], columns: List[str]
) -> dict:
    """Normalize a full SQL result set into canonical answer form.

    Returns:
        dict with keys:
            - answer: the canonical answer (scalar, list of tuples, etc.)
            - shape: 'scalar' | 'row' | 'column' | 'table' | 'empty'
            - columns: list of column names (lowercased)
            - raw_rows: original rows for debugging
    """
    shape = detect_shape(rows, len(columns))
    norm_cols = [c.strip().lower() for c in columns]
    norm_rows = [normalize_row(r) for r in rows]

    if shape == "scalar":
        answer = norm_rows[0][0]
    elif shape == "row":
        answer = norm_rows[0]
    elif shape == "column":
        answer = sorted(norm_rows)
    else:
        answer = sorted(norm_rows)

    return {
        "answer": answer,
        "shape": shape,
        "columns": norm_cols,
        "raw_rows": rows,
    }


def canonicalize(answer: Any) -> dict:
    """Convert an arbitrary answer value into canonical form.

    Handles: scalars (int, float, str), tuples, lists, and nested structures.
    Returns the same shape dict as normalize_result.
    """
    if answer is None:
        return {"answer": None, "shape": "scalar", "columns": [], "raw_rows": []}

    if isinstance(answer, (int, float)):
        answer = normalize_value(answer)
        return {"answer": answer, "shape": "scalar", "columns": [], "raw_rows": []}

    if isinstance(answer, str):
        answer = normalize_value(answer)
        return {"answer": answer, "shape": "scalar", "columns": [], "raw_rows": []}

    if isinstance(answer, (tuple, list)):
        if len(answer) == 0:
            return {"answer": answer, "shape": "empty", "columns": [], "raw_rows": []}

        first = answer[0]

        if isinstance(first, (tuple, list)):
            norm_rows = [tuple(normalize_value(v) for v in row) for row in answer]
            num_cols = len(norm_rows[0]) if norm_rows else 0
            shape = detect_shape(norm_rows, num_cols)
            if shape == "scalar":
                return {
                    "answer": norm_rows[0][0], "shape": "scalar",
                    "columns": [], "raw_rows": norm_rows,
                }
            if shape == "column":
                return {
                    "answer": sorted(norm_rows), "shape": "column",
                    "columns": [], "raw_rows": norm_rows,
                }
            return {
                "answer": sorted(norm_rows), "shape": shape,
                "columns": [], "raw_rows": norm_rows,
            }

        norm_vals = [normalize_value(v) for v in answer]
        return {"answer": tuple(norm_vals), "shape": "row", "columns": [], "raw_rows": []}

    return {"answer": answer, "shape": "unknown", "columns": [], "raw_rows": []}
