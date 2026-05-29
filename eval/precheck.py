"""Deterministic pre-check for answer comparison.

Runs before the LLM verdict checker. Handles straightforward matches
(scalars, lists, tables) with fuzzy numeric tolerance and
case-insensitive string comparison. Returns "undecided" for genuinely
ambiguous cases that need LLM judgment.
"""

from dataclasses import dataclass
from typing import Any, List

from .answer_parser import normalize_agent_parsed, parse_agent_answer

NUMERIC_REL_TOLERANCE = 0.01
NUMERIC_ABS_TOLERANCE = 0.01
INT_ABS_TOLERANCE = 0


@dataclass
class PrecheckResult:
    status: str  # "match" | "wrong_answer" | "mismatch" | "undecided"
    confidence: float
    explanation: str = ""


def precheck_match(agent_answer: Any, gold_answer: Any) -> PrecheckResult:
    """Deterministic comparison of agent answer vs gold answer.

    Returns PrecheckResult with status:
    - "match": confident the answers are equivalent
    - "wrong_answer": confident the answers differ
    - "mismatch": shapes are incompatible (scalar vs list, etc.)
    - "undecided": cannot determine deterministically
    """
    if agent_answer is None or (isinstance(agent_answer, str) and agent_answer.strip() == ""):
        return PrecheckResult("undecided", 0.0, "empty agent answer")

    gold_norm = _normalize_gold(gold_answer)
    agent_parsed = parse_agent_answer(agent_answer)
    agent_norm = normalize_agent_parsed(agent_parsed)

    gold_shape = _detect_shape(gold_norm)
    agent_shape = _detect_shape(agent_norm)

    if gold_shape == "scalar" and agent_shape == "scalar":
        return _compare_scalars(agent_norm, gold_norm)

    if gold_shape in ("list", "column") and agent_shape in ("list", "column", "scalar"):
        return _compare_lists(agent_norm, agent_shape, gold_norm, gold_shape)

    if gold_shape == "table" and agent_shape == "list":
        if isinstance(agent_answer, str) and ';' in agent_answer:
            return PrecheckResult("undecided", 0.5,
                                  "agent has semicolons suggesting structured text, "
                                  "gold is table, defer to LLM")
        a_rows = _ensure_table(agent_norm)
        g_rows = _ensure_table(gold_norm)
        overlap = _row_overlap(a_rows, g_rows)
        if overlap >= 0.5:
            return PrecheckResult("undecided", 0.5,
                                  f"agent list vs gold table with {overlap:.0%} "
                                  "row overlap, defer to LLM")
        return PrecheckResult("wrong_answer", 0.9,
                              "agent list vs gold table, low overlap")

    if gold_shape == "table" and agent_shape in ("table", "list"):
        return _compare_tables(agent_norm, agent_shape, gold_norm, gold_shape)

    if gold_shape == "scalar" and agent_shape in ("list", "column", "table"):
        return PrecheckResult("mismatch", 0.95,
                              f"gold is scalar but agent returned {agent_shape}")

    if gold_shape in ("list", "column", "table") and agent_shape == "scalar":
        if isinstance(gold_norm, (list, tuple)) and len(gold_norm) > 0:
            if isinstance(agent_norm, str):
                all_same = all(
                    _values_match(agent_norm, g) for g in gold_norm
                )
                if all_same:
                    return PrecheckResult("undecided", 0.6,
                                          "agent scalar matches all gold list elements, "
                                          "defer to LLM")
            if isinstance(agent_norm, (int, float)):
                if len(gold_norm) == agent_norm:
                    return PrecheckResult(
                        "undecided", 0.5,
                        "agent returned count matching gold list length"
                    )
        return PrecheckResult("undecided", 0.5,
                              f"gold is {gold_shape} but agent returned scalar")

    return PrecheckResult("undecided", 0.0,
                          f"shapes: agent={agent_shape}, gold={gold_shape}")


def _normalize_gold(gold: Any) -> Any:
    """Normalize gold answer from SQL execution into canonical form.

    Flattens column-shaped answers: [['foo'], ['bar']] -> ('foo', 'bar')
    Keeps table shape: [[1, 'a'], [2, 'b']] -> sorted tuples
    """
    if gold is None:
        return None

    if isinstance(gold, (int, float, str, bool)):
        return _norm_scalar(gold)

    if isinstance(gold, (list, tuple)):
        if len(gold) == 0:
            return []

        first = gold[0]

        if isinstance(first, (list, tuple)):
            if len(first) == 1:
                flat = tuple(_norm_scalar(row[0]) for row in gold)
                return flat
            else:
                rows = [tuple(_norm_scalar(v) for v in row) for row in gold]
                return sorted(rows)

        return tuple(_norm_scalar(v) for v in gold)

    return gold


def _norm_scalar(val: Any) -> Any:
    """Normalize a scalar value for comparison."""
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
            return int(val)
        except ValueError:
            pass
        try:
            fv = float(val)
            if fv == int(fv):
                return int(fv)
            return round(fv, 10)
        except ValueError:
            pass
        return val.lower()
    return val


def _detect_shape(val: Any) -> str:
    """Detect the shape of a normalized answer value."""
    if val is None:
        return "scalar"
    if isinstance(val, (int, float, str)):
        return "scalar"
    if isinstance(val, (list, tuple)):
        if len(val) == 0:
            return "list"
        first = val[0]
        if isinstance(first, (list, tuple)):
            return "table"
        return "list"
    return "scalar"


def _compare_scalars(agent: Any, gold: Any) -> PrecheckResult:
    """Compare two scalar values."""
    if agent is None and gold is None:
        return PrecheckResult("match", 1.0, "both None")
    if agent is None or gold is None:
        return PrecheckResult("wrong_answer", 0.95, "one is None")

    if isinstance(agent, (int, float)) and isinstance(gold, (int, float)):
        if _numeric_match(agent, gold):
            return PrecheckResult("match", 0.99,
                                  f"numeric match: {agent} ≈ {gold}")
        return PrecheckResult("wrong_answer", 0.99,
                              f"numeric mismatch: {agent} vs {gold}")

    if isinstance(agent, str) and isinstance(gold, str):
        if _string_match(agent, gold):
            return PrecheckResult("match", 0.99,
                                  f"string match: '{agent}' ≈ '{gold}'")
        if _is_likely_yn(agent) and _is_yn_bool(gold):
            return _compare_yn_bool(agent, gold)
        if _is_affirmative_pair(agent, gold):
            return PrecheckResult("undecided", 0.5,
                                  f"affirmative pair: '{agent}' vs '{gold}', "
                                  "defer to LLM")
        a_low = agent.lower().strip()
        g_low = gold.lower().strip()
        if len(g_low) >= 4 and g_low in a_low:
            return PrecheckResult("undecided", 0.5,
                                  f"gold '{g_low}' is substring of agent '{a_low}', "
                                  "defer to LLM")
        if len(a_low) >= 4 and a_low in g_low:
            return PrecheckResult("undecided", 0.5,
                                  f"agent '{a_low}' is substring of gold '{g_low}', "
                                  "defer to LLM")
        return PrecheckResult("wrong_answer", 0.95,
                              f"string mismatch: '{agent}' vs '{gold}'")

    if isinstance(agent, str) and isinstance(gold, (int, float)):
        try:
            a_num = float(agent)
            if _numeric_match(a_num, float(gold)):
                return PrecheckResult("match", 0.98,
                                      f"parsed string '{agent}' ≈ {gold}")
        except (ValueError, TypeError):
            pass
        return PrecheckResult("wrong_answer", 0.95,
                              f"type mismatch: str '{agent}' vs num {gold}")

    if isinstance(agent, (int, float)) and isinstance(gold, str):
        try:
            g_num = float(gold)
            if _numeric_match(float(agent), g_num):
                return PrecheckResult("match", 0.98,
                                      f"num {agent} ≈ parsed string '{gold}'")
        except (ValueError, TypeError):
            pass
        return PrecheckResult("wrong_answer", 0.95,
                              f"type mismatch: num {agent} vs str '{gold}'")

    return PrecheckResult("undecided", 0.5,
                          f"scalar types: agent={type(agent).__name__}, gold={type(gold).__name__}")


def _compare_lists(
    agent: Any, agent_shape: str, gold: Any, gold_shape: str
) -> PrecheckResult:
    """Compare list-shaped answers."""
    if not isinstance(agent, (list, tuple)):
        agent = [agent]
    if not isinstance(gold, (list, tuple)):
        gold = [gold]

    agent_items = list(agent)
    gold_items = list(gold)

    if len(agent_items) == 0 and len(gold_items) == 0:
        return PrecheckResult("match", 1.0, "both empty lists")

    if len(agent_items) != len(gold_items):
        joined_agent = _try_join(agent_items)
        if joined_agent is not None:
            if _values_match(joined_agent, _try_join(gold_items)):
                return PrecheckResult("match", 0.97,
                                      "joined agent matches joined gold")

        if agent_items and isinstance(agent_items[0], str) and len(agent_items) == 1:
            if len(gold_items) > 1:
                return PrecheckResult("undecided", 0.5,
                                      "agent single-value vs gold multi-value, "
                                      "defer to LLM")

        len_agent = len(agent_items)
        len_gold = len(gold_items)

        if len_gold > len_agent and len_agent <= 3:
            return PrecheckResult("undecided", 0.5,
                                  f"agent has {len_agent} items, gold has "
                                  f"{len_gold} — likely split-name difference, "
                                  "defer to LLM")

        if _element_overlap(agent_items, gold_items) >= 0.8:
            return PrecheckResult("undecided", 0.6,
                                  f"similar elements but length differs: "
                                  f"{len_agent} vs {len_gold}")
        return PrecheckResult("wrong_answer", 0.9,
                              f"length mismatch: {len_agent} vs {len_gold}")

    all_match = True
    for a, g in zip(agent_items, gold_items):
        if isinstance(a, (int, float)) and isinstance(g, (int, float)):
            if not _numeric_match(a, g):
                all_match = False
                break
        elif isinstance(a, str) and isinstance(g, str):
            if not _string_match(a, g):
                all_match = False
                break
        elif a != g:
            all_match = False
            break

    if all_match:
        return PrecheckResult("match", 0.98, "ordered list match")

    agent_sorted = _sort_values(agent_items)
    gold_sorted = _sort_values(gold_items)

    set_match = True
    for a, g in zip(agent_sorted, gold_sorted):
        if isinstance(a, (int, float)) and isinstance(g, (int, float)):
            if not _numeric_match(a, g):
                set_match = False
                break
        elif isinstance(a, str) and isinstance(g, str):
            if not _string_match(a, g):
                set_match = False
                break
        elif a != g:
            set_match = False
            break

    if set_match:
        return PrecheckResult("match", 0.97, "unordered list match (set-equivalent)")

    return PrecheckResult("wrong_answer", 0.9,
                          "list values differ")


def _compare_tables(
    agent: Any, agent_shape: str, gold: Any, gold_shape: str
) -> PrecheckResult:
    """Compare table-shaped answers (list of tuples)."""
    agent_rows = _ensure_table(agent)
    gold_rows = _ensure_table(gold)

    if len(agent_rows) != len(gold_rows):
        overlap = _row_overlap(agent_rows, gold_rows)
        if overlap >= 0.8:
            return PrecheckResult("undecided", 0.6,
                                  f"similar rows but count differs: "
                                  f"{len(agent_rows)} vs {len(gold_rows)}")
        return PrecheckResult("wrong_answer", 0.9,
                              f"row count: {len(agent_rows)} vs {len(gold_rows)}")

    if len(agent_rows) == 0:
        return PrecheckResult("match", 1.0, "both empty tables")

    agent_sorted = sorted(agent_rows, key=_row_sort_key)
    gold_sorted = sorted(gold_rows, key=_row_sort_key)

    all_match = True
    for a_row, g_row in zip(agent_sorted, gold_sorted):
        if len(a_row) != len(g_row):
            all_match = False
            break
        for a, g in zip(a_row, g_row):
            if not _values_match(a, g):
                all_match = False
                break
        if not all_match:
            break

    if all_match:
        return PrecheckResult("match", 0.97, "table match (sorted)")

    return PrecheckResult("undecided", 0.5,
                          "table rows differ, defer to LLM")


def _ensure_table(val: Any) -> List[tuple]:
    """Ensure value is a list of tuples."""
    if isinstance(val, (list, tuple)):
        result = []
        for item in val:
            if isinstance(item, (list, tuple)):
                result.append(tuple(item))
            else:
                result.append((item,))
        return result
    return [(val,)]


def _row_sort_key(row: tuple) -> tuple:
    """Sort key for table rows that handles mixed types."""
    parts = []
    for v in row:
        if isinstance(v, (int, float)):
            parts.append((0, v, ""))
        elif isinstance(v, str):
            parts.append((1, 0, v.lower()))
        elif isinstance(v, (list, tuple)):
            parts.append((2, 0, str(v)))
        else:
            parts.append((3, 0, str(v)))
    return tuple(parts)


def _values_match(a: Any, g: Any) -> bool:
    """Compare two values with fuzzy matching."""
    if a is None and g is None:
        return True
    if a is None or g is None:
        return False
    if isinstance(a, (int, float)) and isinstance(g, (int, float)):
        return _numeric_match(a, g)
    if isinstance(a, str) and isinstance(g, str):
        return _string_match(a, g)
    if isinstance(a, (int, float)) and isinstance(g, str):
        try:
            return _numeric_match(float(a), float(g))
        except (ValueError, TypeError):
            return False
    if isinstance(a, str) and isinstance(g, (int, float)):
        try:
            return _numeric_match(float(a), float(g))
        except (ValueError, TypeError):
            return False
    return a == g


def _numeric_match(a: float, b: float) -> bool:
    """Compare two numbers with relative + absolute tolerance."""
    if a == b:
        return True
    if isinstance(a, int) and isinstance(b, int):
        return False
    abs_diff = abs(a - b)
    rel_tol = max(abs(a), abs(b)) * NUMERIC_REL_TOLERANCE
    return abs_diff <= max(rel_tol, NUMERIC_ABS_TOLERANCE)


def _string_match(a: str, b: str) -> bool:
    """Case-insensitive, whitespace-normalized string comparison."""
    a_clean = a.strip().lower().strip("\"'")
    b_clean = b.strip().lower().strip("\"'")
    if a_clean == b_clean:
        return True
    a_plus = a_clean.replace("+", " ").replace(",", " ")
    b_plus = b_clean.replace("+", " ").replace(",", " ")
    a_norm = " ".join(a_plus.split())
    b_norm = " ".join(b_plus.split())
    if a_norm == b_norm:
        return True
    a_no_sep = a_clean.replace(" ", "").replace("-", "").replace("_", "").replace("+", "")
    b_no_sep = b_clean.replace(" ", "").replace("-", "").replace("_", "").replace("+", "")
    return a_no_sep == b_no_sep


def _sort_values(vals: list) -> list:
    """Sort a list of values for set comparison."""
    def sort_key(v):
        if isinstance(v, (int, float)):
            return (0, v)
        if isinstance(v, str):
            return (1, v.lower())
        if isinstance(v, (list, tuple)):
            return (2, str(v))
        return (3, str(v))

    return sorted(vals, key=sort_key)


def _element_overlap(a: list, b: list) -> float:
    """Fraction of elements in common between two lists (fuzzy)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    matches = 0
    for av in a:
        for bv in b:
            if _values_match(av, bv):
                matches += 1
                break
    return matches / max(len(a), len(b))


def _row_overlap(a: List[tuple], b: List[tuple]) -> float:
    """Fraction of rows in common between two tables."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    matches = 0
    for arow in a:
        for brow in b:
            if len(arow) == len(brow):
                if all(_values_match(av, bv) for av, bv in zip(arow, brow)):
                    matches += 1
                    break
    return matches / max(len(a), len(b))


def _is_likely_yn(val: str) -> bool:
    return val.lower().strip() in ("yes", "no", "true", "false")


def _is_yn_bool(val: str) -> bool:
    return val in ("true", "false")


def _is_affirmative_pair(a: str, b: str) -> bool:
    affirmatives = {"yes", "true", "+", "1", "positive"}
    negatives = {"no", "false", "-", "0", "negative"}
    a_low = a.lower().strip()
    b_low = b.lower().strip()
    if a_low in affirmatives and b_low in affirmatives:
        return True
    if a_low in negatives and b_low in negatives:
        return True
    return False


def _compare_yn_bool(agent: str, gold: str) -> PrecheckResult:
    a = agent.lower().strip()
    if a in ("yes", "true") and gold == "true":
        return PrecheckResult("match", 0.99, f"'{a}' matches '{gold}'")
    if a in ("no", "false") and gold == "false":
        return PrecheckResult("match", 0.99, f"'{a}' matches '{gold}'")
    return PrecheckResult("wrong_answer", 0.95, f"'{a}' vs '{gold}'")


def _try_join(items: list) -> Any:
    """Try to join list items into a single comparable value."""
    if not items:
        return None
    if all(isinstance(v, str) for v in items):
        joined = " ".join(items)
        return _norm_scalar(joined)
    if all(isinstance(v, (int, float, str)) for v in items):
        parts = [str(v) for v in items]
        joined = " ".join(parts)
        return _norm_scalar(joined)
    return None
