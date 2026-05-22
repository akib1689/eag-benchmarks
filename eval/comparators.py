"""Three-tier answer comparison for the evaluation pipeline.

Tries matching strategies in order of strictness:
  1. Exact match — values identical after normalization
  2. Set equivalence — order-agnostic row-level matching
  3. Fuzzy numeric — relative tolerance for numeric results
"""

from typing import Any, Tuple

from .normalizer import canonicalize, normalize_value

FUZZY_TOLERANCE = 0.01


def _values_equal(a: Any, b: Any) -> bool:
    """Check equality of two normalized values."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return a == b


def _is_numeric(val: Any) -> bool:
    return isinstance(val, (int, float))


def exact_match(pred: Any, gold: Any) -> bool:
    """Exact equality after canonical normalization."""
    pred_c = canonicalize(pred)
    gold_c = canonicalize(gold)

    if pred_c["shape"] != gold_c["shape"]:
        return False

    return _values_equal(pred_c["answer"], gold_c["answer"])


def _rows_match_unordered(pred_rows: list, gold_rows: list) -> bool:
    """Check if two lists of tuples contain the same rows regardless of order."""
    if len(pred_rows) != len(gold_rows):
        return False

    gold_sorted = sorted(gold_rows)
    pred_sorted = sorted(pred_rows)

    for p_row, g_row in zip(pred_sorted, gold_sorted):
        if len(p_row) != len(g_row):
            return False
        for p_val, g_val in zip(p_row, g_row):
            if not _values_equal(p_val, g_val):
                return False
    return True


def set_match(pred: Any, gold: Any) -> bool:
    """Order-agnostic row-level matching for multi-row results."""
    pred_c = canonicalize(pred)
    gold_c = canonicalize(gold)

    if pred_c["shape"] == "scalar":
        return _values_equal(pred_c["answer"], gold_c["answer"])

    pred_answer = pred_c["answer"]
    gold_answer = gold_c["answer"]

    if isinstance(pred_answer, (list, tuple)) and isinstance(
        gold_answer, (list, tuple)
    ):
        if len(pred_answer) == 0 and len(gold_answer) == 0:
            return True

        if len(pred_answer) > 0 and len(gold_answer) > 0:
            if isinstance(pred_answer[0], (list, tuple)) and isinstance(
                gold_answer[0], (list, tuple)
            ):
                return _rows_match_unordered(pred_answer, gold_answer)

            if isinstance(pred_answer[0], tuple) and isinstance(
                gold_answer[0], tuple
            ):
                return _rows_match_unordered(pred_answer, gold_answer)

    return _values_equal(
        normalize_value(pred_answer), normalize_value(gold_answer)
    )


def _fuzzy_numeric_equal(
    pred: float, gold: float, tolerance: float = FUZZY_TOLERANCE
) -> bool:
    """Compare two numeric values with relative tolerance."""
    if pred == gold:
        return True
    if gold == 0:
        return pred == 0
    return abs(pred - gold) / max(abs(gold), 1e-10) <= tolerance


def fuzzy_match(
    pred: Any, gold: Any, tolerance: float = FUZZY_TOLERANCE
) -> bool:
    """Fuzzy numeric comparison for scalar and multi-value numeric results."""
    pred_c = canonicalize(pred)
    gold_c = canonicalize(gold)

    if pred_c["shape"] != gold_c["shape"]:
        return False

    if pred_c["shape"] == "scalar":
        p, g = pred_c["answer"], gold_c["answer"]
        if _is_numeric(p) and _is_numeric(g):
            return _fuzzy_numeric_equal(float(p), float(g), tolerance)
        return False

    if pred_c["shape"] in ("row", "column", "table"):
        pred_answer = pred_c["answer"]
        gold_answer = gold_c["answer"]

        if isinstance(pred_answer, tuple) and isinstance(gold_answer, tuple):
            if len(pred_answer) != len(gold_answer):
                return False
            return all(
                _fuzzy_numeric_equal(float(p), float(g), tolerance)
                for p, g in zip(pred_answer, gold_answer)
                if _is_numeric(p) and _is_numeric(g)
            )

    return False


def compare_answers(
    pred: Any, gold: Any, tolerance: float = FUZZY_TOLERANCE
) -> Tuple[bool, float, str]:
    """Compare predicted and gold answers using tiered matching.

    Tries exact -> set -> fuzzy in order. Returns on first success.

    Args:
        pred: The predicted answer (any format).
        gold: The gold answer (normalized from SQL execution).
        tolerance: Relative tolerance for fuzzy numeric matching.

    Returns:
        (correct, confidence, tier_name)
    """
    if gold is None:
        return False, 0.0, "none"

    if exact_match(pred, gold):
        return True, 1.0, "exact"

    if set_match(pred, gold):
        return True, 0.95, "set"

    if fuzzy_match(pred, gold, tolerance):
        return True, 1.0 - tolerance, "fuzzy"

    return False, 0.0, "none"
