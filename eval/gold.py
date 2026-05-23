"""Gold answer generation for evaluation.

Executes the gold SQL query against BIRD databases and returns
normalized answers for comparison with agent outputs.
"""

import sqlite3
from pathlib import Path
from typing import Dict, Optional, Tuple

from .normalizer import normalize_result

BIRD_ROOT = Path("data/bird")

_cache: Dict[Tuple[str, str], Optional[dict]] = {}


def _find_db(db_id: str) -> Optional[Path]:
    """Resolve the database path, checking both minidev and legacy layouts."""
    candidates = [
        BIRD_ROOT / "minidev" / "MINIDEV" / "dev_databases" / db_id / f"{db_id}.sqlite",
        BIRD_ROOT / "dev_databases" / db_id / f"{db_id}.sqlite",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _execute_gold_sql(gold_sql: str, db_id: str, timeout: int = 30) -> Optional[dict]:
    """Execute gold SQL and return raw results."""
    db_path = _find_db(db_id)
    if db_path is None:
        return None

    try:
        conn = sqlite3.connect(str(db_path), timeout=timeout)
        conn.execute("PRAGMA query_only = 1")
        cursor = conn.execute(gold_sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        conn.close()
        return normalize_result(rows, columns)
    except Exception:
        return None


def get_gold_answer(gold_sql: str, db_id: str) -> Optional[dict]:
    """Get the normalized gold answer for a BIRD question.

    Results are cached in-memory since gold SQL is deterministic.

    Args:
        gold_sql: The gold SQL query from the BIRD dataset.
        db_id: The database identifier.

    Returns:
        Normalized answer dict with keys: answer, shape, columns, raw_rows.
        None if execution fails.
    """
    cache_key = (db_id, gold_sql)
    if cache_key in _cache:
        return _cache[cache_key]

    result = _execute_gold_sql(gold_sql, db_id)
    _cache[cache_key] = result
    return result
