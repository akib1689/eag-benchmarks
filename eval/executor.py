import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

BIRD_ROOT = Path("data/bird")


def execute_sql(db_id: str, sql: str, timeout: int = 30) -> Optional[list]:
    """Sandboxed SQL execution. Returns result rows or None on error."""
    db_path = BIRD_ROOT / "dev_databases" / db_id / f"{db_id}.sqlite"
    try:
        conn = sqlite3.connect(str(db_path), timeout=timeout)
        conn.execute("PRAGMA query_only = 1")
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return df.values.tolist()
    except Exception:
        return None


def execution_accuracy(pred_sql: str, gold_sql: str, db_id: str) -> bool:
    """Compare result sets (order-agnostic) for execution accuracy.

    This matches the official BIRD evaluation: two queries are equivalent
    if they produce the same set of result rows regardless of ordering.
    """
    pred = execute_sql(db_id, pred_sql)
    gold = execute_sql(db_id, gold_sql)

    if pred is None or gold is None:
        return False

    return set(tuple(row) for row in pred) == set(tuple(row) for row in gold)
