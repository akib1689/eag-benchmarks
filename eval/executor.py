import sqlite3
from pathlib import Path
from typing import Optional

BIRD_ROOT = Path("data/bird")


def execute_sql(db_id: str, sql: str, timeout: int = 30) -> Optional[list]:
    """Execute a SQL query against a BIRD SQLite database.

    Returns result rows as list of tuples, or None on error.
    This is a low-level utility used by tools and gold answer generation.
    """
    db_path = _find_db(db_id)
    if db_path is None:
        return None
    try:
        conn = sqlite3.connect(str(db_path), timeout=timeout)
        conn.execute("PRAGMA query_only = 1")
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        return rows
    except Exception:
        return None


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
