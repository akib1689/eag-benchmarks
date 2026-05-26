import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

BIRD_ROOT = Path("data/bird")


def load_mini_dev(limit: Optional[int] = None) -> List[Dict]:
    """Load the BIRD Mini-Dev SQLite split.

    Each item is a dict with keys: question_id, db_id, question, evidence, SQL, difficulty.
    """
    candidates = [
        BIRD_ROOT / "minidev" / "MINIDEV" / "mini_dev_sqlite.json",
        BIRD_ROOT / "mini_dev_sqlite.json",
    ]
    json_path = None
    for path in candidates:
        if path.exists():
            json_path = path
            break

    if json_path is None:
        raise FileNotFoundError(
            f"BIRD Mini-Dev data not found. Searched: {[str(p) for p in candidates]}. "
            "Run: bash scripts/download_bird.sh"
        )

    with open(json_path) as f:
        data = json.load(f)

    return data[:limit] if limit else data


def get_schema(db_id: str) -> str:
    """Extract CREATE TABLE statements from a BIRD SQLite database.

    This gives the LLM schema-only context — no raw data rows are exposed,
    which is the core EAG principle: data sovereignty.
    """
    db_path = BIRD_ROOT / "dev_databases" / db_id / f"{db_id}.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
    )
    schema = "\n".join(row[0] for row in cursor.fetchall())
    conn.close()
    return schema


def get_table_names(db_id: str) -> List[str]:
    """Return list of table names for a given database."""
    db_path = BIRD_ROOT / "dev_databases" / db_id / f"{db_id}.sqlite"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    names = [row[0] for row in cursor.fetchall()]
    conn.close()
    return names
