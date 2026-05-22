import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

from . import ToolABC, register_tool

BIRD_ROOT = Path("data/bird")


@register_tool
class ExecuteSQLTool(ToolABC):
    """Execute a SQL query against a BIRD SQLite database and return results.

    The agent passes 'sql' and 'db_id' as parameters. The tool connects to the
    corresponding database, executes the query in read-only mode, and returns
    the full result set as a formatted text table.
    """

    @property
    def name(self) -> str:
        return "Execute"

    @property
    def description(self) -> str:
        return (
            "Execute a SQL query against the database. "
            "Params: {\"sql\": \"<SQL query>\", \"db_id\": \"<database id>\"}"
        )

    def run(self, params: Dict[str, Any]) -> str:
        sql = params.get("sql", "").strip()
        db_id = params.get("db_id", "").strip()

        if not sql:
            return "Error: No SQL query provided."
        if not db_id:
            return "Error: No db_id provided."

        result = self._execute(db_id, sql)
        if result is None:
            return "Error: Query execution failed. Check SQL syntax."

        columns, rows = result
        return self._format_results(columns, rows)

    def _execute(self, db_id: str, sql: str, timeout: int = 30) -> Optional[tuple]:
        db_path = BIRD_ROOT / "minidev" / "MINIDEV" / "dev_databases" / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            db_path = BIRD_ROOT / "dev_databases" / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            return None

        try:
            conn = sqlite3.connect(str(db_path), timeout=timeout)
            conn.execute("PRAGMA query_only = 1")
            cursor = conn.execute(sql)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            conn.close()
            return columns, rows
        except Exception:
            return None

    def _format_results(self, columns: list, rows: list) -> str:
        if not columns and not rows:
            return "Query executed successfully. No results returned."

        col_widths = [len(str(c)) for c in columns]
        str_rows = []
        for row in rows:
            str_row = [str(v) if v is not None else "NULL" for v in row]
            str_rows.append(str_row)
            for i, val in enumerate(str_row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(val))

        header = " | ".join(str(c).ljust(col_widths[i]) for i, c in enumerate(columns))
        separator = "-+-".join("-" * w for w in col_widths)
        data_lines = []
        for row in str_rows:
            line = " | ".join(val.ljust(col_widths[i]) for i, val in enumerate(row))
            data_lines.append(line)

        parts = [header, separator]
        if len(data_lines) > 50:
            parts.extend(data_lines[:50])
            parts.append(f"... ({len(data_lines) - 50} more rows)")
        else:
            parts.extend(data_lines)

        parts.append(f"\nTotal rows: {len(data_lines)}")
        return "\n".join(parts)
