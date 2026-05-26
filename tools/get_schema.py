import csv
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import ToolABC, register_tool

BIRD_ROOT = Path("data/bird")


@register_tool
class GetSchemaTool(ToolABC):
    """Retrieve the enriched schema for a BIRD database.

    Combines information from multiple BIRD metadata sources:
      - CREATE TABLE statements (types, constraints)
      - dev_tables.json (human-readable column names, PKs, FKs)
      - database_description/*.csv (column descriptions, value descriptions)

    Returns a human-friendly schema that helps the agent understand column
    meanings, relationships, and value formats.
    """

    @property
    def name(self) -> str:
        return "GetSchema"

    @property
    def description(self) -> str:
        return (
            "Get the enriched database schema including table structures, "
            "column descriptions, primary keys, and foreign key relationships. "
            "Params: {\"db_id\": \"<database id>\"}"
        )

    def run(self, params: Dict[str, Any]) -> str:
        db_id = params.get("db_id", "").strip()

        if not db_id:
            return "Error: No db_id provided."

        db_path = self._find_db(db_id)
        if db_path is None:
            return f"Error: Database '{db_id}' not found."

        tables_meta = self._load_dev_tables(db_id)
        descriptions = self._load_descriptions(db_id)
        raw_schema = self._load_raw_schema(db_path)

        if not raw_schema and not tables_meta:
            return f"Error: No schema found for database '{db_id}'."

        return self._format_enriched(db_id, raw_schema, tables_meta, descriptions, db_path)

    def _format_enriched(
        self,
        db_id: str,
        raw_schema: Dict[str, str],
        tables_meta: Optional[Dict],
        descriptions: Dict[str, Dict[str, Dict[str, str]]],
        db_path: Optional[Path] = None,
    ) -> str:
        conn: Optional[sqlite3.Connection] = None
        if db_path and db_path.exists():
            try:
                conn = sqlite3.connect(str(db_path))
                conn.execute("PRAGMA query_only = 1")
            except Exception:
                conn = None

        if tables_meta:
            table_names = tables_meta.get("table_names_original", [])
            col_names_orig = tables_meta.get("column_names_original", [])
            col_names_human = tables_meta.get("column_names", [])
            col_types = tables_meta.get("column_types", [])
            pks = tables_meta.get("primary_keys", [])
            fks = tables_meta.get("foreign_keys", [])
        else:
            table_names = list(raw_schema.keys())
            col_names_orig = []
            col_names_human = []
            col_types = []
            pks = []
            fks = []

        pk_set = set()
        for pk in pks:
            if isinstance(pk, list):
                for p in pk:
                    pk_set.add(p)
            else:
                pk_set.add(pk)

        fk_map: Dict[int, int] = {}
        for fk_pair in fks:
            if len(fk_pair) == 2:
                fk_map[fk_pair[0]] = fk_pair[1]

        col_by_table: Dict[int, list] = {}
        for idx, (tidx, cname) in enumerate(col_names_orig):
            if tidx == -1:
                continue
            col_by_table.setdefault(tidx, []).append(idx)

        fk_strs: Dict[int, str] = {}
        table_col_names: Dict[int, Dict[int, str]] = {}
        for tidx in range(len(table_names)):
            table_col_names[tidx] = {}
            for col_idx in col_by_table.get(tidx, []):
                cname = col_names_orig[col_idx][1]
                table_col_names[tidx][col_idx] = cname
                if col_idx in fk_map:
                    ref_col_idx = fk_map[col_idx]
                    ref_tidx = col_names_orig[ref_col_idx][0]
                    ref_cname = col_names_orig[ref_col_idx][1]
                    if ref_tidx < len(table_names):
                        ref_tname = table_names[ref_tidx]
                        fk_strs[col_idx] = f"FK -> {ref_tname}.{ref_cname}"

        parts = [f"Database: {db_id}\n"]

        for tidx, tname in enumerate(table_names):
            parts.append(f"Table: {tname}")

            table_descs = descriptions.get(tname, {})
            indices = col_by_table.get(tidx, [])

            for col_idx in indices:
                cname = col_names_orig[col_idx][1]
                human = col_names_human[col_idx][1] if col_idx < len(col_names_human) else cname
                ctype = col_types[col_idx] if col_idx < len(col_types) else "text"

                annotations = []
                if col_idx in pk_set:
                    annotations.append("PK")
                if col_idx in fk_strs:
                    annotations.append(fk_strs[col_idx])

                ann_str = f" [{', '.join(annotations)}]" if annotations else ""

                desc_info = table_descs.get(cname, {})
                col_desc = desc_info.get("column_description", "")
                val_desc = desc_info.get("value_description", "")
                data_fmt = desc_info.get("data_format", "")

                line = f"  {cname} ({ctype}){ann_str}"
                if human and human != cname:
                    line += f' "{human}"'
                if col_desc:
                    line += f" — {col_desc}"
                format_shown = False
                if data_fmt and data_fmt.lower() not in (ctype, "text", ""):
                    line += f" (format: {data_fmt})"
                    format_shown = True
                if val_desc:
                    val_clean = val_desc.replace("\n", " ").strip()
                    line += f" | {val_clean}"

                if conn and ctype == "text":
                    distinct = self._get_distinct_values(conn, tname, cname)
                    if distinct is not None and len(distinct) <= 20:
                        line += f" {{{', '.join(distinct)}}}"

                if conn and not format_shown and self._is_date_column(cname, ctype):
                    date_info = self._detect_date_format(conn, tname, cname)
                    if date_info:
                        line += f" [format: {date_info['pattern']}]"

                parts.append(line)

            if tname in raw_schema:
                raw = raw_schema[tname]
                if "references" in raw.lower() or "foreign" in raw.lower():
                    parts.append(f"  [DDL: {raw.strip()}]")

            if conn:
                row_count = self._get_row_count(conn, tname)
                if row_count is not None:
                    parts.append(f"  [{row_count} rows]")

            parts.append("")

        if conn:
            try:
                conn.close()
            except Exception:
                pass

        return "\n".join(parts)

    def _load_dev_tables(self, db_id: str) -> Optional[Dict]:
        for candidate in [
            BIRD_ROOT / "minidev" / "MINIDEV" / "dev_tables.json",
            BIRD_ROOT / "dev_tables.json",
        ]:
            if candidate.exists():
                try:
                    with open(candidate) as f:
                        all_tables = json.load(f)
                    for entry in all_tables:
                        if entry.get("db_id") == db_id:
                            return entry
                except (json.JSONDecodeError, ValueError):
                    pass
        return None

    def _load_descriptions(self, db_id: str) -> Dict[str, Dict[str, Dict[str, str]]]:
        result: Dict[str, Dict[str, Dict[str, str]]] = {}
        for base in [
            BIRD_ROOT / "minidev" / "MINIDEV" / "dev_databases" / db_id / "database_description",
            BIRD_ROOT / "dev_databases" / db_id / "database_description",
        ]:
            if not base.exists():
                continue
            for csv_path in sorted(base.glob("*.csv")):
                table_name = csv_path.stem
                cols: Dict[str, Dict[str, str]] = {}
                try:
                    with open(csv_path, newline="", encoding="utf-8-sig") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            col_name = row.get("original_column_name", "").strip()
                            if not col_name:
                                continue
                            cols[col_name] = {
                                "column_description": row.get("column_description", "").strip(),
                                "value_description": row.get("value_description", "").strip(),
                                "data_format": row.get("data_format", "").strip(),
                            }
                except Exception:
                    continue
                if cols:
                    result[table_name] = cols
        return result

    def _load_raw_schema(self, db_path: Path) -> Dict[str, str]:
        result: Dict[str, str] = {}
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type='table' AND sql IS NOT NULL "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
            for name, sql in cursor.fetchall():
                result[name] = sql
            conn.close()
        except Exception:
            pass
        return result

    def _find_db(self, db_id: str) -> Optional[Path]:
        candidates = [
            BIRD_ROOT / "minidev" / "MINIDEV" / "dev_databases" / db_id / f"{db_id}.sqlite",
            BIRD_ROOT / "dev_databases" / db_id / f"{db_id}.sqlite",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    @staticmethod
    def _is_date_column(cname: str, ctype: str) -> bool:
        if ctype == "date":
            return True
        if ctype == "text" and "date" in cname.lower():
            return True
        return False

    def _get_row_count(self, conn: sqlite3.Connection, table: str) -> Optional[int]:
        try:
            cursor = conn.execute(f'SELECT COUNT(*) FROM "{table}"')
            return cursor.fetchone()[0]
        except Exception:
            return None

    def _get_distinct_values(
        self, conn: sqlite3.Connection, table: str, column: str, limit: int = 21
    ) -> Optional[List[str]]:
        try:
            cursor = conn.execute(
                f'SELECT DISTINCT "{column}" FROM "{table}" '
                f'WHERE "{column}" IS NOT NULL LIMIT {limit}'
            )
            return [str(r[0]) for r in cursor.fetchall()]
        except Exception:
            return None

    def _detect_date_format(
        self, conn: sqlite3.Connection, table: str, column: str
    ) -> Optional[Dict[str, str]]:
        try:
            cursor = conn.execute(
                f'SELECT DISTINCT "{column}" FROM "{table}" '
                f'WHERE "{column}" IS NOT NULL LIMIT 5'
            )
            samples = [str(r[0]) for r in cursor.fetchall()]
            if not samples:
                return None
            pattern = self._infer_date_pattern(samples[0])
            if pattern:
                return {"sample": samples[0], "pattern": pattern}
            return None
        except Exception:
            return None

    @staticmethod
    def _infer_date_pattern(value: str) -> Optional[str]:
        if re.match(r"^\d{6}$", value):
            month = int(value[4:6])
            if 1 <= month <= 12:
                return "YYYYMM"
        if re.match(r"^\d{4}-\d{2}-\d{2}$", value):
            return "YYYY-MM-DD"
        if re.match(r"^\d{4}/\d{2}/\d{2}$", value):
            return "YYYY/MM/DD"
        if re.match(r"^\d{2}/\d{2}/\d{4}$", value):
            return "MM/DD/YYYY or DD/MM/YYYY"
        return None
