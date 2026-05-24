import csv
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

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

        return self._format_enriched(db_id, raw_schema, tables_meta, descriptions)

    def _format_enriched(
        self,
        db_id: str,
        raw_schema: Dict[str, str],
        tables_meta: Optional[Dict],
        descriptions: Dict[str, Dict[str, Dict[str, str]]],
    ) -> str:
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
                if data_fmt and data_fmt.lower() not in (ctype, "text", ""):
                    line += f" (format: {data_fmt})"
                if val_desc:
                    val_clean = val_desc.replace("\n", " ").strip()
                    line += f" | {val_clean}"

                parts.append(line)

            if tname in raw_schema:
                raw = raw_schema[tname]
                if "references" in raw.lower() or "foreign" in raw.lower():
                    parts.append(f"  [DDL: {raw.strip()}]")

            parts.append("")

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
