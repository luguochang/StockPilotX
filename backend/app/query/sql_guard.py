from __future__ import annotations

import re
from typing import Any


class SQLSafetyValidator:
    """Read-only SQL validator for SQL Agent PoC guardrails."""

    _FORBIDDEN_PATTERNS = (
        r"\b(insert|update|delete|drop|alter|truncate|create|replace|grant|revoke)\b",
        r"\b(load_file|outfile|sleep|benchmark)\b",
        r"--",
        r"/\*",
        r";",
    )

    @classmethod
    def validate_select_sql(
        cls,
        sql: str,
        *,
        allowed_tables: set[str],
        allowed_columns: set[str],
        max_limit: int = 500,
    ) -> dict[str, Any]:
        statement = str(sql or "").strip()
        if not statement:
            return {"ok": False, "reason": "empty_sql"}
        normalized = re.sub(r"\s+", " ", statement.lower()).strip()
        if not normalized.startswith("select "):
            return {"ok": False, "reason": "only_select_allowed"}

        for pattern in cls._FORBIDDEN_PATTERNS:
            if re.search(pattern, normalized):
                return {"ok": False, "reason": "forbidden_sql_pattern", "pattern": pattern}

        from_tables = re.findall(r"\bfrom\s+([a-zA-Z_][\w]*)", normalized)
        join_tables = re.findall(r"\bjoin\s+([a-zA-Z_][\w]*)", normalized)
        used_tables = set(from_tables + join_tables)
        if not used_tables:
            return {"ok": False, "reason": "missing_table"}
        if not used_tables.issubset({t.lower() for t in allowed_tables}):
            return {"ok": False, "reason": "table_not_allowed", "tables": sorted(used_tables)}

        # Approximate column extraction for PoC security boundary.
        select_match = re.search(r"^select\s+(.+?)\s+from\s+", normalized)
        if not select_match:
            return {"ok": False, "reason": "invalid_select_clause"}
        selected = select_match.group(1).strip()
        if selected != "*":
            columns = [c.strip() for c in selected.split(",") if c.strip()]
            cleaned_columns = {
                re.sub(r"\bas\b.+$", "", re.sub(r"\w+\((.*?)\)", r"\1", col)).strip().split(".")[-1]
                for col in columns
            }
            cleaned_columns = {c for c in cleaned_columns if c}
            if cleaned_columns and not cleaned_columns.issubset({c.lower() for c in allowed_columns}):
                return {"ok": False, "reason": "column_not_allowed", "columns": sorted(cleaned_columns)}

        limit_match = re.search(r"\blimit\s+(\d+)\b", normalized)
        if not limit_match:
            return {"ok": False, "reason": "limit_required"}
        limit = int(limit_match.group(1))
        if limit > int(max_limit):
            return {"ok": False, "reason": "limit_exceeded", "limit": limit, "max_limit": int(max_limit)}

        return {
            "ok": True,
            "reason": "ok",
            "tables": sorted(used_tables),
            "limit": limit,
        }
