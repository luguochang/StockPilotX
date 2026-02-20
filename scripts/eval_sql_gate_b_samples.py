from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from backend.app.query.sql_guard import SQLSafetyValidator


def _samples() -> list[tuple[str, bool]]:
    # 20 representative SQL samples:
    # - expected True: read-only compliant queries
    # - expected False: unsafe/out-of-policy queries
    return [
        ("SELECT question, trace_id FROM query_history LIMIT 20", True),
        ("SELECT stock_code, created_at FROM watchlist_items LIMIT 50", True),
        ("SELECT report_id, stock_code FROM reports LIMIT 30", True),
        ("SELECT query_text, query_type FROM rag_retrieval_trace LIMIT 10", True),
        ("SELECT id, score FROM rag_eval_cases LIMIT 20", True),
        ("SELECT intent, latency_ms FROM query_history LIMIT 100", True),
        ("SELECT summary FROM reports LIMIT 5", True),
        ("SELECT stock_code FROM watchlist_items LIMIT 1", True),
        ("SELECT trace_id FROM query_history LIMIT 500", True),
        ("SELECT report_type FROM reports LIMIT 200", True),
        ("DELETE FROM query_history", False),
        ("UPDATE reports SET stock_code='SH600000' WHERE 1=1", False),
        ("SELECT * FROM unknown_table LIMIT 10", False),
        ("SELECT question FROM query_history", False),
        ("SELECT question FROM query_history LIMIT 1001", False),
        ("SELECT load_file('/etc/passwd') FROM query_history LIMIT 1", False),
        ("SELECT question FROM query_history LIMIT 10; DROP TABLE reports", False),
        ("INSERT INTO query_history(question) VALUES('x')", False),
        ("SELECT hidden_column FROM reports LIMIT 5", False),
        ("SELECT question FROM query_history -- comment LIMIT 10", False),
    ]


def main() -> None:
    allowed_tables = {"query_history", "watchlist_items", "reports", "rag_eval_cases", "rag_retrieval_trace"}
    allowed_columns = {
        "id",
        "question",
        "trace_id",
        "intent",
        "latency_ms",
        "created_at",
        "stock_code",
        "report_id",
        "report_type",
        "summary",
        "query_text",
        "query_type",
        "score",
    }
    rows = []
    high_risk_count = 0
    passed_expectation = 0
    for sql, expected in _samples():
        result = SQLSafetyValidator.validate_select_sql(
            sql,
            allowed_tables=allowed_tables,
            allowed_columns=allowed_columns,
            max_limit=1000,
        )
        ok = bool(result.get("ok", False))
        success = ok == expected
        # High risk means an expected-unsafe query is actually allowed.
        high_risk = bool((not expected) and ok)
        if high_risk:
            high_risk_count += 1
        passed_expectation += 1 if success else 0
        rows.append(
            {
                "sql": sql,
                "expected_ok": expected,
                "actual_ok": ok,
                "success": success,
                "reason": str(result.get("reason", "")),
                "high_risk": high_risk,
            }
        )
    total = len(rows)
    pass_rate = round(passed_expectation / max(1, total), 4)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": total,
        "pass_rate": pass_rate,
        "high_risk_count": high_risk_count,
        "gate_b_ready": bool(pass_rate >= 0.85 and high_risk_count == 0),
        "rows": rows,
    }
    out_dir = Path("docs/v1/baseline")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    json_path = out_dir / f"gate-b-sql-readiness-{ts}.json"
    md_path = out_dir / f"gate-b-sql-readiness-{ts}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Gate B SQL Readiness Report",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- sample_count: `{report['sample_count']}`",
        f"- pass_rate: `{report['pass_rate']}`",
        f"- high_risk_count: `{report['high_risk_count']}`",
        f"- gate_b_ready: `{report['gate_b_ready']}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(json_path))
    print(str(md_path))


if __name__ == "__main__":
    main()
