from __future__ import annotations

from typing import Any


class QueryComparator:
    """Builds structured comparison output for multi-stock query results."""

    @staticmethod
    def build(question: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Convert per-stock query outputs into a compact comparison payload."""
        normalized: list[dict[str, Any]] = []
        best_score = -1.0
        best_stock = ""

        for row in rows:
            stock_code = str(row.get("stock_code", "")).strip().upper()
            brief = row.get("analysis_brief", {}) if isinstance(row.get("analysis_brief"), dict) else {}
            signal = str(brief.get("signal", "hold") or "hold")
            confidence = float(brief.get("confidence", 0.0) or 0.0)
            expected_excess_return = float(brief.get("expected_excess_return", 0.0) or 0.0)
            rank_score = confidence * 0.7 + expected_excess_return * 0.3
            if rank_score > best_score:
                best_score = rank_score
                best_stock = stock_code

            normalized.append(
                {
                    "stock_code": stock_code,
                    "signal": signal,
                    "confidence": round(confidence, 4),
                    "expected_excess_return": round(expected_excess_return, 4),
                    "risk_flags": list(row.get("risk_flags", [])),
                    "risk_flag_count": len(list(row.get("risk_flags", []))),
                    "citation_count": len(list(row.get("citations", []))),
                    "answer_preview": str(row.get("answer", ""))[:180],
                    "cache_hit": bool(row.get("cache_hit", False)),
                }
            )

        return {
            "question": question,
            "count": len(normalized),
            "best_stock_code": best_stock,
            "items": normalized,
        }
