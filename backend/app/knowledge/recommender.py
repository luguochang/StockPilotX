from __future__ import annotations

from collections import Counter
import re
from typing import Any


class DocumentRecommender:
    """Knowledge Hub document recommender (Phase1 minimal production version).

    Strategy:
    1) History-based relevance:
       - Learn user's frequently queried stock codes from query history.
    2) Context-based relevance:
       - Match requested stock/question keywords against doc chunk metadata/text.
    3) Graph-based relevance:
       - Boost chunks whose industry/concept words overlap graph neighbors.
    4) Rank + deduplicate:
       - Aggregate chunk-level scores to doc-level recommendations.
    """

    def recommend(
        self,
        *,
        chunks: list[dict[str, Any]],
        query_history_rows: list[dict[str, Any]],
        context: dict[str, Any],
        graph_terms: list[str],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        safe_top_k = max(1, min(30, int(top_k)))
        stock_code = str(context.get("stock_code", "")).strip().upper()
        question = str(context.get("question", "")).strip()
        question_terms = self._extract_terms(question)
        history_stocks = self._history_stock_counter(query_history_rows)

        # doc_id -> score details
        scored: dict[str, dict[str, Any]] = {}
        for chunk in chunks:
            doc_id = str(chunk.get("doc_id", "")).strip()
            if not doc_id:
                continue
            quality = float(chunk.get("quality_score", 0.0) or 0.0)
            chunk_text = str(chunk.get("chunk_text_redacted", "") or "")
            chunk_text_upper = chunk_text.upper()
            chunk_stocks = {str(x).strip().upper() for x in chunk.get("stock_codes", []) if str(x).strip()}

            score = quality * 0.35
            reasons: list[str] = []

            if stock_code and stock_code in chunk_stocks:
                score += 0.40
                reasons.append("stock_match")

            # Prefer docs related to user's high-frequency query stocks.
            history_boost = 0.0
            for code, freq in history_stocks.items():
                if code in chunk_stocks:
                    history_boost += min(0.20, 0.03 * freq)
            if history_boost > 0:
                score += history_boost
                reasons.append("history_match")

            # Keyword overlap with question text.
            if question_terms:
                overlap = sum(1 for term in question_terms if term in chunk_text_upper)
                if overlap > 0:
                    score += min(0.15, overlap * 0.03)
                    reasons.append("question_match")

            # Graph-concept overlap from stock neighborhood.
            if graph_terms:
                graph_hit = sum(1 for term in graph_terms if term and term in chunk_text_upper)
                if graph_hit > 0:
                    score += min(0.10, graph_hit * 0.02)
                    reasons.append("graph_match")

            # Aggregate chunk score into doc score (max + slight density bonus).
            node = scored.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "score": 0.0,
                    "match_count": 0,
                    "reasons": set(),
                    "stock_codes": set(),
                    "source": str(chunk.get("source", "")),
                    "updated_at": str(chunk.get("updated_at", "")),
                },
            )
            if score > float(node["score"]):
                node["score"] = score
            node["match_count"] = int(node["match_count"]) + (1 if reasons else 0)
            node["reasons"].update(reasons)
            node["stock_codes"].update(chunk_stocks)
            if str(chunk.get("updated_at", "")) > str(node.get("updated_at", "")):
                node["updated_at"] = str(chunk.get("updated_at", ""))

        ranked = []
        for _, item in scored.items():
            density_bonus = min(0.08, float(item["match_count"]) * 0.01)
            ranked.append(
                {
                    "doc_id": item["doc_id"],
                    "score": round(float(item["score"]) + density_bonus, 4),
                    "reasons": sorted(item["reasons"]),
                    "stock_codes": sorted(item["stock_codes"]),
                    "source": item["source"],
                    "updated_at": item["updated_at"],
                }
            )
        ranked.sort(key=lambda x: (float(x["score"]), str(x.get("updated_at", ""))), reverse=True)
        return ranked[:safe_top_k]

    @staticmethod
    def _history_stock_counter(rows: list[dict[str, Any]]) -> Counter[str]:
        counter: Counter[str] = Counter()
        for row in rows:
            for code in row.get("stock_codes", []) or []:
                token = str(code).strip().upper()
                if token:
                    counter[token] += 1
        return counter

    @staticmethod
    def _extract_terms(question: str) -> list[str]:
        # Keep Chinese/English/numeric tokens with enough length to avoid noisy one-char matches.
        terms = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]{2,12}", str(question or "").upper())
        stop = {"PLEASE", "ANALYZE", "分析", "请问", "请分析", "近期", "最新"}
        return [t for t in terms if t not in stop]

