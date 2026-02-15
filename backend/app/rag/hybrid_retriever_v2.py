from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from backend.app.rag.retriever import HybridRetriever, RetrievalItem


SemanticSearchFn = Callable[[str, int], list[RetrievalItem]]


class HybridRetrieverV2:
    """混合检索 V2：词法召回 + 语义召回 + 统一重排。"""

    def __init__(
        self,
        *,
        corpus: list[RetrievalItem],
        semantic_search_fn: SemanticSearchFn | None = None,
    ) -> None:
        self._lexical = HybridRetriever(corpus=corpus)
        self._semantic_search_fn = semantic_search_fn

    def retrieve(
        self,
        query: str,
        top_k_vector: int = 12,
        top_k_bm25: int = 20,
        rerank_top_n: int = 10,
    ) -> list[RetrievalItem]:
        lexical_hits = self._lexical.retrieve(
            query=query,
            top_k_vector=top_k_vector,
            top_k_bm25=top_k_bm25,
            rerank_top_n=max(rerank_top_n, 12),
        )
        semantic_hits: list[RetrievalItem] = []
        if self._semantic_search_fn is not None:
            try:
                semantic_hits = self._semantic_search_fn(query, top_k_vector)
            except Exception:
                semantic_hits = []
        if not semantic_hits:
            return lexical_hits[:rerank_top_n]

        candidate: dict[tuple[str, str], RetrievalItem] = {}
        lexical_rank: dict[tuple[str, str], float] = {}
        semantic_rank: dict[tuple[str, str], float] = {}
        for idx, item in enumerate(lexical_hits):
            key = (item.source_id, item.text)
            candidate[key] = item
            lexical_rank[key] = max(lexical_rank.get(key, 0.0), self._rank_score(idx, len(lexical_hits)))
        for idx, item in enumerate(semantic_hits):
            key = (item.source_id, item.text)
            if key not in candidate:
                candidate[key] = item
            semantic_rank[key] = max(semantic_rank.get(key, 0.0), self._rank_score(idx, len(semantic_hits)))

        now = datetime.now(timezone.utc)
        merged: list[RetrievalItem] = []
        for key, item in candidate.items():
            lex = lexical_rank.get(key, 0.0)
            sem = semantic_rank.get(key, 0.0)
            freshness = self._freshness_score(item.event_time, now)
            score = 0.45 * lex + 0.35 * sem + 0.10 * float(item.reliability_score) + 0.10 * freshness
            meta = dict(item.metadata or {})
            meta.update(
                {
                    "lex_rank_score": round(lex, 4),
                    "sem_rank_score": round(sem, 4),
                    "freshness_score": round(freshness, 4),
                }
            )
            merged.append(
                RetrievalItem(
                    text=item.text,
                    source_id=item.source_id,
                    source_url=item.source_url,
                    score=round(score, 6),
                    event_time=item.event_time,
                    reliability_score=item.reliability_score,
                    metadata=meta,
                )
            )
        merged.sort(key=lambda x: x.score, reverse=True)
        return merged[:rerank_top_n]

    @staticmethod
    def _rank_score(idx: int, total: int) -> float:
        if total <= 0:
            return 0.0
        return max(0.0, 1.0 - (idx / max(1, total)))

    @staticmethod
    def _freshness_score(event_time: datetime, now: datetime) -> float:
        try:
            delta_days = abs((now - event_time).total_seconds()) / 86400.0
            return max(0.0, 1.0 - (delta_days / 180.0))
        except Exception:
            return 0.0
