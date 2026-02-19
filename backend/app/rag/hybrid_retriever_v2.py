from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from backend.app.rag.retriever import HybridRetriever, RetrievalItem, _tokenize


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
        # Stage-1: build a sufficiently large coarse candidate pool.
        coarse_pool_size = max(rerank_top_n * 4, 20)
        lexical_hits = self._lexical.retrieve(
            query=query,
            top_k_vector=max(top_k_vector, rerank_top_n * 3),
            top_k_bm25=max(top_k_bm25, rerank_top_n * 3),
            rerank_top_n=coarse_pool_size,
        )
        semantic_hits: list[RetrievalItem] = []
        if self._semantic_search_fn is not None:
            try:
                semantic_hits = self._semantic_search_fn(query, coarse_pool_size)
            except Exception:
                semantic_hits = []

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
        query_tokens = _tokenize(query)
        merged: list[RetrievalItem] = []
        # Stage-2: featureized reranking over coarse candidates.
        for key, item in candidate.items():
            lex = lexical_rank.get(key, 0.0)
            sem = semantic_rank.get(key, 0.0)
            freshness = self._freshness_score(item.event_time, now)
            overlap = self._token_overlap(query_tokens, _tokenize(item.text))
            # Combine lexical/semantic relevance with freshness/reliability and direct query overlap.
            score = (
                0.34 * lex
                + 0.28 * sem
                + 0.18 * overlap
                + 0.10 * float(item.reliability_score)
                + 0.10 * freshness
            )
            meta = dict(item.metadata or {})
            meta.update(
                {
                    "retrieval_stage": "coarse_to_rerank_v2",
                    "lex_rank_score": round(lex, 4),
                    "sem_rank_score": round(sem, 4),
                    "query_overlap_score": round(overlap, 4),
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

        # Stage-2b: source diversity penalty to avoid top-N being dominated by one source.
        diversified: list[RetrievalItem] = []
        source_count: dict[str, int] = {}
        for item in merged:
            sid = str(item.source_id or "")
            hit_no = source_count.get(sid, 0)
            penalty = 0.03 * hit_no
            final_score = round(max(0.0, float(item.score) - penalty), 6)
            source_count[sid] = hit_no + 1
            meta = dict(item.metadata or {})
            meta.update({"source_diversity_penalty": round(penalty, 4), "rerank_score": final_score})
            diversified.append(
                RetrievalItem(
                    text=item.text,
                    source_id=item.source_id,
                    source_url=item.source_url,
                    score=final_score,
                    event_time=item.event_time,
                    reliability_score=item.reliability_score,
                    metadata=meta,
                )
            )
        diversified.sort(key=lambda x: x.score, reverse=True)
        return diversified[:rerank_top_n]

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

    @staticmethod
    def _token_overlap(query_tokens: list[str], doc_tokens: list[str]) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0
        qset = set(query_tokens)
        dset = set(doc_tokens)
        if not qset:
            return 0.0
        return len(qset & dset) / len(qset)
