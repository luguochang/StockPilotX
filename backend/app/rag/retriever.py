from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import re
from typing import Iterable


@dataclass(slots=True)
class RetrievalItem:
    """检索条目对象。"""

    text: str
    source_id: str
    source_url: str
    score: float
    event_time: datetime
    reliability_score: float
    metadata: dict | None = None


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"[^\w\u4e00-\u9fff]+", text.lower()) if t]


def _char_ngrams(text: str, n: int = 2) -> set[str]:
    cleaned = re.sub(r"\s+", "", text.lower())
    if len(cleaned) < n:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}


class HybridRetriever:
    """Hybrid Retriever（BM25 + Vector + Rerank 的轻量实现）。

    说明：
    - BM25: 词项相关性
    - Vector: 字符 n-gram 的近似语义相似度
    - Rerank: 综合打分 + 可靠度校准
    """

    def __init__(self, corpus: list[RetrievalItem] | None = None) -> None:
        self._corpus = corpus or self._default_corpus()
        self._doc_tokens = [_tokenize(item.text) for item in self._corpus]
        self._avg_doc_len = sum(len(t) for t in self._doc_tokens) / max(1, len(self._doc_tokens))
        self._idf = self._build_idf(self._doc_tokens)

    def retrieve(self, query: str, top_k_vector: int = 12, top_k_bm25: int = 20, rerank_top_n: int = 10) -> list[RetrievalItem]:
        query_tokens = _tokenize(query)

        # 1) BM25 召回
        bm25_scored = []
        for idx, item in enumerate(self._corpus):
            score = self._bm25(query_tokens, self._doc_tokens[idx])
            bm25_scored.append((idx, score))
        bm25_scored.sort(key=lambda x: x[1], reverse=True)
        bm25_top = bm25_scored[:top_k_bm25]

        # 2) Vector 召回（n-gram Jaccard）
        q_ngrams = _char_ngrams(query)
        vec_scored = []
        for idx, item in enumerate(self._corpus):
            d_ngrams = _char_ngrams(item.text)
            inter = len(q_ngrams & d_ngrams)
            union = len(q_ngrams | d_ngrams) or 1
            score = inter / union
            vec_scored.append((idx, score))
        vec_scored.sort(key=lambda x: x[1], reverse=True)
        vec_top = vec_scored[:top_k_vector]

        # 3) 合并候选 + 重排
        candidate_ids = {idx for idx, _ in bm25_top} | {idx for idx, _ in vec_top}
        bm25_map = {idx: s for idx, s in bm25_top}
        vec_map = {idx: s for idx, s in vec_top}
        reranked: list[RetrievalItem] = []
        for idx in candidate_ids:
            item = self._corpus[idx]
            bm = bm25_map.get(idx, 0.0)
            vc = vec_map.get(idx, 0.0)
            # 综合分数：BM25(55%) + Vector(35%) + 可靠度(10%)
            score = (bm * 0.55) + (vc * 0.35) + (item.reliability_score * 0.10)
            reranked.append(
                RetrievalItem(
                    text=item.text,
                    source_id=item.source_id,
                    source_url=item.source_url,
                    score=round(score, 6),
                    event_time=item.event_time,
                    reliability_score=item.reliability_score,
                    metadata={"bm25": bm, "vector": vc},
                )
            )
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:rerank_top_n]

    @staticmethod
    def _build_idf(docs: list[list[str]]) -> dict[str, float]:
        n = len(docs)
        df: dict[str, int] = {}
        for tokens in docs:
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1
        return {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def _bm25(self, query_tokens: Iterable[str], doc_tokens: list[str], k1: float = 1.2, b: float = 0.75) -> float:
        tf: dict[str, int] = {}
        for token in doc_tokens:
            tf[token] = tf.get(token, 0) + 1
        score = 0.0
        doc_len = len(doc_tokens) or 1
        for token in query_tokens:
            if token not in tf:
                continue
            idf = self._idf.get(token, 0.0)
            freq = tf[token]
            denom = freq + k1 * (1 - b + b * doc_len / max(1.0, self._avg_doc_len))
            score += idf * (freq * (k1 + 1)) / denom
        return score

    @staticmethod
    def _default_corpus() -> list[RetrievalItem]:
        return [
            RetrievalItem(
                text="公司公告显示年度营收同比增长，现金流改善。",
                source_id="cninfo",
                source_url="https://www.cninfo.com.cn/",
                score=0.0,
                event_time=datetime(2025, 3, 28, tzinfo=timezone.utc),
                reliability_score=0.98,
            ),
            RetrievalItem(
                text="行业景气度修复，但原材料波动仍带来利润率压力。",
                source_id="eastmoney",
                source_url="https://www.eastmoney.com/",
                score=0.0,
                event_time=datetime(2025, 10, 11, tzinfo=timezone.utc),
                reliability_score=0.68,
            ),
            RetrievalItem(
                text="公司披露研发投入上升，产品结构升级。",
                source_id="sse_szse",
                source_url="https://www.sse.com.cn/",
                score=0.0,
                event_time=datetime(2025, 5, 20, tzinfo=timezone.utc),
                reliability_score=0.97,
            ),
            RetrievalItem(
                text="毛利率短期承压，但库存周转效率改善，经营韧性增强。",
                source_id="szse",
                source_url="https://www.szse.cn/",
                score=0.0,
                event_time=datetime(2025, 9, 18, tzinfo=timezone.utc),
                reliability_score=0.95,
            ),
        ]

