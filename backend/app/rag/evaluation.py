from __future__ import annotations

import math
from typing import Any

from backend.app.rag.retriever import HybridRetriever


class RetrievalEvaluator:
    """检索评测器：Recall@k / MRR / nDCG。"""

    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever

    def run(self, dataset: list[dict[str, Any]], k: int = 5) -> dict[str, float]:
        recalls = []
        mrrs = []
        ndcgs = []
        for case in dataset:
            query = case["query"]
            positives = set(case["positive_source_ids"])
            results = self.retriever.retrieve(query, rerank_top_n=k)
            pred = [item.source_id for item in results]
            recalls.append(self._recall_at_k(pred, positives))
            mrrs.append(self._mrr(pred, positives))
            ndcgs.append(self._ndcg_at_k(pred, positives, k))
        n = max(1, len(dataset))
        return {
            "recall_at_k": round(sum(recalls) / n, 4),
            "mrr": round(sum(mrrs) / n, 4),
            "ndcg_at_k": round(sum(ndcgs) / n, 4),
        }

    @staticmethod
    def _recall_at_k(pred: list[str], positives: set[str]) -> float:
        if not positives:
            return 0.0
        hit = sum(1 for p in pred if p in positives)
        return hit / len(positives)

    @staticmethod
    def _mrr(pred: list[str], positives: set[str]) -> float:
        for i, p in enumerate(pred, start=1):
            if p in positives:
                return 1.0 / i
        return 0.0

    @staticmethod
    def _ndcg_at_k(pred: list[str], positives: set[str], k: int) -> float:
        dcg = 0.0
        for i, p in enumerate(pred[:k], start=1):
            rel = 1.0 if p in positives else 0.0
            dcg += rel / math.log2(i + 1)
        ideal_hits = min(k, len(positives))
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
        if idcg == 0:
            return 0.0
        return dcg / idcg


def default_retrieval_dataset() -> list[dict[str, Any]]:
    """内置检索评测集（MVP 基线）。"""
    return [
        {"query": "营收增长与现金流改善", "positive_source_ids": ["cninfo"]},
        {"query": "行业波动利润率压力", "positive_source_ids": ["eastmoney"]},
        {"query": "研发投入上升 产品升级", "positive_source_ids": ["sse_szse"]},
        {"query": "毛利率承压 库存周转效率", "positive_source_ids": ["szse"]},
    ]

