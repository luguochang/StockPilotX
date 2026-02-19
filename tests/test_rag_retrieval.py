from __future__ import annotations

from datetime import datetime, timezone
import unittest

from backend.app.rag.evaluation import RetrievalEvaluator, default_retrieval_dataset
from backend.app.rag.hybrid_retriever_v2 import HybridRetrieverV2
from backend.app.rag.retriever import HybridRetriever
from backend.app.rag.retriever import RetrievalItem


class RagRetrievalTestCase(unittest.TestCase):
    """RAG-001：Hybrid 检索与指标基线测试。"""

    def test_hybrid_retrieve_returns_ranked_items(self) -> None:
        retriever = HybridRetriever()
        items = retriever.retrieve("营收增长与现金流", rerank_top_n=3)
        self.assertEqual(len(items), 3)
        self.assertGreaterEqual(items[0].score, items[1].score)
        self.assertIn("bm25", items[0].metadata or {})
        self.assertIn("vector", items[0].metadata or {})

    def test_retrieval_metrics_baseline(self) -> None:
        retriever = HybridRetriever()
        evaluator = RetrievalEvaluator(retriever)
        metrics = evaluator.run(default_retrieval_dataset(), k=5)
        self.assertGreaterEqual(metrics["recall_at_k"], 0.8)
        self.assertGreaterEqual(metrics["mrr"], 0.5)
        self.assertGreaterEqual(metrics["ndcg_at_k"], 0.6)

    def test_hybrid_retriever_v2_has_two_stage_rerank_metadata(self) -> None:
        corpus = HybridRetriever()._default_corpus()

        def semantic_search(_: str, top_k: int) -> list[RetrievalItem]:
            out: list[RetrievalItem] = []
            for item in corpus[: max(1, min(top_k, 3))]:
                out.append(
                    RetrievalItem(
                        text=item.text,
                        source_id=item.source_id,
                        source_url=item.source_url,
                        score=0.8,
                        event_time=item.event_time if isinstance(item.event_time, datetime) else datetime.now(timezone.utc),
                        reliability_score=item.reliability_score,
                        metadata={"retrieval_track": "qa_summary"},
                    )
                )
            return out

        retriever = HybridRetrieverV2(corpus=corpus, semantic_search_fn=semantic_search)
        items = retriever.retrieve("营收增长 现金流 风险", rerank_top_n=3)
        self.assertEqual(len(items), 3)
        meta = items[0].metadata or {}
        self.assertEqual(str(meta.get("retrieval_stage", "")), "coarse_to_rerank_v2")
        self.assertIn("query_overlap_score", meta)
        self.assertIn("source_diversity_penalty", meta)
        self.assertIn("rerank_score", meta)


if __name__ == "__main__":
    unittest.main()
