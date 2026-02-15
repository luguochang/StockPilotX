from __future__ import annotations

import unittest

from backend.app.rag.evaluation import RetrievalEvaluator, default_retrieval_dataset
from backend.app.rag.retriever import HybridRetriever


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


if __name__ == "__main__":
    unittest.main()

