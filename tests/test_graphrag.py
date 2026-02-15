from __future__ import annotations

import unittest

from backend.app.rag.graphrag import GraphRAGService, InMemoryGraphStore


class GraphRagTestCase(unittest.TestCase):
    """RAG-002：GraphRAG 关系检索与引用测试。"""

    def test_query_subgraph_with_inmemory_store(self) -> None:
        svc = GraphRAGService(store=InMemoryGraphStore())
        result = svc.query_subgraph("分析产业链关系", ["SH600000"])
        self.assertEqual(result["mode"], "graph_rag")
        self.assertGreaterEqual(len(result["relations"]), 1)
        self.assertGreaterEqual(len(result["citations"]), 1)
        self.assertIn("summary", result)


if __name__ == "__main__":
    unittest.main()

