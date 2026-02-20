from __future__ import annotations

from datetime import datetime, timezone
import tempfile
import time
import unittest

from backend.app.agents.workflow import AgentWorkflow, route_intent_with_confidence
from backend.app.config import Settings
from backend.app.memory.store import MemoryStore
from backend.app.middleware.hooks import BudgetMiddleware, GuardrailMiddleware, MiddlewareStack
from backend.app.rag.graphrag import GraphRAGService
from backend.app.rag.retriever import RetrievalItem
from backend.app.state import AgentState


class _SlowRetriever:
    def retrieve(self, query: str, *_args):  # noqa: ANN001
        if "risk dimension" in query:
            time.sleep(0.25)
        return [
            RetrievalItem(
                text=f"{query}-evidence",
                source_id="t",
                source_url="u",
                score=0.9,
                event_time=datetime.now(timezone.utc),
                reliability_score=0.9,
                metadata={"retrieval_track": "unit_test"},
            )
        ]


class _LowScoreThenRewriteRetriever:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def retrieve(self, query: str, *_args, **_kwargs):  # noqa: ANN001
        self.calls.append(query)
        score = 0.15
        text = "low-score-evidence"
        if "company fundamentals" in query:
            score = 0.91
            text = "rewritten-query-evidence"
        return [
            RetrievalItem(
                text=text,
                source_id="c",
                source_url="u",
                score=score,
                event_time=datetime.now(timezone.utc),
                reliability_score=0.9,
                metadata={"retrieval_track": "unit_test"},
            )
        ]


class Phase0OptimizationV11Tests(unittest.TestCase):
    def test_route_intent_with_confidence(self) -> None:
        result = route_intent_with_confidence("对比 SH600000 vs SZ000001 的估值")
        self.assertEqual(result.intent, "compare")
        self.assertGreaterEqual(result.confidence, 0.62)
        self.assertTrue(result.matched["compare"])

    def test_memory_store_ttl_similarity_and_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(f"{tmpdir}/memory.db")
            store.add_memory("u1", "task", {"question": "浦发银行走势", "summary": "银行板块回暖"})
            store.add_memory("u1", "task", {"question": "过期样本", "summary": "old"}, ttl_seconds=1)
            hits = store.similarity_search("u1", "浦发银行后续走势", top_k=3)
            self.assertTrue(hits)
            self.assertIn("similarity_score", hits[0])
            time.sleep(1.1)
            deleted = store.cleanup_expired()
            self.assertGreaterEqual(deleted, 1)
            stats = store.stats()
            self.assertGreaterEqual(stats.get("similarity_hit_rate", 0.0), 0.0)
            store.close()

    def test_deep_retrieve_timeout_isolation(self) -> None:
        settings = Settings(deep_subtask_timeout_seconds=0.05)
        middleware = MiddlewareStack([GuardrailMiddleware(), BudgetMiddleware()], settings=settings)
        workflow = AgentWorkflow(
            retriever=_SlowRetriever(),  # type: ignore[arg-type]
            graph_rag=GraphRAGService(),
            middleware_stack=middleware,
            trace_emit=lambda _trace_id, _name, _payload: None,
            external_model_call=None,
            external_model_stream_call=None,
            enable_local_fallback=True,
        )
        state = AgentState(user_id="u", question="q", trace_id="t")
        state.retrieval_plan = {"rerank_top_n": 5}
        _ = workflow._deep_retrieve("test question", state)
        timeout_subtasks = list(state.analysis.get("timeout_subtasks", []))
        self.assertTrue(any("risk dimension" in x for x in timeout_subtasks))

    def test_workflow_prepare_state_preserves_intent_confidence(self) -> None:
        settings = Settings(deep_subtask_timeout_seconds=0.2)
        middleware = MiddlewareStack([GuardrailMiddleware(), BudgetMiddleware()], settings=settings)
        workflow = AgentWorkflow(
            retriever=_SlowRetriever(),  # type: ignore[arg-type]
            graph_rag=GraphRAGService(),
            middleware_stack=middleware,
            trace_emit=lambda _trace_id, _name, _payload: None,
            external_model_call=None,
            external_model_stream_call=None,
            enable_local_fallback=True,
        )
        state = AgentState(
            user_id="u-ob",
            question="请做深度归因分析",
            stock_codes=["SH600000"],
            trace_id="trace-ob",
        )
        state.retrieval_plan = {"top_k_vector": 8, "top_k_bm25": 8, "rerank_top_n": 5}
        _ = workflow.prepare_prompt(state, memory_hint=[])
        self.assertGreater(float(state.analysis.get("intent_confidence", 0.0)), 0.0)
        self.assertIn("fact_count", state.analysis)

    def test_corrective_rag_rewrite_applies_on_low_score(self) -> None:
        retriever = _LowScoreThenRewriteRetriever()
        settings = Settings(corrective_rag_enabled=True, corrective_rag_rewrite_threshold=0.4)
        middleware = MiddlewareStack([GuardrailMiddleware(), BudgetMiddleware()], settings=settings)
        workflow = AgentWorkflow(
            retriever=retriever,  # type: ignore[arg-type]
            graph_rag=GraphRAGService(),
            middleware_stack=middleware,
            trace_emit=lambda _trace_id, _name, _payload: None,
            external_model_call=None,
            external_model_stream_call=None,
            enable_local_fallback=True,
        )
        state = AgentState(user_id="u", question="q", trace_id="t")
        state.retrieval_plan = {"rerank_top_n": 5}
        items = workflow._deep_retrieve("test question", state)
        self.assertTrue(bool(state.analysis.get("corrective_rag_applied", False)))
        self.assertTrue(any("rewritten-query-evidence" in x.text for x in items))

    def test_react_deep_mode_respects_max_iterations(self) -> None:
        retriever = _LowScoreThenRewriteRetriever()
        settings = Settings(
            react_deep_enabled=True,
            react_max_iterations=2,
            corrective_rag_enabled=False,
        )
        middleware = MiddlewareStack([GuardrailMiddleware(), BudgetMiddleware()], settings=settings)
        workflow = AgentWorkflow(
            retriever=retriever,  # type: ignore[arg-type]
            graph_rag=GraphRAGService(),
            middleware_stack=middleware,
            trace_emit=lambda _trace_id, _name, _payload: None,
            external_model_call=None,
            external_model_stream_call=None,
            enable_local_fallback=True,
        )
        state = AgentState(user_id="u", question="q", intent="deep", trace_id="t")
        state.retrieval_plan = {"rerank_top_n": 5}
        _ = workflow._deep_retrieve("test react question", state)
        self.assertEqual(int(state.analysis.get("react_iterations_planned", 0)), 2)
        self.assertEqual(int(state.analysis.get("react_iterations_executed", 0)), 2)
        self.assertGreaterEqual(len(retriever.calls), 6)


if __name__ == "__main__":
    unittest.main()
