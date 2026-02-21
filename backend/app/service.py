from __future__ import annotations

import base64
import csv
import difflib
import hashlib
import io
import json
import queue
import re
import threading
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import time
import uuid
from datetime import datetime, timedelta, timezone
from statistics import mean
from typing import Any, Callable

from backend.app.capabilities import build_capability_snapshot
from backend.app.agents.langgraph_runtime import build_workflow_runtime
from backend.app.agents.workflow import AgentWorkflow
from backend.app.config import Settings
from backend.app.datasources import (
    build_default_announcement_service,
    build_default_fund_service,
    build_default_financial_service,
    build_default_history_service,
    build_default_macro_service,
    build_default_news_service,
    build_default_quote_service,
    build_default_research_service,
)
from backend.app.data.ingestion import IngestionService, IngestionStore
from backend.app.data.scheduler import JobConfig, LocalJobScheduler
from backend.app.deepthink_exporter import DeepThinkReportExporter
from backend.app.knowledge.recommender import DocumentRecommender
from backend.app.evals.service import EvalService
from backend.app.llm.gateway import MultiProviderLLMGateway
from backend.app.memory.store import MemoryStore
from backend.app.middleware.hooks import BudgetMiddleware, GuardrailMiddleware, MiddlewareStack
from backend.app.models import Citation, QueryRequest, QueryResponse, ReportRequest, ReportResponse
from backend.app.observability.tracing import TraceStore
from backend.app.prompt.registry import PromptRegistry
from backend.app.prompt.runtime import PromptRuntime
from backend.app.predict.service import PredictionService, PredictionStore
from backend.app.query.comparator import QueryComparator
from backend.app.query.optimizer import QueryOptimizer
from backend.app.rag.evaluation import RetrievalEvaluator, default_retrieval_dataset
from backend.app.rag.graphrag import GraphRAGService
from backend.app.rag.embedding_provider import EmbeddingProvider, EmbeddingRuntimeConfig
from backend.app.rag.hybrid_retriever_v2 import HybridRetrieverV2
from backend.app.rag.parsing import DocumentParsingRouter
from backend.app.rag.retriever import HybridRetriever, RetrievalItem
from backend.app.rag.vector_store import LocalSummaryVectorStore, VectorSummaryRecord
from backend.app.state import AgentState
from backend.app.web.service import WebAppService
from backend.app.web.store import WebStore

from backend.app.service_modules import (
    RuntimeCoreMixin,
    QueryMixin,
    DataIngestionMixin,
    ReportMixin,
    RagMixin,
    PredictMixin,
    AnalysisMixin,
    JournalMixin,
    PortfolioWatchlistMixin,
    AuthSchedulerMixin,
    OpsMixin,
)


try:
    from openpyxl import load_workbook
except Exception:  # pragma: no cover - optional dependency
    load_workbook = None  # type: ignore[assignment]



class ReportTaskCancelled(RuntimeError):
    """Raised when the async report task is cancelled by user action."""



class AShareAgentService(
    RuntimeCoreMixin,
    QueryMixin,
    DataIngestionMixin,
    ReportMixin,
    RagMixin,
    PredictMixin,
    AnalysisMixin,
    JournalMixin,
    PortfolioWatchlistMixin,
    AuthSchedulerMixin,
    OpsMixin,
):
    """???????

    ???????? Service ???????????? service_modules?
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize service dependencies and runtime components."""
        self.settings = settings or Settings.from_env()
        self.traces = TraceStore()
        self.memory = MemoryStore(self.settings.memory_db_path)
        self.prompts = PromptRegistry(self.settings.prompt_db_path)
        self.prompt_runtime = PromptRuntime(self.prompts)
        self.eval_service = EvalService()
        self.llm_gateway = MultiProviderLLMGateway(settings=self.settings, trace_emit=self.traces.emit)
        self.web = WebAppService(
            store=WebStore(self.settings.web_db_path),
            jwt_secret=self.settings.jwt_secret,
            jwt_expire_seconds=self.settings.jwt_expire_seconds,
        )
        # Phase-B 鍚戦噺妫€绱㈢粍浠讹細鏀寔鍙厤缃?embedding provider + 鏈湴鍚戦噺绱㈠紩銆?
        self.embedding_provider = EmbeddingProvider(
            EmbeddingRuntimeConfig(
                provider=self.settings.embedding_provider,
                model=self.settings.embedding_model,
                base_url=self.settings.embedding_base_url,
                api_key=self.settings.embedding_api_key,
                dim=self.settings.embedding_dim,
                timeout_seconds=self.settings.embedding_timeout_seconds,
                batch_size=self.settings.embedding_batch_size,
                fallback_to_local=self.settings.embedding_fallback_to_local,
            ),
            trace_emit=self.traces.emit,
        )
        self.vector_store = LocalSummaryVectorStore(
            index_dir=self.settings.rag_vector_index_dir,
            embedding_provider=self.embedding_provider,
            dim=self.settings.embedding_dim,
            enable_faiss=self.settings.rag_vector_enabled,
        )
        self._vector_signature = ""
        self._vector_refreshed_at = 0.0
        self.document_parser = DocumentParsingRouter(prefer_docling=True)

        self.ingestion_store = IngestionStore()
        self.ingestion = IngestionService(
            # Keep datasource wiring centralized to make migration incremental.
            quote_service=build_default_quote_service(self.settings),
            announcement_service=build_default_announcement_service(self.settings),
            history_service=build_default_history_service(self.settings),
            financial_service=build_default_financial_service(self.settings),
            news_service=build_default_news_service(self.settings),
            research_service=build_default_research_service(self.settings),
            macro_service=build_default_macro_service(self.settings),
            fund_service=build_default_fund_service(self.settings),
            store=self.ingestion_store,
        )
        self.doc_recommender = DocumentRecommender()
        self.deepthink_exporter = DeepThinkReportExporter()
        self.prediction = PredictionService(
            quote_service=self.ingestion.quote_service,
            traces=self.traces,
            store=PredictionStore(),
            history_service=self.ingestion.history_service,
        )
        # Query Hub optimizer: in-memory cache + timeout guard.
        self.query_optimizer = QueryOptimizer(cache_size=300, ttl_seconds=180, timeout_seconds=30)
        self.scheduler = LocalJobScheduler()
        self._register_default_jobs()

        middleware = MiddlewareStack(
            middlewares=[GuardrailMiddleware(), BudgetMiddleware()],
            settings=self.settings,
        )
        self.workflow = AgentWorkflow(
            retriever=HybridRetriever(),
            graph_rag=GraphRAGService(),
            middleware_stack=middleware,
            trace_emit=self.traces.emit,
            prompt_renderer=lambda variables: self.prompt_runtime.build("fact_qa", variables),
            external_model_call=self.llm_gateway.generate,
            external_model_stream_call=self.llm_gateway.stream_generate,
            enable_local_fallback=self.settings.llm_fallback_to_local,
        )
        self.workflow_runtime = build_workflow_runtime(
            workflow=self.workflow,
            prefer_langgraph=self.settings.use_langgraph_runtime,
        )

        # 鎶ュ憡瀛樺偍锛歁VP 鍏堜娇鐢ㄥ唴瀛樺瓧鍏?
        self._reports: dict[str, dict[str, Any]] = {}
        # Report JSON bundle schema version used by export + version diff APIs.
        self._report_bundle_schema_version = "2.2.0"
        # Async report generation runtime state for /v1/report/tasks.
        self._report_task_lock = threading.RLock()
        self._report_tasks: dict[str, dict[str, Any]] = {}
        self._report_task_executor = ThreadPoolExecutor(max_workers=2)
        # Runtime guards: avoid tasks hanging in running/partial_ready without visible heartbeat.
        self._report_task_timeout_seconds = 300
        self._report_task_stall_seconds = 75
        self._report_task_heartbeat_interval_seconds = 1.0
        self._backtest_runs: dict[str, dict[str, Any]] = {}
        self._deep_archive_ts_format = "%Y-%m-%d %H:%M:%S"
        self._deep_archive_export_executor = ThreadPoolExecutor(max_workers=2)
        self._deep_round_mutex = threading.Lock()
        self._deep_round_inflight: set[str] = set()
        # In-memory datasource operation logs for /v1/datasources/* observability APIs.
        self._datasource_logs: list[dict[str, Any]] = []
        self._datasource_log_seq = 0
        self._register_default_agent_cards()


