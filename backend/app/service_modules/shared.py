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
from backend.app.query.sql_guard import SQLSafetyValidator
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

try:
    from openpyxl import load_workbook
except Exception:  # pragma: no cover - optional dependency
    load_workbook = None  # type: ignore[assignment]



class ReportTaskCancelled(RuntimeError):
    """Raised when the async report task is cancelled by user action."""


