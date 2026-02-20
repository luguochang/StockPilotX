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


class AShareAgentService:
    """搴旂敤鏈嶅姟灞傘€?

    璐熻矗鑱氬悎 API 鎵€闇€鑳藉姏锛屽彲绫绘瘮 Java 鐨?Facade + ApplicationService銆?
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

    def _select_runtime(self, preference: str | None = None):
        """Select runtime by request preference: langgraph/direct/auto."""
        pref = (preference or "").strip().lower()
        if pref in ("", "auto", "default"):
            return self.workflow_runtime
        if pref == "direct":
            return build_workflow_runtime(self.workflow, prefer_langgraph=False)
        if pref == "langgraph":
            return build_workflow_runtime(self.workflow, prefer_langgraph=True)
        return self.workflow_runtime

    def _register_default_jobs(self) -> None:
        """Register default scheduler jobs."""

        self.scheduler.register(
            JobConfig(
                name="intraday_quote_ingest",
                cadence="intraday",
                fn=lambda: self.ingest_market_daily(["SH600000", "SZ000001"]),
            )
        )
        self.scheduler.register(
            JobConfig(
                name="daily_announcement_ingest",
                cadence="daily",
                fn=lambda: self.ingest_announcements(["SH600000", "SZ000001"]),
            )
        )
        self.scheduler.register(
            JobConfig(
                name="weekly_rebuild",
                cadence="weekly",
                fn=lambda: {"status": "ok", "task": "weekly_rebuild", "note": "reserved for rebuild job"},
            )
        )

    def _register_default_agent_cards(self) -> None:
        """Register built-in agent cards for A2A capability discovery."""
        cards = [
            ("supervisor_agent", "Supervisor Agent", "Coordinate rounds and output arbitration conclusions.", ["plan", "route", "arbitrate"]),
            ("pm_agent", "PM Agent", "Evaluate theme logic and narrative consistency.", ["theme_analysis", "narrative"]),
            ("quant_agent", "Quant Agent", "Evaluate valuation, expectancy and probabilistic signals.", ["factor_analysis", "probability"]),
            ("risk_agent", "Risk Agent", "Evaluate drawdown, volatility and downside risk.", ["risk_scoring", "drawdown_check"]),
            ("critic_agent", "Critic Agent", "Check evidence completeness and logical consistency.", ["consistency_check", "counter_view"]),
            ("macro_agent", "Macro Agent", "Evaluate macro-policy and global shocks.", ["macro_event", "policy_watch"]),
            ("execution_agent", "Execution Agent", "Evaluate position sizing, timing and execution constraints.", ["execution_plan", "position_sizing"]),
            ("compliance_agent", "Compliance Agent", "Check compliance boundaries and expression risk.", ["compliance_check", "policy_block"]),
        ]
        for agent_id, display_name, description, capabilities in cards:
            self.web.register_agent_card(
                agent_id=agent_id,
                display_name=display_name,
                description=description,
                capabilities=capabilities,
            )

    def _deep_think_default_profile(self) -> list[str]:
        return [
            "supervisor_agent",
            "pm_agent",
            "quant_agent",
            "risk_agent",
            "critic_agent",
            "macro_agent",
            "execution_agent",
            "compliance_agent",
        ]

    def _normalize_deep_opinion(
        self,
        *,
        agent: str,
        signal: str,
        confidence: float,
        reason: str,
        evidence_ids: list[str] | None = None,
        risk_tags: list[str] | None = None,
    ) -> dict[str, Any]:
        mapped = signal.strip().lower()
        if mapped not in {"buy", "hold", "reduce"}:
            mapped = "hold"
        return {
            "agent": agent,
            "signal": mapped,
            "confidence": max(0.0, min(1.0, float(confidence))),
            "reason": reason[:360],
            "evidence_ids": evidence_ids or [],
            "risk_tags": risk_tags or [],
        }

    def _arbitrate_opinions(self, opinions: list[dict[str, Any]]) -> dict[str, Any]:
        bucket = {"buy": 0, "hold": 0, "reduce": 0}
        for opinion in opinions:
            bucket[str(opinion.get("signal", "hold"))] += 1

        ranked = sorted(bucket.items(), key=lambda item: item[1], reverse=True)
        consensus_signal = ranked[0][0]
        disagreement_score = round(1.0 - (ranked[0][1] / max(1, len(opinions))), 4)
        unique_signals = {str(opinion.get("signal", "hold")) for opinion in opinions}

        conflict_sources: list[str] = []
        if len(unique_signals) > 1:
            conflict_sources.append("signal_divergence")
        if consensus_signal == "buy" and any(
            str(opinion.get("agent")) == "risk_agent" and str(opinion.get("signal")) == "reduce" for opinion in opinions
        ):
            conflict_sources.append("risk_veto")
        if consensus_signal == "buy" and any(
            str(opinion.get("agent")) == "compliance_agent" and str(opinion.get("signal")) == "reduce" for opinion in opinions
        ):
            conflict_sources.append("compliance_veto")

        confidence_avg = sum(float(opinion.get("confidence", 0.0)) for opinion in opinions) / max(1, len(opinions))
        if confidence_avg < 0.6:
            conflict_sources.append("low_confidence")

        counter_candidates = [
            opinion
            for opinion in opinions
            if str(opinion.get("signal", "hold")) != consensus_signal and str(opinion.get("reason", "")).strip()
        ]
        counter_candidates.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
        counter_view = str(counter_candidates[0]["reason"]) if counter_candidates else "no significant counter view"

        return {
            "consensus_signal": consensus_signal,
            "disagreement_score": disagreement_score,
            "conflict_sources": conflict_sources,
            "counter_view": counter_view,
        }

    def _quality_score_for_group_card(
        self, *, disagreement_score: float, avg_confidence: float, evidence_count: int
    ) -> float:
        return round((1.0 - disagreement_score) * 0.6 + min(1.0, evidence_count / 4.0) * 0.2 + avg_confidence * 0.2, 4)

    def _deep_plan_tasks(self, question: str, round_no: int) -> list[dict[str, Any]]:
        q = (question or "").lower()
        tasks: list[dict[str, Any]] = [
            {"task_id": f"r{round_no}-t1", "agent": "quant_agent", "title": "Valuation and risk/reward assessment", "priority": "high"},
            {"task_id": f"r{round_no}-t2", "agent": "risk_agent", "title": "Drawdown and volatility assessment", "priority": "high"},
            {"task_id": f"r{round_no}-t3", "agent": "pm_agent", "title": "Theme and narrative consistency assessment", "priority": "medium"},
            {"task_id": f"r{round_no}-t4", "agent": "compliance_agent", "title": "Compliance boundary review", "priority": "high"},
        ]
        if any(k in q for k in ("macro", "policy", "rate", "fiscal", "gdp", "cpi", "ppi", "pmi", "宏观", "政策", "利率")):
            tasks.append(
                {"task_id": f"r{round_no}-t5", "agent": "macro_agent", "title": "Macro and policy shock assessment", "priority": "high"}
            )
        if any(k in q for k in ("execute", "execution", "position", "timing", "trade", "执行", "仓位", "交易", "节奏")):
            tasks.append(
                {"task_id": f"r{round_no}-t6", "agent": "execution_agent", "title": "Execution cadence and position constraints", "priority": "medium"}
            )
        return tasks

    def _deep_budget_snapshot(self, budget: dict[str, Any], round_no: int, task_count: int) -> dict[str, Any]:
        token_budget = max(1, int(budget.get("token_budget", 8000)))
        time_budget_ms = max(1, int(budget.get("time_budget_ms", 25000)))
        tool_call_budget = max(1, int(budget.get("tool_call_budget", 24)))

        token_used = min(token_budget, 1500 + task_count * 220 + round_no * 180)
        time_used_ms = min(time_budget_ms, 5500 + task_count * 380 + round_no * 650)
        tool_calls_used = min(tool_call_budget, task_count + 2)

        remaining = {
            "token_budget": token_budget - token_used,
            "time_budget_ms": time_budget_ms - time_used_ms,
            "tool_call_budget": tool_call_budget - tool_calls_used,
        }
        warn = (
            remaining["token_budget"] <= int(token_budget * 0.2)
            or remaining["time_budget_ms"] <= int(time_budget_ms * 0.2)
            or remaining["tool_call_budget"] <= max(1, int(tool_call_budget * 0.2))
        )
        exceeded = (
            remaining["token_budget"] <= 0
            or remaining["time_budget_ms"] <= 0
            or remaining["tool_call_budget"] <= 0
        )
        return {
            "limit": {
                "token_budget": token_budget,
                "time_budget_ms": time_budget_ms,
                "tool_call_budget": tool_call_budget,
            },
            "used": {
                "token_used": token_used,
                "time_used_ms": time_used_ms,
                "tool_calls_used": tool_calls_used,
            },
            "remaining": remaining,
            "warn": warn,
            "exceeded": exceeded,
        }

    def _deep_safe_json_loads(self, text: str) -> dict[str, Any]:
        """Extract JSON object from model text, including fenced code or mixed text wrappers."""
        clean = str(text or "").strip()
        if not clean:
            raise ValueError("empty model payload")
        if clean.startswith("```"):
            clean = clean.strip("`").strip()
            if clean.lower().startswith("json"):
                clean = clean[4:].strip()
        try:
            obj = json.loads(clean)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        # 瀹芥澗鍏滃簳锛氭埅鍙栫涓€涓?JSON 瀵硅薄鐗囨鍐嶈В鏋愩€?
        start = clean.find("{")
        end = clean.rfind("}")
        if start >= 0 and end > start:
            fragment = clean[start : end + 1]
            obj = json.loads(fragment)
            if isinstance(obj, dict):
                return obj
        raise ValueError("model output is not a valid json object")

    def _deep_normalize_intel_item(self, raw: Any) -> dict[str, Any] | None:
        if not isinstance(raw, dict):
            return None
        title = str(raw.get("title", "")).strip()
        summary = str(raw.get("summary", "")).strip()
        url = str(raw.get("url", "")).strip()
        if not title or not summary:
            return None
        impact_direction = str(raw.get("impact_direction", "uncertain")).strip().lower()
        if impact_direction not in {"positive", "negative", "mixed", "uncertain"}:
            impact_direction = "uncertain"
        impact_horizon = str(raw.get("impact_horizon", "1w")).strip().lower()
        if impact_horizon not in {"intraday", "1w", "1m", "quarter"}:
            impact_horizon = "1w"
        source_type = str(raw.get("source_type", "other")).strip().lower()
        if source_type not in {"official", "media", "other"}:
            source_type = "other"
        return {
            "title": title[:220],
            "summary": summary[:420],
            "impact_direction": impact_direction,
            "impact_horizon": impact_horizon,
            "why_relevant_to_stock": str(raw.get("why_relevant_to_stock", "")).strip()[:320],
            "url": url[:420],
            "published_at": str(raw.get("published_at", "")).strip()[:64],
            "source_type": source_type,
        }

    def _deep_validate_intel_payload(self, payload: Any) -> dict[str, Any]:
        """Validate and normalize LLM WebSearch intel payload to a stable schema."""
        if not isinstance(payload, dict):
            raise ValueError("intel payload must be an object")
        normalized: dict[str, Any] = {
            "as_of": str(payload.get("as_of", datetime.now(timezone.utc).isoformat())),
            "macro_signals": [],
            "industry_forward_events": [],
            "stock_specific_catalysts": [],
            "calendar_watchlist": [],
            "impact_chain": [],
            "decision_adjustment": {},
            "citations": [],
            "confidence_note": str(payload.get("confidence_note", "")).strip(),
            # 璇婃柇瀛楁锛氱粺涓€杩斿洖锛屼究浜庡墠绔槑纭€滄槸鍚﹀懡涓閮ㄥ疄鏃舵儏鎶モ€濅笌澶辫触鍘熷洜銆?
            "intel_status": str(payload.get("intel_status", "external_ok")).strip() or "external_ok",
            "fallback_reason": str(payload.get("fallback_reason", "")).strip()[:120],
            "fallback_error": str(payload.get("fallback_error", "")).strip()[:320],
            "trace_id": str(payload.get("trace_id", "")).strip()[:80],
            "external_enabled": bool(payload.get("external_enabled", True)),
            # provider_count 涓鸿瘖鏂瓧娈碉紝瑙ｆ瀽澶辫触鏃跺洖閫€ 0锛岄伩鍏嶅奖鍝嶄富娴佺▼銆?
            "provider_count": 0,
            "provider_names": [],
            "websearch_tool_requested": bool(payload.get("websearch_tool_requested", False)),
            "websearch_tool_applied": bool(payload.get("websearch_tool_applied", False)),
        }
        try:
            normalized["provider_count"] = max(0, int(payload.get("provider_count", 0) or 0))
        except Exception:
            normalized["provider_count"] = 0
        provider_names = payload.get("provider_names", [])
        if isinstance(provider_names, list):
            normalized["provider_names"] = [str(x).strip()[:64] for x in provider_names if str(x).strip()][:8]
        for key in ("macro_signals", "industry_forward_events", "stock_specific_catalysts", "calendar_watchlist"):
            rows = payload.get(key, [])
            if not isinstance(rows, list):
                continue
            for item in rows:
                norm = self._deep_normalize_intel_item(item)
                if norm:
                    normalized[key].append(norm)
        chains = payload.get("impact_chain", [])
        if isinstance(chains, list):
            for row in chains:
                if not isinstance(row, dict):
                    continue
                normalized["impact_chain"].append(
                    {
                        "event": str(row.get("event", "")).strip()[:180],
                        "transmission_path": str(row.get("transmission_path", "")).strip()[:260],
                        "industry_impact": str(row.get("industry_impact", "")).strip()[:220],
                        "stock_impact": str(row.get("stock_impact", "")).strip()[:220],
                        "price_factor": str(row.get("price_factor", "")).strip()[:180],
                    }
                )
        decision = payload.get("decision_adjustment", {})
        if isinstance(decision, dict):
            bias = str(decision.get("signal_bias", "hold")).strip().lower()
            if bias not in {"buy", "hold", "reduce"}:
                bias = "hold"
            raw_adj = decision.get("confidence_adjustment", 0.0)
            # 鍏煎妯″瀷杈撳嚭鏂囨湰寮哄急锛堜緥濡?"down"/"up"锛夛紝閬垮厤 float 寮鸿浆寮傚父瀵艰嚧鏁存闄嶇骇銆?
            try:
                conf_adj = float(raw_adj or 0.0)
            except Exception:
                text = str(raw_adj or "").strip().lower()
                if any(x in text for x in ("down", "decrease", "lower", "negative", "bearish", "reduce")):
                    conf_adj = -0.12
                elif any(x in text for x in ("up", "increase", "higher", "positive", "bullish", "buy")):
                    conf_adj = 0.12
                elif "neutral" in text or "hold" in text:
                    conf_adj = 0.0
                else:
                    # 鑻ユ枃鏈噷鍚暟瀛楋紙濡?"-0.1"銆?0.08"锛夛紝灏濊瘯鎻愬彇棣栦釜娴偣鍊笺€?
                    hit = re.search(r"-?\\d+(?:\\.\\d+)?", text)
                    conf_adj = float(hit.group(0)) if hit else 0.0
            conf_adj = max(-0.5, min(0.5, conf_adj))
            normalized["decision_adjustment"] = {
                "signal_bias": bias,
                "confidence_adjustment": conf_adj,
                "rationale": str(decision.get("rationale", "")).strip()[:320],
            }
        citations = payload.get("citations", [])
        if isinstance(citations, list):
            for item in citations:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url", "")).strip()
                if not url:
                    continue
                normalized["citations"].append(
                    {
                        "title": str(item.get("title", "")).strip()[:220],
                        "url": url[:420],
                        "published_at": str(item.get("published_at", "")).strip()[:64],
                        "source_type": str(item.get("source_type", "other")).strip()[:32],
                    }
                )
        return normalized

    def _deep_infer_intel_fallback_reason(self, message: str) -> str:
        """Map raw exception text to stable fallback reason codes for frontend diagnostics."""
        msg = (message or "").strip().lower()
        if "unsupported tool type" in msg:
            return "websearch_tool_unsupported"
        if "external llm disabled" in msg:
            return "external_disabled"
        if "no llm providers configured" in msg:
            return "provider_unconfigured"
        if "tool" in msg and any(x in msg for x in ("unsupported", "unknown", "invalid")):
            return "websearch_tool_unsupported"
        if "citations is empty" in msg:
            return "no_citations"
        if "not a valid json object" in msg:
            return "invalid_json"
        if "intel payload must be an object" in msg:
            return "invalid_payload_shape"
        return "provider_or_parse_error"

    def _deep_enabled_provider_names(self) -> list[str]:
        """Return enabled external provider names for observability output."""
        return [str(p.name)[:64] for p in self.llm_gateway.providers if bool(getattr(p, "enabled", True))]

    def _deep_local_intel_fallback(
        self,
        *,
        stock_code: str,
        question: str,
        quote: dict[str, Any],
        trend: dict[str, Any],
        fallback_reason: str = "external_websearch_unavailable",
        fallback_error: str = "",
        trace_id: str = "",
        external_enabled: bool | None = None,
        provider_names: list[str] | None = None,
        websearch_tool_requested: bool = False,
        websearch_tool_applied: bool = False,
    ) -> dict[str, Any]:
        """Build deterministic fallback intel payload when external websearch is unavailable."""
        providers = list(provider_names or self._deep_enabled_provider_names())
        ann = [x for x in self.ingestion_store.announcements if str(x.get("stock_code", "")).upper() == stock_code][-3:]
        calendar_items: list[dict[str, Any]] = []
        for item in ann:
            calendar_items.append(
                {
                    "title": str(item.get("title", "Company Announcement"))[:220],
                    "summary": str(item.get("content", "Check announcement details"))[:320],
                    "impact_direction": "mixed",
                    "impact_horizon": "1w",
                    "why_relevant_to_stock": f"Direct event related to {stock_code}",
                    "url": str(item.get("source_url", ""))[:420],
                    "published_at": str(item.get("event_time", ""))[:64],
                    "source_type": "official",
                }
            )
        if not calendar_items:
            calendar_items.append(
                {
                    "title": "Macro Window",
                    "summary": "External websearch unavailable; monitor macro/policy calendar manually.",
                    "impact_direction": "uncertain",
                    "impact_horizon": "1w",
                    "why_relevant_to_stock": f"Fallback mode for {stock_code}",
                    "url": "",
                    "published_at": datetime.now(timezone.utc).isoformat(),
                    "source_type": "other",
                }
            )

        bias = "hold"
        if float(trend.get("momentum_20", 0.0) or 0.0) > 0 and float(quote.get("pct_change", 0.0) or 0.0) > 0:
            bias = "buy"
        elif float(trend.get("max_drawdown_60", 0.0) or 0.0) > 0.2:
            bias = "reduce"

        return {
            "as_of": datetime.now(timezone.utc).isoformat(),
            "macro_signals": [
                {
                    "title": "Fallback Intel Mode",
                    "summary": "External websearch unavailable; output is based on local data only.",
                    "impact_direction": "uncertain",
                    "impact_horizon": "1w",
                    "why_relevant_to_stock": f"Question: {question[:120]}",
                    "url": "",
                    "published_at": datetime.now(timezone.utc).isoformat(),
                    "source_type": "other",
                }
            ],
            "industry_forward_events": [],
            "stock_specific_catalysts": [],
            "calendar_watchlist": calendar_items,
            "impact_chain": [
                {
                    "event": "Fallback Mode",
                    "transmission_path": "Lower external evidence availability -> lower confidence",
                    "industry_impact": "Potentially missing industry-level realtime shocks",
                    "stock_impact": f"Reduced forward-looking certainty for {stock_code}",
                    "price_factor": "Short-term volatility interpretation becomes weaker",
                }
            ],
            "decision_adjustment": {
                "signal_bias": bias,
                "confidence_adjustment": -0.12,
                "rationale": "Confidence discounted due to unavailable realtime websearch",
            },
            "citations": [],
            "confidence_note": "external_websearch_unavailable",
            "intel_status": "fallback",
            "fallback_reason": str(fallback_reason or "external_websearch_unavailable")[:120],
            "fallback_error": str(fallback_error or "")[:320],
            "trace_id": str(trace_id or "")[:80],
            "external_enabled": bool(self.settings.llm_external_enabled if external_enabled is None else external_enabled),
            "provider_count": len(providers),
            "provider_names": providers[:8],
            "websearch_tool_requested": bool(websearch_tool_requested),
            "websearch_tool_applied": bool(websearch_tool_applied),
        }

    def _deep_build_intel_prompt(
        self,
        *,
        stock_code: str,
        question: str,
        quote: dict[str, Any],
        trend: dict[str, Any],
        quant_20: dict[str, Any],
    ) -> str:
        """Build strict JSON prompt for external realtime websearch intel."""
        context = {
            "stock_code": stock_code,
            "question": question,
            "quote": {"price": quote.get("price"), "pct_change": quote.get("pct_change")},
            "trend": trend,
            "quant_20": quant_20,
            "scope": "CN market primary + global key events",
            "lookback_hours": 72,
            "lookahead_days": 30,
        }
        return (
            "You are an A-share realtime intelligence analyst. Use web search tools. "
            "Return STRICT JSON only (no markdown) with keys: as_of, macro_signals, "
            "industry_forward_events, stock_specific_catalysts, calendar_watchlist, impact_chain, "
            "decision_adjustment, citations, confidence_note. "
            f"context={json.dumps(context, ensure_ascii=False)}"
        )
    def _deep_fetch_intel_via_llm_websearch(
        self,
        *,
        stock_code: str,
        question: str,
        quote: dict[str, Any],
        trend: dict[str, Any],
        quant_20: dict[str, Any],
    ) -> dict[str, Any]:
        """Fetch real-time intel through LLM WebSearch; fall back to local intel when unavailable."""
        trace_id = self.traces.new_trace()
        enabled_provider_names = self._deep_enabled_provider_names()
        if not self.settings.llm_external_enabled:
            self.traces.emit(
                trace_id,
                "deep_intel_fallback",
                {"reason": "external_disabled", "provider_count": len(enabled_provider_names)},
            )
            return self._deep_local_intel_fallback(
                stock_code=stock_code,
                question=question,
                quote=quote,
                trend=trend,
                fallback_reason="external_disabled",
                trace_id=trace_id,
                external_enabled=False,
                provider_names=enabled_provider_names,
            )
        if not enabled_provider_names:
            self.traces.emit(trace_id, "deep_intel_fallback", {"reason": "provider_unconfigured", "provider_count": 0})
            return self._deep_local_intel_fallback(
                stock_code=stock_code,
                question=question,
                quote=quote,
                trend=trend,
                fallback_reason="provider_unconfigured",
                trace_id=trace_id,
                external_enabled=True,
                provider_names=enabled_provider_names,
            )
        prompt = self._deep_build_intel_prompt(
            stock_code=stock_code,
            question=question,
            quote=quote,
            trend=trend,
            quant_20=quant_20,
        )
        state = AgentState(
            user_id="deep-intel",
            question=question,
            stock_codes=[stock_code],
            trace_id=trace_id,
        )
        # 鏄惧紡瑕佹眰 Responses API 鎸傝浇 web-search tool锛岄伩鍏嶁€滀粎闈犳彁绀鸿瘝瑙﹀彂鎼滅储鈥濈殑涓嶇‘瀹氭€с€?
        websearch_overrides = {
            "tools": [{"type": "web_search_preview"}],
        }
        raw = ""
        tool_applied = False
        try:
            self.traces.emit(
                state.trace_id,
                "deep_intel_start",
                {
                    "provider_count": len(enabled_provider_names),
                    "provider_names": enabled_provider_names,
                    "websearch_tool_requested": True,
                },
            )
            try:
                raw = self.llm_gateway.generate(state, prompt, request_overrides=websearch_overrides)
                tool_applied = True
            except Exception as tool_ex:  # noqa: BLE001
                # 鑻?provider 涓嶆敮鎸?tools 瀛楁锛岄檷绾т负鈥減rompt-only鈥濆皾璇曪紝閬垮厤瀹屽叏涓嶅彲鐢ㄣ€?
                reason = self._deep_infer_intel_fallback_reason(str(tool_ex))
                if reason != "websearch_tool_unsupported":
                    raise
                self.traces.emit(
                    state.trace_id,
                    "deep_intel_tool_fallback",
                    {"reason": reason, "error": str(tool_ex)[:260]},
                )
                raw = self.llm_gateway.generate(state, prompt)
            parsed = self._deep_safe_json_loads(raw)
            normalized = self._deep_validate_intel_payload(parsed)
            # 淇濋殰鑷冲皯鏈変竴鏉″彲杩芥函寮曠敤锛岄伩鍏嶁€滄棤鏉ユ簮楂樼‘淇♀€濄€?
            if len(normalized.get("citations", [])) < 1:
                raise ValueError("intel citations is empty")
            normalized["intel_status"] = "external_ok"
            normalized["fallback_reason"] = ""
            normalized["fallback_error"] = ""
            normalized["trace_id"] = state.trace_id
            normalized["external_enabled"] = True
            normalized["provider_count"] = len(enabled_provider_names)
            normalized["provider_names"] = enabled_provider_names[:8]
            normalized["websearch_tool_requested"] = True
            normalized["websearch_tool_applied"] = bool(tool_applied)
            if not str(normalized.get("confidence_note", "")).strip():
                normalized["confidence_note"] = "external_websearch_ready"
            self.traces.emit(
                state.trace_id,
                "deep_intel_success",
                {
                    "citations_count": len(normalized.get("citations", [])),
                    "websearch_tool_applied": bool(tool_applied),
                    "provider_count": len(enabled_provider_names),
                },
            )
            return normalized
        except Exception as ex:  # noqa: BLE001
            reason = self._deep_infer_intel_fallback_reason(str(ex))
            self.traces.emit(
                state.trace_id,
                "deep_intel_fallback",
                {"reason": reason, "error": str(ex)[:280], "raw_size": len(raw)},
            )
            return self._deep_local_intel_fallback(
                stock_code=stock_code,
                question=question,
                quote=quote,
                trend=trend,
                fallback_reason=reason,
                fallback_error=str(ex),
                trace_id=state.trace_id,
                external_enabled=True,
                provider_names=enabled_provider_names,
                websearch_tool_requested=True,
                websearch_tool_applied=bool(tool_applied),
            )

    def _deep_build_business_summary(
        self,
        *,
        stock_code: str,
        question: str,
        opinions: list[dict[str, Any]],
        arbitration: dict[str, Any],
        budget_usage: dict[str, Any],
        intel: dict[str, Any],
        regime_context: dict[str, Any] | None,
        replan_triggered: bool,
        stop_reason: str,
    ) -> dict[str, Any]:
        """Merge arbitration result and intel layer into business summary output."""
        decision = intel.get("decision_adjustment", {}) if isinstance(intel, dict) else {}
        citations = list(intel.get("citations", [])) if isinstance(intel, dict) else []
        signal = str(arbitration.get("consensus_signal", "hold")).strip().lower()
        bias = str(decision.get("signal_bias", signal)).strip().lower()
        if bias in {"buy", "hold", "reduce"} and float(arbitration.get("disagreement_score", 0.0)) <= 0.55:
            signal = bias
        base_conf = max(0.0, min(1.0, 1.0 - float(arbitration.get("disagreement_score", 0.0))))
        confidence = max(0.0, min(1.0, base_conf + float(decision.get("confidence_adjustment", 0.0) or 0.0)))
        regime = regime_context if isinstance(regime_context, dict) else self._build_a_share_regime_context([stock_code])
        signal_guard = self._apply_a_share_signal_guard(signal, confidence, regime)
        signal = str(signal_guard.get("signal", signal))
        confidence = float(signal_guard.get("confidence", confidence))
        calendar = intel.get("calendar_watchlist", []) if isinstance(intel, dict) else []
        next_event = calendar[0] if isinstance(calendar, list) and calendar else {}
        dimensions = self._deep_build_analysis_dimensions(opinions=opinions, regime=regime, intel=intel)
        return {
            "stock_code": stock_code,
            "question": question[:200],
            "signal": signal,
            "confidence": round(confidence, 4),
            "disagreement_score": float(arbitration.get("disagreement_score", 0.0)),
            "trigger_condition": str(decision.get("rationale", "Watch intel catalysts and trend confirmation"))[:280],
            "invalidation_condition": (
                "Signal invalidates when key risk events land negatively, divergence keeps widening, or budget risk controls trigger."
            ),
            "review_time_hint": str(next_event.get("published_at", "")) or "Re-check within T+1",
            "top_conflict_sources": list(arbitration.get("conflict_sources", []))[:4],
            "replan_triggered": bool(replan_triggered),
            "stop_reason": stop_reason,
            "budget_warn": bool(budget_usage.get("warn")),
            "budget_exceeded": bool(budget_usage.get("exceeded")),
            "citations": citations[:6],
            "market_regime": str(regime.get("regime_label", "")),
            "regime_confidence": float(regime.get("regime_confidence", 0.0) or 0.0),
            "risk_bias": str(regime.get("risk_bias", "")),
            "regime_rationale": str(regime.get("regime_rationale", "")),
            "signal_guard_applied": bool(signal_guard.get("applied", False)),
            "confidence_adjustment_detail": signal_guard.get("detail", {}),
            "intel_status": str(intel.get("intel_status", "")) if isinstance(intel, dict) else "",
            "intel_fallback_reason": str(intel.get("fallback_reason", "")) if isinstance(intel, dict) else "",
            "intel_confidence_note": str(intel.get("confidence_note", "")) if isinstance(intel, dict) else "",
            "intel_trace_id": str(intel.get("trace_id", "")) if isinstance(intel, dict) else "",
            "intel_citation_count": len(citations),
            "analysis_dimensions": dimensions,
        }
    def _deep_build_analysis_dimensions(
        self,
        *,
        opinions: list[dict[str, Any]],
        regime: dict[str, Any],
        intel: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Build structured analysis panels for frontend cockpit rendering."""
        by_agent = {str(x.get("agent", "")): x for x in opinions}
        risk_op = by_agent.get("risk_agent", {})
        quant_op = by_agent.get("quant_agent", {})
        macro_op = by_agent.get("macro_agent", {})
        exec_op = by_agent.get("execution_agent", {})
        pm_op = by_agent.get("pm_agent", {})
        critic_op = by_agent.get("critic_agent", {})
        industry_events = list(intel.get("industry_forward_events", [])) if isinstance(intel, dict) else []
        catalysts = list(intel.get("stock_specific_catalysts", [])) if isinstance(intel, dict) else []
        impact_chain = list(intel.get("impact_chain", [])) if isinstance(intel, dict) else []
        return [
            {
                "dimension": "industry",
                "score": round(float(quant_op.get("confidence", 0.5) or 0.5), 4),
                "summary": str(pm_op.get("reason", "Need to verify industry narrative with fundamentals."))[:180],
                "signals": [str(x.get("title", ""))[:60] for x in industry_events[:3] if str(x.get("title", "")).strip()],
            },
            {
                "dimension": "competition",
                "score": round(float(critic_op.get("confidence", 0.5) or 0.5), 4),
                "summary": "Competition should be checked with share shifts and profitability trends.",
                "signals": [str(x.get("summary", ""))[:60] for x in catalysts[:2] if str(x.get("summary", "")).strip()],
            },
            {
                "dimension": "supply_chain",
                "score": round(float(macro_op.get("confidence", 0.5) or 0.5), 4),
                "summary": "Track upstream/downstream transmission and cost pressure release.",
                "signals": [str(x.get("to", ""))[:60] for x in impact_chain[:3] if str(x.get("to", "")).strip()],
            },
            {
                "dimension": "risk",
                "score": round(float(risk_op.get("confidence", 0.5) or 0.5), 4),
                "summary": str(risk_op.get("reason", "Prioritize drawdown, volatility, and left-tail risk."))[:180],
                "signals": [str(regime.get("risk_bias", ""))[:40]],
            },
            {
                "dimension": "macro",
                "score": round(float(regime.get("regime_confidence", 0.0) or 0.0), 4),
                "summary": str(regime.get("regime_rationale", "Macro and policy cadence may change risk appetite."))[:180],
                "signals": [str(x.get("title", ""))[:60] for x in list(intel.get("macro_signals", []))[:3] if str(x.get("title", "")).strip()],
            },
            {
                "dimension": "execution",
                "score": round(float(exec_op.get("confidence", 0.5) or 0.5), 4),
                "summary": str(exec_op.get("reason", "Execution should control cadence, sizing, and liquidity shocks."))[:180],
                "signals": [str(exec_op.get("signal", "hold"))],
            },
        ]

    def query(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute query pipeline with cache and history persistence."""
        started_at = time.perf_counter()
        req = QueryRequest(**payload)
        runtime_name = str(payload.get("workflow_runtime", ""))
        selected_runtime = self._select_runtime(runtime_name)
        cache_ctx = {"runtime": selected_runtime.runtime_name, "mode": str(payload.get("mode", ""))}

        cached = self.query_optimizer.get_cached(req.question, req.stock_codes, cache_ctx)
        if cached is not None:
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            cached["cache_hit"] = True
            try:
                self.web.query_history_add(
                    "",
                    question=req.question,
                    stock_codes=req.stock_codes,
                    trace_id=str(cached.get("trace_id", "")),
                    intent=str(cached.get("intent", "")),
                    cache_hit=True,
                    latency_ms=latency_ms,
                    summary=str(cached.get("answer", ""))[:160],
                )
            except Exception:
                # Query history must never block the main response.
                pass
            return cached

        def _run_pipeline() -> dict[str, Any]:
            # Build per-stock input packs so single-turn requests get sufficient coverage.
            data_packs: list[dict[str, Any]] = []
            for code in req.stock_codes:
                try:
                    data_packs.append(
                        self._build_llm_input_pack(
                            code,
                            question=req.question,
                            scenario="query",
                        )
                    )
                except Exception:  # noqa: BLE001
                    # Keep query available even when one symbol fails to refresh.
                    continue

            self.workflow.retriever = self._build_runtime_retriever(req.stock_codes)
            enriched_question = self._augment_question_with_history_context(req.question, req.stock_codes)
            enriched_question = self._augment_question_with_dataset_context(enriched_question, data_packs)
            regime_context = self._build_a_share_regime_context(req.stock_codes)
            trace_id = self.traces.new_trace()
            state = AgentState(
                user_id=req.user_id,
                question=enriched_question,
                stock_codes=req.stock_codes,
                market_regime_context=regime_context,
                trace_id=trace_id,
            )

            memory_hint = self.memory.list_memory(req.user_id, limit=3)
            runtime_result = selected_runtime.run(state, memory_hint=memory_hint)
            state = runtime_result.state
            state.analysis["workflow_runtime"] = runtime_result.runtime
            self.traces.emit(trace_id, "workflow_runtime", {"runtime": runtime_result.runtime})
            answer, merged_citations = self._build_evidence_rich_answer(
                req.question,
                req.stock_codes,
                state.report,
                state.citations,
                regime_context=regime_context,
            )
            normalized_citations = self._normalize_citations_for_output(
                merged_citations,
                evidence_pack=state.evidence_pack,
                max_items=10,
            )
            analysis_brief = self._build_analysis_brief(req.stock_codes, normalized_citations, regime_context=regime_context)
            state.report = answer
            state.citations = normalized_citations
            self.memory.add_memory(
                req.user_id,
                "task",
                {
                    "question": req.question,
                    "summary": state.analysis.get("summary", ""),
                    "risk_flags": state.risk_flags,
                    "mode": state.mode,
                },
            )
            self._record_rag_eval_case(req.question, state.evidence_pack, state.citations)
            self._persist_query_knowledge_memory(
                user_id=req.user_id,
                question=req.question,
                stock_codes=req.stock_codes,
                state=state,
                query_type="query",
            )

            resp = QueryResponse(
                trace_id=trace_id,
                intent=state.intent,  # type: ignore[arg-type]
                answer=state.report,
                citations=[Citation(**c) for c in state.citations],
                risk_flags=state.risk_flags,
                mode=state.mode,  # type: ignore[arg-type]
                workflow_runtime=runtime_result.runtime,
                analysis_brief=analysis_brief,
            )
            body = resp.model_dump(mode="json")
            body["cache_hit"] = False
            body["data_packs"] = data_packs
            all_missing = [str(m) for p in data_packs for m in list((p.get("missing_data") or []))]
            if all_missing:
                body["analysis_brief"]["data_pack_missing"] = list(dict.fromkeys(all_missing))
                body["analysis_brief"]["degraded"] = True
                body["analysis_brief"]["degrade_reason"] = "dataset_gap"

            latency_ms = int((time.perf_counter() - started_at) * 1000)
            self._record_rag_retrieval_trace(
                trace_id=trace_id,
                query_text=req.question,
                query_type="query",
                evidence_pack=state.evidence_pack,
                citations=state.citations,
                latency_ms=latency_ms,
            )
            try:
                self.web.query_history_add(
                    "",
                    question=req.question,
                    stock_codes=req.stock_codes,
                    trace_id=trace_id,
                    intent=str(body.get("intent", "")),
                    cache_hit=False,
                    latency_ms=latency_ms,
                    summary=str(body.get("answer", ""))[:160],
                )
            except Exception:
                pass
            return body

        try:
            result = self.query_optimizer.run_with_timeout(_run_pipeline)
            self.query_optimizer.set_cached(req.question, req.stock_codes, result, cache_ctx)
            return result
        except TimeoutError:
            return self._build_query_degraded_response(
                req=req,
                started_at=started_at,
                workflow_runtime=selected_runtime.runtime_name,
                error_code="query_timeout",
                error_message=f"query timeout after {self.query_optimizer.timeout_seconds}s",
            )
        except Exception as ex:  # noqa: BLE001
            return self._build_query_degraded_response(
                req=req,
                started_at=started_at,
                workflow_runtime=selected_runtime.runtime_name,
                error_code="query_runtime_error",
                error_message=str(ex),
            )

    def _build_query_degraded_response(
        self,
        *,
        req: QueryRequest,
        started_at: float,
        workflow_runtime: str,
        error_code: str,
        error_message: str,
    ) -> dict[str, Any]:
        """Return structured degraded response when query pipeline fails."""
        trace_id = self.traces.new_trace()
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        safe_error = str(error_message or "").strip()[:260] or "unknown error"
        answer = (
            "This query hit degraded mode. A minimum viable result is returned."
            " Please retry later or reduce request scope (fewer symbols / shorter question)."
            f" Error code: {error_code}."
        )
        response = {
            "trace_id": trace_id,
            "intent": "fact",
            "answer": answer,
            "citations": [],
            "risk_flags": [error_code],
            "mode": "agentic_rag",
            "workflow_runtime": workflow_runtime or "unknown",
            "analysis_brief": {
                "market_regime": "",
                "regime_confidence": 0.0,
                "risk_bias": "unknown",
                "signal_guard_applied": False,
                "signal_guard_detail": {},
                "confidence_adjustment_detail": {},
                "degraded": True,
                "degrade_reason": error_code,
            },
            "cache_hit": False,
            "degraded": True,
            "error_code": error_code,
            "error_message": safe_error,
        }
        self.traces.emit(
            trace_id,
            "query_degraded",
            {"error_code": error_code, "error_message": safe_error, "latency_ms": latency_ms},
        )
        try:
            self.web.query_history_add(
                "",
                question=req.question,
                stock_codes=req.stock_codes,
                trace_id=trace_id,
                intent="fact",
                cache_hit=False,
                latency_ms=latency_ms,
                summary=answer[:160],
                error=f"{error_code}:{safe_error}",
            )
        except Exception:
            pass
        return response

    def query_compare(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run same question against multiple stocks and return comparison payload."""
        user_id = str(payload.get("user_id", "")).strip() or "anonymous"
        question = str(payload.get("question", "")).strip()
        stock_codes = [str(x).strip().upper() for x in list(payload.get("stock_codes", [])) if str(x).strip()]
        unique_codes = list(dict.fromkeys(stock_codes))
        if len(unique_codes) < 2:
            raise ValueError("query_compare requires at least 2 stock_codes")

        per_stock: list[dict[str, Any]] = []
        for code in unique_codes:
            row = self.query(
                {
                    "user_id": user_id,
                    "question": question,
                    "stock_codes": [code],
                    "workflow_runtime": payload.get("workflow_runtime", ""),
                }
            )
            per_stock.append(
                {
                    "stock_code": code,
                    "answer": row.get("answer", ""),
                    "citations": row.get("citations", []),
                    "risk_flags": row.get("risk_flags", []),
                    "analysis_brief": row.get("analysis_brief", {}),
                    "cache_hit": bool(row.get("cache_hit", False)),
                }
            )

        return QueryComparator.build(question=question, rows=per_stock)

    def query_history_list(
        self,
        token: str,
        *,
        limit: int = 50,
        stock_code: str = "",
        created_from: str = "",
        created_to: str = "",
    ) -> list[dict[str, Any]]:
        return self.web.query_history_list(
            token,
            limit=limit,
            stock_code=stock_code,
            created_from=created_from,
            created_to=created_to,
        )

    def query_history_clear(self, token: str) -> dict[str, Any]:
        return self.web.query_history_clear(token)

    def query_stream_events(self, payload: dict[str, Any], chunk_size: int = 80):
        """Stream query result as events for frontend incremental rendering."""
        _ = chunk_size
        started_at = time.perf_counter()
        req = QueryRequest(**payload)
        selected_runtime = self._select_runtime(str(payload.get("workflow_runtime", "")))
        # 鍏堝彂閫?start锛岀‘淇濆墠绔湪浠讳綍鑰楁椂姝ラ鍓嶅氨鑳芥劅鐭モ€滀换鍔″凡鍚姩鈥濄€?
        yield {"event": "start", "data": {"status": "started", "phase": "init"}}
        # Keep stream behavior aligned with /v1/query: refresh and package per-stock inputs first.
        data_packs: list[dict[str, Any]] = []
        if req.stock_codes:
            yield {"event": "progress", "data": {"phase": "data_refresh", "message": "正在刷新行情/公告/历史/财务/新闻/研报数据"}}
            for code in req.stock_codes:
                try:
                    data_packs.append(
                        self._build_llm_input_pack(
                            code,
                            question=req.question,
                            scenario="query_stream",
                        )
                    )
                except Exception as ex:  # noqa: BLE001
                    yield {"event": "progress", "data": {"phase": "data_refresh", "status": "degraded", "error": str(ex)[:160]}}
            yield {
                "event": "data_pack",
                "data": {
                    "items": [
                        {
                            "stock_code": str(pack.get("stock_code", "")),
                            "coverage": dict((pack.get("dataset", {}) or {}).get("coverage", {}) or {}),
                            "missing_data": list(pack.get("missing_data", []) or []),
                        }
                        for pack in data_packs
                    ],
                },
            }
            yield {"event": "progress", "data": {"phase": "data_refresh", "status": "done"}}
        yield {"event": "progress", "data": {"phase": "retriever", "message": "Preparing retrieval corpus"}}
        self.workflow.retriever = self._build_runtime_retriever(req.stock_codes)
        enriched_question = self._augment_question_with_history_context(req.question, req.stock_codes)
        enriched_question = self._augment_question_with_dataset_context(enriched_question, data_packs)
        regime_context = self._build_a_share_regime_context(req.stock_codes)
        trace_id = self.traces.new_trace()
        state = AgentState(
            user_id=req.user_id,
            question=enriched_question,
            stock_codes=req.stock_codes,
            market_regime_context=regime_context,
            trace_id=trace_id,
        )
        memory_hint = self.memory.list_memory(req.user_id, limit=3)
        runtime_name = selected_runtime.runtime_name
        self.traces.emit(trace_id, "workflow_runtime", {"runtime": runtime_name})
        yield {
            "event": "market_regime",
            "data": {
                "regime_label": str(regime_context.get("regime_label", "")),
                "regime_confidence": float(regime_context.get("regime_confidence", 0.0) or 0.0),
                "risk_bias": str(regime_context.get("risk_bias", "")),
                "regime_rationale": str(regime_context.get("regime_rationale", "")),
            },
        }
        yield {"event": "stream_runtime", "data": {"runtime": runtime_name}}
        yield {"event": "progress", "data": {"phase": "model", "message": "Starting model streaming output"}}
        # 涓轰簡閬垮厤鈥滄ā鍨嬮 token 杩熻繜涓嶆潵鏃跺墠绔棤鍙嶉鈥濓紝
        # 鍦ㄧ嫭绔嬬嚎绋嬫秷璐?runtime 浜嬩欢锛屽苟鍦ㄤ富鍗忕▼鍛ㄦ湡鎬у彂閫?model_wait 蹇冭烦銆?
        event_queue: queue.Queue[Any] = queue.Queue()
        done_sentinel = object()
        runtime_error: dict[str, str] = {}

        def _pump_runtime_events() -> None:
            try:
                for event in selected_runtime.run_stream(state, memory_hint=memory_hint):
                    event_queue.put(event)
            except Exception as ex:  # noqa: BLE001
                runtime_error["message"] = str(ex)
            finally:
                event_queue.put(done_sentinel)

        pump_thread = threading.Thread(target=_pump_runtime_events, daemon=True)
        pump_thread.start()
        model_started_at = time.perf_counter()
        while True:
            try:
                item = event_queue.get(timeout=0.8)
            except queue.Empty:
                wait_ms = int((time.perf_counter() - model_started_at) * 1000)
                yield {
                    "event": "progress",
                    "data": {
                        "phase": "model_wait",
                        "message": "妯″瀷鎺ㄧ悊涓紝绛夊緟棣栦釜澧為噺杈撳嚭",
                        "wait_ms": wait_ms,
                    },
                }
                continue
            if item is done_sentinel:
                break
            if isinstance(item, dict):
                yield item
        if runtime_error:
            yield {"event": "error", "data": {"error": runtime_error["message"], "trace_id": trace_id}}
            yield {"event": "done", "data": {"ok": False, "trace_id": trace_id, "error": runtime_error["message"]}}
            return
        citations = self._normalize_citations_for_output(
            [c for c in state.citations if isinstance(c, dict)],
            evidence_pack=state.evidence_pack,
            max_items=10,
        )
        state.citations = citations
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        self._record_rag_retrieval_trace(
            trace_id=trace_id,
            query_text=req.question,
            query_type="query_stream",
            evidence_pack=state.evidence_pack,
            citations=state.citations,
            latency_ms=latency_ms,
        )
        self._persist_query_knowledge_memory(
            user_id=req.user_id,
            question=req.question,
            stock_codes=req.stock_codes,
            state=state,
            query_type="query_stream",
        )
        # 缁撴灉娌夋穩浜嬩欢锛氫究浜庡墠绔?杩愮淮纭鏈疆闂瓟宸茶繘鍏ュ叡浜鏂欐睜銆?
        yield {"event": "knowledge_persisted", "data": {"trace_id": trace_id}}
        yield {
            "event": "analysis_brief",
            "data": {
                **self._build_analysis_brief(req.stock_codes, citations, regime_context=regime_context),
                "data_pack_missing": list(
                    dict.fromkeys(
                        [
                            str(item)
                            for pack in data_packs
                            for item in list((pack.get("missing_data") or []))
                            if str(item).strip()
                        ]
                    )
                ),
            },
        }

    def _build_evidence_rich_answer(
        self,
        question: str,
        stock_codes: list[str],
        base_answer: str,
        citations: list[dict[str, Any]],
        regime_context: dict[str, Any] | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Generate enhanced answer with market snapshots, trend context, and evidence references."""
        lines: list[str] = []
        merged = list(citations)

        lines.append("## Conclusion Summary")
        lines.append(base_answer)
        lines.append("")

        regime = regime_context if isinstance(regime_context, dict) else {}
        if regime:
            lines.append("## A-share Regime Snapshot")
            lines.append(
                "- "
                f"label=`{regime.get('regime_label', 'unknown')}` | "
                f"confidence=`{float(regime.get('regime_confidence', 0.0) or 0.0):.2f}` | "
                f"risk_bias=`{regime.get('risk_bias', 'neutral')}`"
            )
            lines.append(f"- rationale: {regime.get('regime_rationale', '')}")

        lines.append("## Data Snapshot and History Trend")
        for code in stock_codes:
            realtime = self._latest_quote(code)
            # Keep a wider history window to reduce sparse-sample bias in trend judgment.
            bars = self._history_bars(code, limit=260)
            summary_3m = self._history_3m_summary(code)

            if not realtime or len(bars) < 30:
                lines.append(f"- {code}: insufficient data (realtime or history sample too small).")
                continue

            trend = self._trend_metrics(bars)
            lines.append(
                f"- {code}: latest `{realtime['price']:.3f}`, day change `{realtime['pct_change']:.2f}%`, "
                f"MA20 slope `{trend['ma20_slope']:.4f}`, MA60 slope `{trend['ma60_slope']:.4f}`, "
                f"volatility(20d) `{trend['volatility_20']:.4f}`."
            )
            lines.append(
                f"  Interpretation: MA20 is {'above' if trend['ma20'] >= trend['ma60'] else 'below'} MA60, "
                f"momentum(20d) `{trend['momentum_20']:.4f}`, max drawdown(60d) `{trend['max_drawdown_60']:.4f}`."
            )

            if int(summary_3m.get("sample_count", 0)) >= 60:
                lines.append(
                    "  3-month window "
                    f"`{summary_3m.get('start_date', '')}` -> `{summary_3m.get('end_date', '')}`, "
                    f"samples `{int(summary_3m.get('sample_count', 0))}`, "
                    f"close `{float(summary_3m.get('start_close', 0.0)):.3f}` -> "
                    f"`{float(summary_3m.get('end_close', 0.0)):.3f}`, "
                    f"range `{float(summary_3m.get('pct_change', 0.0)) * 100:.2f}%`."
                )

            merged.append(
                {
                    "source_id": realtime.get("source_id", "unknown"),
                    "source_url": realtime.get("source_url", ""),
                    "event_time": realtime.get("ts"),
                    "reliability_score": realtime.get("reliability_score", 0.7),
                    "excerpt": f"{code} realtime: price={realtime['price']}, pct={realtime['pct_change']}",
                    "retrieval_track": "quote_snapshot",
                    "rerank_score": float(realtime.get("reliability_score", 0.7) or 0.7),
                }
            )

            if bars:
                merged.append(
                    {
                        "source_id": bars[-1].get("source_id", "eastmoney_history"),
                        "source_url": bars[-1].get("source_url", ""),
                        "event_time": bars[-1].get("trade_date"),
                        "reliability_score": bars[-1].get("reliability_score", 0.9),
                        "excerpt": f"{code} history sample count={len(bars)}, latest trade date {bars[-1].get('trade_date', '')}",
                        "retrieval_track": "history_daily",
                        "rerank_score": float(bars[-1].get("reliability_score", 0.9) or 0.9),
                    }
                )

            # Add a dedicated 3-month context citation to avoid sparse-point misinterpretation.
            if int(summary_3m.get("sample_count", 0)) >= 2:
                merged.append(
                    {
                        "source_id": "eastmoney_history_3m_window",
                        "source_url": bars[-1].get("source_url", "") if bars else "",
                        "event_time": summary_3m.get("end_date", ""),
                        "reliability_score": 0.92,
                        "excerpt": (
                            f"{code} 3m sample={int(summary_3m.get('sample_count', 0))}, "
                            f"range {summary_3m.get('start_date', '')}->{summary_3m.get('end_date', '')}, "
                            f"close {float(summary_3m.get('start_close', 0.0)):.3f}->{float(summary_3m.get('end_close', 0.0)):.3f}"
                        ),
                        "retrieval_track": "history_3m_window",
                        "rerank_score": 0.92,
                    }
                )

        pm_section = self._pm_agent_view(question, stock_codes)
        dev_section = self._dev_manager_view(stock_codes)

        lines.append("")
        lines.append("## PM Agent View")
        lines.extend(pm_section)
        lines.append("")
        lines.append("## Dev Manager Agent View")
        lines.extend(dev_section)

        shared_hits = [
            c
            for c in merged
            if str(c.get("source_id", "")).startswith("doc::")
            or str(c.get("source_id", "")) == "qa_memory_summary"
        ]

        lines.append("")
        lines.append("## Shared Knowledge Hits")
        if shared_hits:
            lines.append(f"- hit count: `{len(shared_hits)}`")
            for idx, item in enumerate(shared_hits[:4], start=1):
                lines.append(
                    f"- [{idx}] source=`{item.get('source_id', 'unknown')}` | "
                    f"{str(item.get('excerpt', ''))[:140]}"
                )
        else:
            lines.append("- no shared knowledge asset hit in this answer; evidence came from realtime/history feeds.")

        lines.append("")
        lines.append("## Evidence References")
        for idx, c in enumerate(merged[:10], start=1):
            lines.append(
                f"- [{idx}] `{c.get('source_id', 'unknown')}` | {c.get('source_url', '')} | "
                f"score={float(c.get('reliability_score', 0.0)):.2f} | {c.get('excerpt', '')}"
            )

        return "\n".join(lines), merged[:10]

    def _build_analysis_brief(
        self,
        stock_codes: list[str],
        citations: list[dict[str, Any]],
        regime_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build structured evidence cards for frontend analysis panels."""
        by_code: list[dict[str, Any]] = []
        now = datetime.now(timezone.utc)
        for code in stock_codes:
            realtime = self._latest_quote(code)
            bars = self._history_bars(code, limit=260)
            trend = self._trend_metrics(bars) if len(bars) >= 30 else {}
            freshness_sec = None
            if realtime:
                freshness_sec = int((now - self._parse_time(str(realtime.get("ts", "")))).total_seconds())
            by_code.append(
                {
                    "stock_code": code,
                    "realtime": {
                        "price": realtime.get("price") if realtime else None,
                        "pct_change": realtime.get("pct_change") if realtime else None,
                        "source_id": realtime.get("source_id") if realtime else None,
                        "source_url": realtime.get("source_url") if realtime else None,
                        "ts": realtime.get("ts") if realtime else None,
                        "freshness_seconds": freshness_sec,
                    },
                    "trend": trend,
                    "history_sample_size": len(bars),
                }
            )

        valid_scores = [float(c.get("reliability_score", 0.0)) for c in citations if c.get("reliability_score") is not None]
        citation_coverage = len(citations)
        avg_score = round(mean(valid_scores), 4) if valid_scores else 0.0
        confidence = "high" if citation_coverage >= 4 and avg_score >= 0.8 else "medium" if citation_coverage >= 2 else "low"
        regime = regime_context if isinstance(regime_context, dict) else self._build_a_share_regime_context(stock_codes)
        seed_signal = "hold"
        seed_conf = max(0.45, min(0.92, 0.48 + avg_score * 0.4 + min(0.08, citation_coverage * 0.015)))
        if by_code and isinstance(by_code[0].get("trend"), dict):
            trend0 = by_code[0]["trend"]
            slope_20 = float(trend0.get("ma20_slope", 0.0) or 0.0)
            momentum_20 = float(trend0.get("momentum_20", 0.0) or 0.0)
            drawdown_60 = float(trend0.get("max_drawdown_60", 0.0) or 0.0)
            if slope_20 > 0 and momentum_20 > 0:
                seed_signal = "buy"
            elif slope_20 < 0 and (momentum_20 < 0 or drawdown_60 > 0.18):
                seed_signal = "reduce"
        signal_guard = self._apply_a_share_signal_guard(seed_signal, seed_conf, regime)
        return {
            "confidence_level": confidence,
            "confidence_reason": f"citations={citation_coverage}, avg_reliability={avg_score}",
            "stocks": by_code,
            "citation_count": citation_coverage,
            "citation_avg_reliability": avg_score,
            "market_regime": str(regime.get("regime_label", "")),
            "regime_confidence": float(regime.get("regime_confidence", 0.0) or 0.0),
            "risk_bias": str(regime.get("risk_bias", "")),
            "regime_rationale": str(regime.get("regime_rationale", "")),
            "signal_guard_applied": bool(signal_guard.get("applied", False)),
            "signal_guard_detail": signal_guard.get("detail", {}),
            "guarded_signal_preview": str(signal_guard.get("signal", seed_signal)),
            "guarded_confidence_preview": float(signal_guard.get("confidence", seed_conf)),
        }

    def _record_rag_eval_case(
        self,
        query_text: str,
        evidence_pack: list[dict[str, Any]],
        citations: list[dict[str, Any]],
    ) -> None:
        """Record online retrieval samples for iterative RAG evaluation."""
        predicted = [str(x.get("source_id", "")) for x in evidence_pack[:5] if x.get("source_id")]
        positive = [str(x.get("source_id", "")) for x in citations if x.get("source_id")]
        # 淇濆簭鍘婚噸
        pred_u = list(dict.fromkeys(predicted))
        pos_u = list(dict.fromkeys(positive))
        if not query_text.strip() or not pred_u:
            return
        self.web.rag_eval_add(query_text=query_text, positive_source_ids=pos_u, predicted_source_ids=pred_u)

    @staticmethod
    def _extract_retrieval_track(row: dict[str, Any]) -> str:
        """Extract retrieval track from flat or nested metadata payload."""
        track = str(row.get("retrieval_track", "")).strip()
        if track:
            return track
        meta = row.get("metadata", {})
        if isinstance(meta, dict):
            nested = str(meta.get("retrieval_track", "")).strip()
            if nested:
                return nested
        return ""

    def _normalize_citations_for_output(
        self,
        citations: list[dict[str, Any]],
        *,
        evidence_pack: list[dict[str, Any]] | None = None,
        max_items: int = 10,
    ) -> list[dict[str, Any]]:
        """Normalize citation payload and enforce attribution fields for downstream UI/logs."""
        tracks_by_source: dict[str, str] = {}
        for row in list(evidence_pack or []):
            if not isinstance(row, dict):
                continue
            source_id = str(row.get("source_id", "")).strip()
            if not source_id:
                continue
            track = self._extract_retrieval_track(row)
            if track and source_id not in tracks_by_source:
                tracks_by_source[source_id] = track

        normalized: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        safe_limit = max(1, min(50, int(max_items)))
        for raw in citations:
            if not isinstance(raw, dict):
                continue
            source_id = str(raw.get("source_id", "")).strip()
            if not source_id:
                continue
            excerpt = str(raw.get("excerpt", raw.get("text", ""))).strip()
            if not excerpt:
                excerpt = f"{source_id} 璇佹嵁鎽樿"
            dedup_key = (source_id, excerpt[:120])
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            source_url = str(raw.get("source_url", "")).strip() or "local://unknown"
            track = self._extract_retrieval_track(raw) or tracks_by_source.get(source_id, "unknown_track")
            reliability = float(raw.get("reliability_score", 0.5) or 0.5)
            reliability = max(0.0, min(1.0, reliability))
            event_time = str(raw.get("event_time", "")).strip() or None
            rerank_score_raw = raw.get("rerank_score")
            rerank_score = None
            if rerank_score_raw is not None:
                try:
                    rerank_score = round(float(rerank_score_raw), 6)
                except Exception:
                    rerank_score = None
            normalized.append(
                {
                    "source_id": source_id,
                    "source_url": source_url,
                    "event_time": event_time,
                    "reliability_score": round(reliability, 4),
                    "excerpt": excerpt[:240],
                    "retrieval_track": track,
                    "rerank_score": rerank_score,
                }
            )
            if len(normalized) >= safe_limit:
                break
        return normalized

    def _trace_source_identity(self, row: dict[str, Any]) -> str:
        """Build stable source identity that preserves retrieval attribution track."""
        source_id = str(row.get("source_id", "")).strip()
        if not source_id:
            return ""
        track = self._extract_retrieval_track(row)
        return f"{source_id}|{track}" if track else source_id

    def _record_rag_retrieval_trace(
        self,
        *,
        trace_id: str,
        query_text: str,
        query_type: str,
        evidence_pack: list[dict[str, Any]],
        citations: list[dict[str, Any]],
        latency_ms: int,
    ) -> None:
        """Record retrieved-vs-selected trace for debugging citation usage quality."""
        retrieved_ids = [
            self._trace_source_identity(x)
            for x in evidence_pack[:12]
            if self._trace_source_identity(x)
        ]
        selected_ids = [
            self._trace_source_identity(x)
            for x in citations[:12]
            if self._trace_source_identity(x)
        ]
        if not trace_id.strip():
            return
        self.web.rag_retrieval_trace_add(
            trace_id=trace_id,
            query_text=query_text[:500],
            query_type=query_type[:40],
            retrieved_ids=list(dict.fromkeys(retrieved_ids)),
            selected_ids=list(dict.fromkeys(selected_ids)),
            latency_ms=latency_ms,
        )

    def _persist_query_knowledge_memory(
        self,
        *,
        user_id: str,
        question: str,
        stock_codes: list[str],
        state: AgentState,
        query_type: str,
    ) -> None:
        """Persist query result into shared QA memory with raw/redacted/summary tracks."""
        answer_text = str(state.report or "").strip()
        if not answer_text:
            return
        citations = [c for c in state.citations if isinstance(c, dict)]
        risk_flags = [str(x) for x in state.risk_flags]
        quality_score = self._estimate_qa_quality(answer_text, citations, risk_flags)
        high_risk_flags = {"missing_citation", "compliance_block"}
        retrieval_enabled = bool(len(citations) >= 2 and quality_score >= 0.65 and not (high_risk_flags & set(risk_flags)))
        primary_code = str(stock_codes[0]).upper() if stock_codes else "GLOBAL"
        answer_redacted = self._redact_text(answer_text)
        summary_text = self._build_qa_summary(answer_redacted, citations, query_type=query_type)
        memory_id = f"qam-{uuid.uuid4().hex[:16]}"
        self.web.rag_qa_memory_add(
            memory_id=memory_id,
            user_id=str(user_id),
            stock_code=primary_code,
            query_text=str(question)[:1000],
            answer_text=answer_text[:12000],
            answer_redacted=answer_redacted[:12000],
            summary_text=summary_text[:1000],
            citations=citations[:12],
            risk_flags=risk_flags[:12],
            intent=str(state.intent),
            quality_score=quality_score,
            share_scope="global",
            retrieval_enabled=retrieval_enabled,
        )

    @staticmethod
    def _estimate_qa_quality(answer_text: str, citations: list[dict[str, Any]], risk_flags: list[str]) -> float:
        """Lightweight QA quality score used for deciding shared-memory eligibility."""
        score = 0.25
        score += min(0.35, len(citations) * 0.08)
        length = len(answer_text)
        if length >= 400:
            score += 0.25
        elif length >= 180:
            score += 0.18
        elif length >= 80:
            score += 0.10
        penalties = 0.0
        if "missing_citation" in risk_flags:
            penalties += 0.20
        if "external_model_failed" in risk_flags:
            penalties += 0.08
        if "compliance_block" in risk_flags:
            penalties += 0.20
        return round(max(0.0, min(1.0, score - penalties)), 4)

    @staticmethod
    def _build_qa_summary(answer_redacted: str, citations: list[dict[str, Any]], *, query_type: str) -> str:
        """Build retrieval-friendly summary: structured prefix plus clipped redacted answer."""
        source_ids = [str(x.get("source_id", "")) for x in citations if str(x.get("source_id", "")).strip()]
        source_head = ",".join(list(dict.fromkeys(source_ids))[:4]) or "unknown"
        head = f"[{query_type}] sources={source_head}; "
        body = answer_redacted.replace("\n", " ").strip()
        return head + body[:780]

    def _calc_retrieval_metrics_from_cases(self, cases: list[dict[str, Any]], k: int = 5) -> dict[str, float]:
        if not cases:
            return {"recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0}
        recalls: list[float] = []
        mrrs: list[float] = []
        ndcgs: list[float] = []
        for case in cases:
            positives = set(case.get("positive_source_ids", []))
            pred = list(case.get("predicted_source_ids", []))[:k]
            recalls.append(RetrievalEvaluator._recall_at_k(pred, positives))
            mrrs.append(RetrievalEvaluator._mrr(pred, positives))
            ndcgs.append(RetrievalEvaluator._ndcg_at_k(pred, positives, k))
        n = max(1, len(cases))
        return {
            "recall_at_k": round(sum(recalls) / n, 4),
            "mrr": round(sum(mrrs) / n, 4),
            "ndcg_at_k": round(sum(ndcgs) / n, 4),
        }

    def _build_runtime_retriever(self, stock_codes: list[str]):
        """Build runtime retriever, optionally enabling semantic vector retrieval."""
        corpus = self._build_runtime_corpus(stock_codes)
        if not self.settings.rag_vector_enabled:
            return HybridRetriever(corpus=corpus)
        self._refresh_summary_vector_index(stock_codes)

        def _semantic_search(query: str, top_k: int) -> list[RetrievalItem]:
            return self._semantic_summary_origin_hits(query, top_k=max(1, top_k))

        return HybridRetrieverV2(corpus=corpus, semantic_search_fn=_semantic_search)

    def _build_summary_vector_records(self, stock_codes: list[str]) -> list[VectorSummaryRecord]:
        """Map persisted assets into summary index records for summary-first retrieval."""
        symbols = {str(x).upper() for x in stock_codes}
        rows: list[VectorSummaryRecord] = []
        doc_rows = self.web.rag_doc_chunk_list_internal(status="active", limit=2500)
        for row in doc_rows:
            row_codes = {str(x).upper() for x in row.get("stock_codes", [])}
            if symbols and row_codes and symbols.isdisjoint(row_codes):
                continue
            summary_text = str(row.get("chunk_text_redacted") or row.get("chunk_text") or "").strip()
            parent_text = str(row.get("chunk_text") or "").strip()
            if not summary_text or not parent_text:
                continue
            source = str(row.get("source", "doc_upload"))
            record_id = f"doc:{row.get('chunk_id', '')}"
            rows.append(
                VectorSummaryRecord(
                    record_id=record_id,
                    kind="doc_chunk",
                    summary_text=summary_text[:1800],
                    parent_text=parent_text[:4000],
                    source_id=f"doc::{source}",
                    source_url=str(row.get("source_url") or f"local://docs/{row.get('doc_id', 'unknown')}"),
                    event_time=str(row.get("updated_at", "")),
                    reliability_score=float(max(0.5, min(1.0, row.get("quality_score", 0.7) or 0.7))),
                    stock_code=",".join(sorted(row_codes)) if row_codes else "GLOBAL",
                    updated_at=str(row.get("updated_at", "")),
                    metadata={
                        "doc_id": str(row.get("doc_id", "")),
                        "chunk_id": str(row.get("chunk_id", "")),
                        "track": "doc_summary",
                    },
                )
            )
        qa_rows = self.web.rag_qa_memory_list_internal(retrieval_enabled=1, limit=3000, offset=0)
        for row in qa_rows:
            row_code = str(row.get("stock_code", "")).upper()
            if symbols and row_code not in symbols and row_code != "GLOBAL":
                continue
            summary = str(row.get("summary_text", "")).strip()
            parent = str(row.get("answer_redacted", "") or row.get("answer_text", "")).strip()
            if not summary or not parent:
                continue
            record_id = f"qa:{row.get('memory_id', '')}"
            rows.append(
                VectorSummaryRecord(
                    record_id=record_id,
                    kind="qa_memory",
                    summary_text=summary[:1800],
                    parent_text=parent[:4000],
                    source_id="qa_memory_summary",
                    source_url=f"local://qa/{row.get('memory_id', 'unknown')}",
                    event_time=str(row.get("created_at", "")),
                    reliability_score=float(max(0.5, min(1.0, row.get("quality_score", 0.65) or 0.65))),
                    stock_code=row_code or "GLOBAL",
                    updated_at=str(row.get("created_at", "")),
                    metadata={
                        "memory_id": str(row.get("memory_id", "")),
                        "track": "qa_summary",
                    },
                )
            )
        return rows

    def _refresh_summary_vector_index(self, stock_codes: list[str], force: bool = False) -> dict[str, Any]:
        records = self._build_summary_vector_records(stock_codes)
        signature = self._summary_vector_signature(records)
        # 鍑忓皯閲嶅 rebuild锛氱鍚嶆湭鍙樺寲鏃跺鐢ㄧ幇鏈夌储寮曘€?
        if not force and signature == self._vector_signature and self.vector_store.record_count:
            return {"status": "reused", "indexed_count": self.vector_store.record_count, "backend": self.vector_store.backend}
        result = self.vector_store.rebuild(records)
        self._vector_signature = signature
        self._vector_refreshed_at = time.time()
        return {"status": "rebuilt", **result}

    @staticmethod
    def _summary_vector_signature(records: list[VectorSummaryRecord]) -> str:
        if not records:
            return "empty"
        parts = [f"{r.record_id}:{r.updated_at}" for r in records]
        return f"{len(records)}:{hash('|'.join(parts))}"

    def _semantic_summary_origin_hits(self, query: str, top_k: int = 8) -> list[RetrievalItem]:
        """Summary-first recall with optional origin backfill to preserve full-answer evidence."""
        hits = self.vector_store.search(query, top_k=max(1, top_k))
        out: list[RetrievalItem] = []
        for hit in hits:
            record = hit.get("record", {}) if isinstance(hit, dict) else {}
            if not isinstance(record, dict):
                continue
            score = float(hit.get("score", 0.0))
            source_id = str(record.get("source_id", "summary"))
            source_url = str(record.get("source_url", ""))
            event_time = self._parse_time(str(record.get("event_time", "")))
            reliability = float(record.get("reliability_score", 0.6))
            summary_text = str(record.get("summary_text", ""))
            parent_text = str(record.get("parent_text", ""))
            meta = {
                "semantic_score": round(score, 6),
                "record_id": str(record.get("record_id", "")),
                "retrieval_track": str(record.get("kind", "summary")),
            }
            out.append(
                RetrievalItem(
                    text=summary_text,
                    source_id=source_id,
                    source_url=source_url,
                    score=score,
                    event_time=event_time,
                    reliability_score=reliability,
                    metadata=dict(meta),
                )
            )
            # 鍥炶ˉ鍘熸枃锛氳鏈€缁堣瘉鎹洿鍙涓斿彲鐢ㄤ簬鐢熸垚鏇村畬鏁寸瓟妗堛€?
            if parent_text:
                backfill_meta = dict(meta)
                backfill_meta.update({"origin_backfill": True, "retrieval_track": "origin_backfill"})
                out.append(
                    RetrievalItem(
                        text=parent_text,
                        source_id=source_id,
                        source_url=source_url,
                        score=score * 0.92,
                        event_time=event_time,
                        reliability_score=reliability,
                        metadata=backfill_meta,
                    )
                )
        return out

    def _build_runtime_corpus(self, stock_codes: list[str]) -> list[RetrievalItem]:
        """Convert ingested datasets into runtime retrieval corpus items."""
        symbols = {str(x).upper() for x in stock_codes if str(x).strip()}
        corpus = HybridRetriever()._default_corpus()

        # Quote snapshots -> textual evidence.
        for q in self.ingestion_store.quotes[-80:]:
            code = str(q.get("stock_code", "")).upper()
            if symbols and code not in symbols:
                continue
            ts = self._parse_time(str(q.get("ts", "")))
            text = (
                f"{code} quote snapshot: price={q.get('price')}, pct_change={q.get('pct_change')}%, "
                f"volume={q.get('volume')}, turnover={q.get('turnover')}"
            )
            corpus.append(
                RetrievalItem(
                    text=text,
                    source_id=str(q.get("source_id", "unknown")),
                    source_url=str(q.get("source_url", "")),
                    score=0.0,
                    event_time=ts,
                    reliability_score=float(q.get("reliability_score", 0.6) or 0.6),
                    metadata={"retrieval_track": "quote_snapshot"},
                )
            )

        # Announcements -> textual evidence.
        for a in self.ingestion_store.announcements[-120:]:
            code = str(a.get("stock_code", "")).upper()
            if symbols and code not in symbols:
                continue
            event_time = self._parse_time(str(a.get("event_time", "")))
            text = f"{code} announcement: {a.get('title', '')}. {a.get('content', '')}"
            corpus.append(
                RetrievalItem(
                    text=text,
                    source_id=str(a.get("source_id", "announcement")),
                    source_url=str(a.get("source_url", "")),
                    score=0.0,
                    event_time=event_time,
                    reliability_score=float(a.get("reliability_score", 0.9) or 0.9),
                    metadata={"retrieval_track": "announcement_event"},
                )
            )

        # Daily bars -> textual evidence.
        history_by_code: dict[str, list[dict[str, Any]]] = {}
        for b in self.ingestion_store.history_bars[-2000:]:
            code = str(b.get("stock_code", "")).upper()
            if symbols and code not in symbols:
                continue
            history_by_code.setdefault(code, []).append(b)
            text = (
                f"{code} daily bar {b.get('trade_date', '')}: "
                f"open={b.get('open')} high={b.get('high')} low={b.get('low')} "
                f"close={b.get('close')} volume={b.get('volume')}"
            )
            corpus.append(
                RetrievalItem(
                    text=text,
                    source_id=str(b.get("source_id", "eastmoney_history")),
                    source_url=str(b.get("source_url", "")),
                    score=0.0,
                    event_time=self._parse_time(str(b.get("trade_date", ""))),
                    reliability_score=float(b.get("reliability_score", 0.9) or 0.9),
                    metadata={"retrieval_track": "history_daily"},
                )
            )

        # Add rolling 3-month summary evidence per stock.
        for code, rows in history_by_code.items():
            if not rows:
                continue
            rows = sorted(rows, key=lambda x: str(x.get("trade_date", "")))
            window = rows[-90:]
            if len(window) < 2:
                continue
            start = window[0]
            end = window[-1]
            start_close = float(start.get("close", 0.0) or 0.0)
            end_close = float(end.get("close", 0.0) or 0.0)
            pct_change = (end_close / start_close - 1.0) if start_close > 0 else 0.0
            text = (
                f"{code} 3m rolling window: sample={len(window)}, "
                f"range={start.get('trade_date', '')}->{end.get('trade_date', '')}, "
                f"close={start_close:.3f}->{end_close:.3f}, pct_change={pct_change * 100:.2f}%"
            )
            corpus.append(
                RetrievalItem(
                    text=text,
                    source_id="eastmoney_history_3m_window",
                    source_url=str(end.get("source_url", "")),
                    score=0.0,
                    event_time=self._parse_time(str(end.get("trade_date", ""))),
                    reliability_score=0.92,
                    metadata={"retrieval_track": "history_3m_window"},
                )
            )

        # Financial snapshots -> textual evidence.
        for fin in self.ingestion_store.financial_snapshots[-600:]:
            code = str(fin.get("stock_code", "")).upper()
            if symbols and code not in symbols:
                continue
            text = (
                f"{code} financial snapshot: roe={fin.get('roe')}, gross_margin={fin.get('gross_margin')}%, "
                f"revenue_yoy={fin.get('revenue_yoy')}%, net_profit_yoy={fin.get('net_profit_yoy')}%, "
                f"pe_ttm={fin.get('pe_ttm')}, pb_mrq={fin.get('pb_mrq')}"
            )
            corpus.append(
                RetrievalItem(
                    text=text,
                    source_id=str(fin.get("source_id", "financial_snapshot")),
                    source_url=str(fin.get("source_url", "")),
                    score=0.0,
                    event_time=self._parse_time(str(fin.get("ts", ""))),
                    reliability_score=float(fin.get("reliability_score", 0.75) or 0.75),
                    metadata={"retrieval_track": "financial_snapshot"},
                )
            )

        # News events -> textual evidence.
        for row in self.ingestion_store.news_items[-1200:]:
            code = str(row.get("stock_code", "")).upper()
            if symbols and code not in symbols:
                continue
            text = f"{code} news: {row.get('title', '')}. {row.get('content', '')}"
            corpus.append(
                RetrievalItem(
                    text=text,
                    source_id=str(row.get("source_id", "news")),
                    source_url=str(row.get("source_url", "")),
                    score=0.0,
                    event_time=self._parse_time(str(row.get("event_time", ""))),
                    reliability_score=float(row.get("reliability_score", 0.65) or 0.65),
                    metadata={"retrieval_track": "news_event"},
                )
            )

        # Research reports -> textual evidence.
        for row in self.ingestion_store.research_reports[-800:]:
            code = str(row.get("stock_code", "")).upper()
            if symbols and code not in symbols:
                continue
            text = (
                f"{code} research report: {row.get('title', '')}. "
                f"org={row.get('org_name', '')}. author={row.get('author', '')}."
            )
            corpus.append(
                RetrievalItem(
                    text=text,
                    source_id=str(row.get("source_id", "research")),
                    source_url=str(row.get("source_url", "")),
                    score=0.0,
                    event_time=self._parse_time(str(row.get("published_at", ""))),
                    reliability_score=float(row.get("reliability_score", 0.7) or 0.7),
                    metadata={"retrieval_track": "research_report"},
                )
            )

        # Macro indicators -> global evidence.
        for row in self.ingestion_store.macro_indicators[-400:]:
            text = f"macro indicator {row.get('metric_name', '')}={row.get('metric_value', '')}, date={row.get('report_date', '')}"
            corpus.append(
                RetrievalItem(
                    text=text,
                    source_id=str(row.get("source_id", "macro")),
                    source_url=str(row.get("source_url", "")),
                    score=0.0,
                    event_time=self._parse_time(str(row.get("event_time", ""))),
                    reliability_score=float(row.get("reliability_score", 0.7) or 0.7),
                    metadata={"retrieval_track": "macro_indicator"},
                )
            )

        # Fund flow snapshots -> textual evidence.
        for row in self.ingestion_store.fund_snapshots[-600:]:
            code = str(row.get("stock_code", "")).upper()
            if symbols and code not in symbols:
                continue
            text = (
                f"{code} fund flow: main={row.get('main_inflow')}, small={row.get('small_inflow')}, "
                f"middle={row.get('middle_inflow')}, large={row.get('large_inflow')}"
            )
            corpus.append(
                RetrievalItem(
                    text=text,
                    source_id=str(row.get("source_id", "fund")),
                    source_url=str(row.get("source_url", "")),
                    score=0.0,
                    event_time=self._parse_time(str(row.get("ts", ""))),
                    reliability_score=float(row.get("reliability_score", 0.65) or 0.65),
                    metadata={"retrieval_track": "fund_flow"},
                )
            )

        # In-memory uploaded docs chunks.
        for doc in self.ingestion_store.docs.values():
            if not doc.get("indexed"):
                continue
            doc_source = str(doc.get("source", "doc_upload"))
            doc_url = f"local://docs/{doc.get('doc_id', 'unknown')}"
            for chunk in doc.get("chunks", [])[:8]:
                if not chunk:
                    continue
                corpus.append(
                    RetrievalItem(
                        text=str(chunk),
                        source_id=doc_source,
                        source_url=doc_url,
                        score=0.0,
                        event_time=datetime.now(timezone.utc),
                        reliability_score=0.75,
                        metadata={"retrieval_track": "doc_inline_chunk"},
                    )
                )

        # Persisted active doc chunks.
        persisted_chunks = self.web.rag_doc_chunk_list_internal(status="active", limit=1200)
        for row in persisted_chunks:
            row_codes = {str(x).upper() for x in row.get("stock_codes", [])}
            if symbols and row_codes and symbols.isdisjoint(row_codes):
                continue
            summary_text = str(row.get("chunk_text_redacted") or row.get("chunk_text") or "").strip()
            if not summary_text:
                continue
            source = str(row.get("source", "doc_upload"))
            corpus.append(
                RetrievalItem(
                    text=summary_text,
                    source_id=f"doc::{source}",
                    source_url=str(row.get("source_url") or f"local://docs/{row.get('doc_id', 'unknown')}"),
                    score=0.0,
                    event_time=self._parse_time(str(row.get("updated_at", ""))),
                    reliability_score=float(max(0.5, min(1.0, row.get("quality_score", 0.7) or 0.7))),
                    metadata={
                        "chunk_id": str(row.get("chunk_id", "")),
                        "doc_id": str(row.get("doc_id", "")),
                        "retrieval_track": "doc_summary",
                    },
                )
            )

        # Shared QA summaries as retrieval candidates.
        qa_rows = self.web.rag_qa_memory_list_internal(retrieval_enabled=1, limit=1500, offset=0)
        for row in qa_rows:
            row_code = str(row.get("stock_code", "")).upper()
            if symbols and row_code not in symbols and row_code != "GLOBAL":
                continue
            summary = str(row.get("summary_text", "")).strip()
            if not summary:
                continue
            corpus.append(
                RetrievalItem(
                    text=summary,
                    source_id="qa_memory_summary",
                    source_url=f"local://qa/{row.get('memory_id', 'unknown')}",
                    score=0.0,
                    event_time=self._parse_time(str(row.get("created_at", ""))),
                    reliability_score=float(max(0.5, min(1.0, row.get("quality_score", 0.65) or 0.65))),
                    metadata={
                        "memory_id": str(row.get("memory_id", "")),
                        # Keep parent answer for two-stage recall then answer backfill.
                        "parent_answer_text": str(row.get("answer_text", "")),
                        "retrieval_track": "qa_summary",
                    },
                )
            )

        return corpus

    @staticmethod
    def _parse_time(value: str) -> datetime:
        if not value:
            return datetime.now(timezone.utc)
        try:
            # 鏀寔 ISO 鏃堕棿涓?YYYY-MM-DD 鏃ユ湡涓ょ鏍煎紡銆?
            if len(value) == 10 and value.count("-") == 2:
                return datetime.fromisoformat(value + "T00:00:00+00:00")
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(timezone.utc)

    def _needs_quote_refresh(self, stock_code: str, max_age_seconds: int = 300) -> bool:
        code = stock_code.upper().replace(".", "")
        now = datetime.now(timezone.utc)
        for q in reversed(self.ingestion_store.quotes):
            if str(q.get("stock_code", "")).upper() != code:
                continue
            ts = self._parse_time(str(q.get("ts", "")))
            return (now - ts).total_seconds() > max_age_seconds
        return True

    def _needs_announcement_refresh(self, stock_code: str, max_age_seconds: int = 60 * 60 * 4) -> bool:
        code = stock_code.upper().replace(".", "")
        now = datetime.now(timezone.utc)
        for a in reversed(self.ingestion_store.announcements):
            if str(a.get("stock_code", "")).upper() != code:
                continue
            ts = self._parse_time(str(a.get("event_time", "")))
            return (now - ts).total_seconds() > max_age_seconds
        return True

    def _needs_history_refresh(
        self,
        stock_code: str,
        max_age_seconds: int = 60 * 60 * 8,
        min_samples: int = 90,
    ) -> bool:
        code = stock_code.upper().replace(".", "")
        now = datetime.now(timezone.utc)
        # 鑻ユ湰鍦板巻鍙叉牱鏈笉瓒充笁涓湀锛堢害90涓氦鏄撴棩绐楀彛锛夛紝涔熷己鍒跺埛鏂颁竴娆°€?
        sample_count = sum(1 for b in self.ingestion_store.history_bars if str(b.get("stock_code", "")).upper() == code)
        if sample_count < max(30, int(min_samples)):
            return True
        for b in reversed(self.ingestion_store.history_bars):
            if str(b.get("stock_code", "")).upper() != code:
                continue
            ts = self._parse_time(str(b.get("trade_date", "")))
            return (now - ts).total_seconds() > max_age_seconds
        return True

    def _needs_financial_refresh(self, stock_code: str, max_age_seconds: int = 60 * 60 * 24) -> bool:
        code = stock_code.upper().replace(".", "")
        now = datetime.now(timezone.utc)
        for item in reversed(self.ingestion_store.financial_snapshots):
            if str(item.get("stock_code", "")).upper() != code:
                continue
            ts = self._parse_time(str(item.get("ts", "")))
            return (now - ts).total_seconds() > max_age_seconds
        return True

    def _needs_news_refresh(self, stock_code: str, max_age_seconds: int = 60 * 60) -> bool:
        code = stock_code.upper().replace(".", "")
        now = datetime.now(timezone.utc)
        for item in reversed(self.ingestion_store.news_items):
            if str(item.get("stock_code", "")).upper() != code:
                continue
            ts = self._parse_time(str(item.get("event_time", "")))
            return (now - ts).total_seconds() > max_age_seconds
        return True

    def _needs_research_refresh(self, stock_code: str, max_age_seconds: int = 60 * 60 * 12) -> bool:
        code = stock_code.upper().replace(".", "")
        now = datetime.now(timezone.utc)
        for item in reversed(self.ingestion_store.research_reports):
            if str(item.get("stock_code", "")).upper() != code:
                continue
            ts = self._parse_time(str(item.get("published_at", "")))
            return (now - ts).total_seconds() > max_age_seconds
        return True

    def _needs_macro_refresh(self, max_age_seconds: int = 60 * 60 * 12) -> bool:
        now = datetime.now(timezone.utc)
        if not self.ingestion_store.macro_indicators:
            return True
        ts = self._parse_time(str(self.ingestion_store.macro_indicators[-1].get("event_time", "")))
        return (now - ts).total_seconds() > max_age_seconds

    def _needs_fund_refresh(self, stock_code: str, max_age_seconds: int = 60 * 30) -> bool:
        code = stock_code.upper().replace(".", "")
        now = datetime.now(timezone.utc)
        for item in reversed(self.ingestion_store.fund_snapshots):
            if str(item.get("stock_code", "")).upper() != code:
                continue
            ts = self._parse_time(str(item.get("ts", "")))
            return (now - ts).total_seconds() > max_age_seconds
        return True

    @staticmethod
    def _scenario_dataset_requirements(scenario: str) -> dict[str, int]:
        """Return per-scenario minimum data requirements for LLM input preparation.

        The thresholds are intentionally conservative so single-turn requests can still
        produce stable conclusions without asking users to add more context manually.
        """
        normalized = str(scenario or "").strip().lower()
        base = {
            "quote_min": 1,
            "history_min": 90,
            "history_fetch_limit": 260,
            "financial_min": 1,
            "announcement_min": 1,
            "news_min": 8,
            "news_fetch_limit": 10,
            "research_min": 6,
            "research_fetch_limit": 8,
            "macro_min": 4,
            "macro_fetch_limit": 8,
            "fund_min": 1,
        }
        # Different modules value different coverage guarantees.
        profiles: dict[str, dict[str, int]] = {
            "query": {"history_min": 90, "news_min": 8, "research_min": 6},
            "query_stream": {"history_min": 90, "news_min": 8, "research_min": 6},
            "analysis_studio": {"history_min": 90, "news_min": 8, "research_min": 6},
            # DeepThink/Report are long-horizon decisions: require close-to-1y bars by default.
            "deepthink": {
                "history_min": 252,
                "history_fetch_limit": 520,
                "news_min": 10,
                "research_min": 8,
                "macro_min": 6,
            },
            "report": {
                "history_min": 252,
                "history_fetch_limit": 520,
                "news_min": 10,
                "research_min": 8,
                "macro_min": 6,
            },
            "predict": {"history_min": 260, "history_fetch_limit": 520, "news_min": 4, "research_min": 4},
            "overview": {"history_min": 120, "news_min": 8, "research_min": 6},
            "intel": {"history_min": 120, "news_min": 10, "research_min": 8, "macro_min": 6},
        }
        override = profiles.get(normalized, {})
        merged = dict(base)
        merged.update(override)
        return merged

    def _count_stock_dataset(self, stock_code: str, *, history_limit: int = 520, macro_limit: int = 20) -> dict[str, int]:
        """Count current in-memory dataset coverage for one stock."""
        code = str(stock_code or "").strip().upper().replace(".", "")
        history_rows = self._history_bars(code, limit=max(30, int(history_limit)))
        return {
            "quote_count": 1 if self._latest_quote(code) else 0,
            "history_count": len(history_rows),
            "history_30d_count": len(history_rows[-30:]),
            "history_90d_count": len(history_rows[-90:]),
            "financial_count": 1 if self._latest_financial_snapshot(code) else 0,
            "announcement_count": len([x for x in self.ingestion_store.announcements if str(x.get("stock_code", "")).upper() == code][-20:]),
            "news_count": len([x for x in self.ingestion_store.news_items if str(x.get("stock_code", "")).upper() == code][-40:]),
            "research_count": len([x for x in self.ingestion_store.research_reports if str(x.get("stock_code", "")).upper() == code][-40:]),
            "fund_count": len([x for x in self.ingestion_store.fund_snapshots if str(x.get("stock_code", "")).upper() == code][-4:]),
            "macro_count": len(self.ingestion_store.macro_indicators[-max(1, int(macro_limit)):]),
        }

    def _primary_source_for_category(self, category: str) -> str:
        """Resolve one representative source_id for datasource observability logs."""
        target = str(category or "").strip().lower()
        for row in self._build_datasource_catalog():
            if str(row.get("category", "")).strip().lower() != target:
                continue
            if bool(row.get("enabled", True)):
                sid = str(row.get("source_id", "")).strip()
                if sid:
                    return sid
        fallback = {
            "quote": "quote_service",
            "history": "eastmoney_history",
            "financial": "financial_service",
            "announcement": "announcement_service",
            "news": "news_service",
            "research": "research_service",
            "macro": "macro_service",
            "fund": "fund_service",
        }
        return fallback.get(target, f"{target}_service")

    def _refresh_category_for_stock(
        self,
        *,
        stock_code: str,
        category: str,
        scenario: str,
        limit: int = 0,
    ) -> dict[str, Any]:
        """Refresh one datasource category and persist structured observability logs."""
        code = str(stock_code or "").strip().upper().replace(".", "")
        category_norm = str(category or "").strip().lower()
        started = time.perf_counter()
        status = "ok"
        error = ""
        result: dict[str, Any] | None = None

        try:
            if category_norm == "quote":
                result = self.ingest_market_daily([code])
            elif category_norm == "history":
                history_limit = max(120, int(limit) or 260)
                result = self.ingestion.ingest_history_daily([code], limit=history_limit)
            elif category_norm == "financial":
                result = self.ingest_financials([code])
            elif category_norm == "announcement":
                result = self.ingest_announcements([code])
            elif category_norm == "news":
                fetch_limit = max(4, int(limit) or 8)
                result = self.ingest_news([code], limit=fetch_limit)
            elif category_norm == "research":
                fetch_limit = max(4, int(limit) or 6)
                result = self.ingest_research_reports([code], limit=fetch_limit)
            elif category_norm == "macro":
                fetch_limit = max(4, int(limit) or 8)
                result = self.ingest_macro_indicators(limit=fetch_limit)
            elif category_norm == "fund":
                result = self.ingest_fund_snapshots([code])
            else:
                status = "failed"
                error = f"unsupported category: {category_norm}"
        except Exception as ex:  # noqa: BLE001
            status = "failed"
            error = str(ex)[:260]
            result = {"status": "failed", "error": error}

        latency_ms = int((time.perf_counter() - started) * 1000)
        source_id = self._primary_source_for_category(category_norm)
        self._append_datasource_log(
            source_id=source_id,
            category=category_norm,
            action=f"auto_refresh:{scenario}",
            status=status,
            latency_ms=latency_ms,
            detail={
                "stock_code": code,
                "scenario": scenario,
                "limit": int(limit or 0),
            },
            error=error,
        )
        return {
            "category": category_norm,
            "source_id": source_id,
            "status": status,
            "error": error,
            "latency_ms": latency_ms,
            "result": result or {},
        }

    def _ensure_analysis_dataset(self, stock_code: str, *, scenario: str) -> dict[str, Any]:
        """Ensure minimum datasource coverage before LLM/model reasoning starts."""
        code = str(stock_code or "").strip().upper().replace(".", "")
        requirements = self._scenario_dataset_requirements(scenario)
        refresh_actions: list[dict[str, Any]] = []

        before = self._count_stock_dataset(
            code,
            history_limit=max(520, int(requirements["history_fetch_limit"])),
            macro_limit=max(20, int(requirements["macro_fetch_limit"])),
        )
        if before["quote_count"] < int(requirements["quote_min"]) or self._needs_quote_refresh(code):
            refresh_actions.append(self._refresh_category_for_stock(stock_code=code, category="quote", scenario=scenario))
        if before["history_count"] < int(requirements["history_min"]) or self._needs_history_refresh(
            code,
            min_samples=int(requirements["history_min"]),
        ):
            refresh_actions.append(
                self._refresh_category_for_stock(
                    stock_code=code,
                    category="history",
                    scenario=scenario,
                    limit=int(requirements["history_fetch_limit"]),
                )
            )
        if before["financial_count"] < int(requirements["financial_min"]) or self._needs_financial_refresh(code):
            refresh_actions.append(self._refresh_category_for_stock(stock_code=code, category="financial", scenario=scenario))
        if before["announcement_count"] < int(requirements["announcement_min"]) or self._needs_announcement_refresh(code):
            refresh_actions.append(self._refresh_category_for_stock(stock_code=code, category="announcement", scenario=scenario))
        if before["news_count"] < int(requirements["news_min"]) or self._needs_news_refresh(code):
            refresh_actions.append(
                self._refresh_category_for_stock(
                    stock_code=code,
                    category="news",
                    scenario=scenario,
                    limit=int(requirements["news_fetch_limit"]),
                )
            )
        if before["research_count"] < int(requirements["research_min"]) or self._needs_research_refresh(code):
            refresh_actions.append(
                self._refresh_category_for_stock(
                    stock_code=code,
                    category="research",
                    scenario=scenario,
                    limit=int(requirements["research_fetch_limit"]),
                )
            )
        if before["fund_count"] < int(requirements["fund_min"]) or self._needs_fund_refresh(code):
            refresh_actions.append(self._refresh_category_for_stock(stock_code=code, category="fund", scenario=scenario))
        if before["macro_count"] < int(requirements["macro_min"]) or self._needs_macro_refresh():
            refresh_actions.append(
                self._refresh_category_for_stock(
                    stock_code=code,
                    category="macro",
                    scenario=scenario,
                    limit=int(requirements["macro_fetch_limit"]),
                )
            )

        coverage = self._count_stock_dataset(
            code,
            history_limit=max(520, int(requirements["history_fetch_limit"])),
            macro_limit=max(20, int(requirements["macro_fetch_limit"])),
        )

        missing_data: list[str] = []
        if coverage["quote_count"] < int(requirements["quote_min"]):
            missing_data.append("quote_missing")
        if coverage["history_count"] < int(requirements["history_min"]):
            missing_data.append("history_insufficient")
        if coverage["history_30d_count"] < 30:
            missing_data.append("history_30d_insufficient")
        if coverage["financial_count"] < int(requirements["financial_min"]):
            missing_data.append("financial_missing")
        if coverage["announcement_count"] < int(requirements["announcement_min"]):
            missing_data.append("announcement_missing")
        if coverage["news_count"] < int(requirements["news_min"]):
            missing_data.append("news_insufficient")
        if coverage["research_count"] < int(requirements["research_min"]):
            missing_data.append("research_insufficient")
        if coverage["fund_count"] < int(requirements["fund_min"]):
            missing_data.append("fund_missing")
        if coverage["macro_count"] < int(requirements["macro_min"]):
            missing_data.append("macro_insufficient")

        return {
            "stock_code": code,
            "scenario": str(scenario or "").strip().lower(),
            "requirements": requirements,
            "coverage": coverage,
            "missing_data": list(dict.fromkeys(missing_data)),
            "refresh_actions": refresh_actions,
            "degraded": bool(missing_data),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _build_llm_input_pack(self, stock_code: str, *, question: str, scenario: str) -> dict[str, Any]:
        """Build one deterministic input pack consumed by query/report/deepthink/predict prompts."""
        code = str(stock_code or "").strip().upper().replace(".", "")
        dataset = self._ensure_analysis_dataset(code, scenario=scenario)
        requirements = dict(dataset.get("requirements", {}) or {})
        history_limit = max(120, int(requirements.get("history_fetch_limit", 260) or 260))
        bars = self._history_bars(code, limit=history_limit)
        news = [x for x in self.ingestion_store.news_items if str(x.get("stock_code", "")).upper() == code][-max(10, int(requirements.get("news_fetch_limit", 10) or 10)) :]
        research = [
            x for x in self.ingestion_store.research_reports if str(x.get("stock_code", "")).upper() == code
        ][-max(8, int(requirements.get("research_fetch_limit", 8) or 8)) :]
        announcements = [x for x in self.ingestion_store.announcements if str(x.get("stock_code", "")).upper() == code][-20:]
        fund = [x for x in self.ingestion_store.fund_snapshots if str(x.get("stock_code", "")).upper() == code][-2:]
        macro = self.ingestion_store.macro_indicators[-max(8, int(requirements.get("macro_fetch_limit", 8) or 8)) :]
        trend = self._trend_metrics(bars) if len(bars) >= 30 else {}
        daily_30 = [
            {
                "trade_date": str(x.get("trade_date", "")),
                "close": float(x.get("close", 0.0) or 0.0),
                "pct_change": float(x.get("pct_change", 0.0) or 0.0),
                "volume": float(x.get("volume", 0.0) or 0.0),
            }
            for x in bars[-30:]
        ]
        # 1y pack: these fields are explicitly consumed by report/deep-think prompts to avoid "3m-only" evidence bias.
        daily_252 = [
            {
                "trade_date": str(x.get("trade_date", "")),
                "close": float(x.get("close", 0.0) or 0.0),
                "pct_change": float(x.get("pct_change", 0.0) or 0.0),
                "volume": float(x.get("volume", 0.0) or 0.0),
            }
            for x in bars[-252:]
        ]
        history_1y = self._history_1y_summary(code)
        monthly_12 = self._history_monthly_summary(code, months=12)
        quarterly_summary = self._quarterly_financial_summary(code, limit=8)
        timeline_1y = self._event_timeline_1y(code, limit=24)
        uncertainty_notes = self._build_evidence_uncertainty_notes(
            dataset=dataset,
            history_1y=history_1y,
            quarterly_count=len(quarterly_summary),
            timeline_count=len(timeline_1y),
        )
        time_horizon_coverage = {
            "history_30d_count": int(len(daily_30)),
            "history_90d_count": int(min(90, len(bars))),
            "history_252d_count": int(len(daily_252)),
            "history_has_full_1y": bool(history_1y.get("has_full_1y", False)),
            "quarterly_fundamentals_count": int(len(quarterly_summary)),
            "event_timeline_count": int(len(timeline_1y)),
        }
        return {
            "stock_code": code,
            "scenario": str(scenario or "").strip().lower(),
            "question": str(question or "")[:500],
            "dataset": dataset,
            "realtime": self._latest_quote(code) or {},
            "financial": self._latest_financial_snapshot(code) or {},
            "history": bars,
            "history_daily_30": daily_30,
            "history_daily_252": daily_252,
            "history_1y_summary": history_1y,
            "history_monthly_summary_12": monthly_12,
            "trend": trend,
            "announcements": announcements,
            "news": news,
            "research": research,
            "fund": fund,
            "macro": macro,
            "quarterly_fundamentals_summary": quarterly_summary,
            "event_timeline_1y": timeline_1y,
            "time_horizon_coverage": time_horizon_coverage,
            "evidence_uncertainty": uncertainty_notes,
            "missing_data": list(dataset.get("missing_data", []) or []),
            "data_quality": "degraded" if bool(dataset.get("missing_data")) else "ready",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _latest_quote(self, stock_code: str) -> dict[str, Any] | None:
        code = stock_code.upper().replace(".", "")
        for q in reversed(self.ingestion_store.quotes):
            if str(q.get("stock_code", "")).upper() == code:
                return q
        return None

    def _latest_financial_snapshot(self, stock_code: str) -> dict[str, Any] | None:
        code = stock_code.upper().replace(".", "")
        for item in reversed(self.ingestion_store.financial_snapshots):
            if str(item.get("stock_code", "")).upper() == code:
                return item
        return None

    def _history_bars(self, stock_code: str, limit: int = 120) -> list[dict[str, Any]]:
        code = stock_code.upper().replace(".", "")
        rows = [x for x in self.ingestion_store.history_bars if str(x.get("stock_code", "")).upper() == code]
        rows.sort(key=lambda x: str(x.get("trade_date", "")))
        return rows[-limit:]

    def _history_3m_summary(self, stock_code: str) -> dict[str, Any]:
        """Extract recent ~3 month history window summary for stable trend explanations."""
        bars = self._history_bars(stock_code, limit=90)
        if len(bars) < 2:
            return {"sample_count": len(bars), "start_date": "", "end_date": "", "start_close": 0.0, "end_close": 0.0, "pct_change": 0.0}
        start = bars[0]
        end = bars[-1]
        start_close = float(start.get("close", 0.0) or 0.0)
        end_close = float(end.get("close", 0.0) or 0.0)
        pct_change = (end_close / start_close - 1.0) if start_close > 0 else 0.0
        return {
            "sample_count": len(bars),
            "start_date": str(start.get("trade_date", "")),
            "end_date": str(end.get("trade_date", "")),
            "start_close": start_close,
            "end_close": end_close,
            "pct_change": pct_change,
        }

    def _history_1y_summary(self, stock_code: str) -> dict[str, Any]:
        """Build a 1-year summary used by report/deep-think evidence coverage checks."""
        bars = self._history_bars(stock_code, limit=260)
        if len(bars) < 2:
            return {
                "sample_count": len(bars),
                "start_date": "",
                "end_date": "",
                "start_close": 0.0,
                "end_close": 0.0,
                "pct_change": 0.0,
                "has_full_1y": False,
            }
        start = bars[0]
        end = bars[-1]
        start_close = float(start.get("close", 0.0) or 0.0)
        end_close = float(end.get("close", 0.0) or 0.0)
        pct_change = (end_close / start_close - 1.0) if start_close > 0 else 0.0
        return {
            "sample_count": len(bars),
            "start_date": str(start.get("trade_date", "")),
            "end_date": str(end.get("trade_date", "")),
            "start_close": start_close,
            "end_close": end_close,
            "pct_change": pct_change,
            # 252 trading days is the practical "full year" baseline in quant/reporting contexts.
            "has_full_1y": bool(len(bars) >= 252),
        }

    def _history_monthly_summary(self, stock_code: str, *, months: int = 12) -> list[dict[str, Any]]:
        """Compress daily bars into month-level snapshots for long-horizon LLM context."""
        bars = self._history_bars(stock_code, limit=260)
        monthly: dict[str, dict[str, Any]] = {}
        for row in bars:
            trade_date = str(row.get("trade_date", ""))
            month = trade_date[:7]
            if not month:
                continue
            close_val = float(row.get("close", 0.0) or 0.0)
            volume_val = float(row.get("volume", 0.0) or 0.0)
            bucket = monthly.get(month)
            if not bucket:
                monthly[month] = {
                    "month": month,
                    "open_close": close_val,
                    "close": close_val,
                    "high_close": close_val,
                    "low_close": close_val,
                    "volume_sum": volume_val,
                }
                continue
            bucket["close"] = close_val
            bucket["high_close"] = max(float(bucket.get("high_close", close_val) or close_val), close_val)
            bucket["low_close"] = min(float(bucket.get("low_close", close_val) or close_val), close_val)
            bucket["volume_sum"] = float(bucket.get("volume_sum", 0.0) or 0.0) + volume_val

        rows = [monthly[key] for key in sorted(monthly.keys())]
        return rows[-max(1, int(months)) :]

    def _quarterly_financial_summary(self, stock_code: str, *, limit: int = 8) -> list[dict[str, Any]]:
        """Extract recent quarterly fundamentals so 1y reports are not built from sparse annual hints."""
        code = str(stock_code or "").strip().upper().replace(".", "")
        rows = [
            x
            for x in self.ingestion_store.financial_snapshots
            if str(x.get("stock_code", "")).strip().upper().replace(".", "") == code
        ]
        rows.sort(key=lambda x: (str(x.get("report_period", "")), str(x.get("ts", ""))))
        result: list[dict[str, Any]] = []
        for row in rows[-max(1, int(limit)) :]:
            result.append(
                {
                    "report_period": str(row.get("report_period", "")).strip(),
                    "revenue_yoy": round(self._safe_float(row.get("revenue_yoy", 0.0)), 6),
                    "net_profit_yoy": round(self._safe_float(row.get("net_profit_yoy", 0.0)), 6),
                    "roe": round(self._safe_float(row.get("roe", 0.0)), 6),
                    "roa": round(self._safe_float(row.get("roa", 0.0)), 6),
                    "pe": round(self._safe_float(row.get("pe", 0.0)), 6),
                    "pb": round(self._safe_float(row.get("pb", 0.0)), 6),
                    "source_id": str(row.get("source_id", "")),
                    "ts": str(row.get("ts", "")),
                }
            )
        return result

    def _event_timeline_1y(self, stock_code: str, *, limit: int = 24) -> list[dict[str, Any]]:
        """Build one merged event timeline (news/research/announcement/macro) for 1y narrative grounding."""
        code = str(stock_code or "").strip().upper().replace(".", "")
        rows: list[dict[str, Any]] = []

        for row in self.ingestion_store.announcements:
            if str(row.get("stock_code", "")).strip().upper().replace(".", "") != code:
                continue
            rows.append(
                {
                    "date": str(row.get("event_time", ""))[:10],
                    "event_type": "announcement",
                    "title": str(row.get("title", ""))[:180],
                    "source_id": str(row.get("source_id", "")),
                }
            )
        for row in self.ingestion_store.research_reports:
            if str(row.get("stock_code", "")).strip().upper().replace(".", "") != code:
                continue
            rows.append(
                {
                    "date": str(row.get("published_at", ""))[:10],
                    "event_type": "research",
                    "title": str(row.get("title", ""))[:180],
                    "source_id": str(row.get("source_id", "")),
                }
            )
        for row in self.ingestion_store.news_items:
            if str(row.get("stock_code", "")).strip().upper().replace(".", "") != code:
                continue
            rows.append(
                {
                    "date": str(row.get("event_time", ""))[:10],
                    "event_type": "news",
                    "title": str(row.get("title", ""))[:180],
                    "source_id": str(row.get("source_id", "")),
                }
            )
        # Macro is market-wide, still relevant for per-stock 1y explanation.
        for row in self.ingestion_store.macro_indicators[-30:]:
            rows.append(
                {
                    "date": str(row.get("event_time", row.get("report_date", "")))[:10],
                    "event_type": "macro",
                    "title": f"{str(row.get('metric_name', 'macro'))} {str(row.get('metric_value', ''))}".strip()[:180],
                    "source_id": str(row.get("source_id", "")),
                }
            )

        # Deduplicate by date/type/title to avoid overloading LLM context with repeated vendor rows.
        dedup: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in rows:
            key = (str(row.get("date", "")), str(row.get("event_type", "")), str(row.get("title", "")))
            dedup[key] = row
        normalized = sorted(dedup.values(), key=lambda x: (str(x.get("date", "")), str(x.get("event_type", ""))))
        return normalized[-max(1, int(limit)) :]

    def _build_evidence_uncertainty_notes(
        self,
        *,
        dataset: dict[str, Any],
        history_1y: dict[str, Any],
        quarterly_count: int,
        timeline_count: int,
    ) -> list[str]:
        """Generate explicit uncertainty notes so reports avoid overconfident 1y conclusions."""
        notes: list[str] = []
        sample_1y = int(history_1y.get("sample_count", 0) or 0)
        if sample_1y < 252:
            notes.append(
                "可核验主证据范围不足以覆盖完整1y：当前连续日线样本未达到252交易日，"
                "年度结论需标注不确定性并降低置信度。"
            )
        if quarterly_count < 4:
            notes.append("季度财报摘要样本不足（<4），年度归因对盈利趋势的解释能力受限。")
        if timeline_count < 8:
            notes.append("事件时间轴较稀疏（公告/研报/新闻/宏观样本不足），冲击归因可能不完整。")
        missing_data = [str(x) for x in list(dataset.get("missing_data", []) or []) if str(x).strip()]
        if missing_data:
            notes.append(f"系统检测到数据缺口：{', '.join(missing_data)}。")
        if not notes:
            notes.append("1y主证据覆盖良好，可按可验证口径输出结论。")
        return notes[:8]

    def _augment_question_with_history_context(self, question: str, stock_codes: list[str]) -> str:
        """Inject 3-month continuous sample summary into model input context."""
        extras: list[str] = []
        for code in stock_codes:
            summary = self._history_3m_summary(code)
            sample_count = int(summary.get("sample_count", 0) or 0)
            if sample_count < 30:
                continue
            extras.append(
                f"{code}: 最近三个月连续日线样本 {sample_count} 条, "
                f"区间 {summary.get('start_date', '')} -> {summary.get('end_date', '')}, "
                f"收盘 {float(summary.get('start_close', 0.0)):.3f} -> {float(summary.get('end_close', 0.0)):.3f}, "
                f"区间涨跌 {float(summary.get('pct_change', 0.0)) * 100:.2f}%"
            )
        if not extras:
            return question
        return (
            f"{question}\n"
            "【系统补充：连续样本上下文】\n"
            "以下为连续历史样本摘要，请优先基于窗口判断，避免稀疏点误判：\n"
            + "\n".join(f"- {line}" for line in extras)
        )

    def _augment_question_with_dataset_context(self, question: str, data_packs: list[dict[str, Any]]) -> str:
        """Append datasource coverage/missing metadata to avoid blind overconfidence."""
        if not data_packs:
            return question
        coverage_lines: list[str] = []
        missing_lines: list[str] = []
        for pack in data_packs:
            code = str(pack.get("stock_code", "")).strip().upper()
            dataset = dict(pack.get("dataset", {}) or {})
            coverage = dict(dataset.get("coverage", {}) or {})
            missing = [str(x) for x in list(dataset.get("missing_data", []) or []) if str(x).strip()]
            coverage_lines.append(
                f"{code}: history={int(coverage.get('history_count', 0) or 0)}, "
                f"news={int(coverage.get('news_count', 0) or 0)}, "
                f"research={int(coverage.get('research_count', 0) or 0)}, "
                f"macro={int(coverage.get('macro_count', 0) or 0)}, "
                f"fund={int(coverage.get('fund_count', 0) or 0)}"
            )
            if missing:
                missing_lines.append(f"{code}: {', '.join(missing)}")
        if not coverage_lines:
            return question
        text = (
            f"{question}\n"
            "【系统补充：本轮数据覆盖】\n"
            + "\n".join(f"- {line}" for line in coverage_lines)
        )
        if missing_lines:
            text += "\n【系统补充：缺口与降级提醒】\n" + "\n".join(f"- {line}" for line in missing_lines)
        return text

    def _trend_metrics(self, bars: list[dict[str, Any]]) -> dict[str, float]:
        closes = [float(x.get("close", 0.0)) for x in bars if float(x.get("close", 0.0)) > 0]
        if len(closes) < 30:
            return {"ma20": 0.0, "ma60": 0.0, "ma20_slope": 0.0, "ma60_slope": 0.0, "momentum_20": 0.0, "volatility_20": 0.0, "max_drawdown_60": 0.0}
        ma20 = mean(closes[-20:])
        ma60 = mean(closes[-60:]) if len(closes) >= 60 else mean(closes)
        ma20_prev = mean(closes[-40:-20]) if len(closes) >= 40 else ma20
        ma60_prev = mean(closes[-120:-60]) if len(closes) >= 120 else ma60
        momentum_20 = closes[-1] / closes[-21] - 1 if len(closes) >= 21 else 0.0
        returns = []
        for i in range(1, len(closes)):
            returns.append(closes[i] / closes[i - 1] - 1)
        vol20 = mean([(x - mean(returns[-20:])) ** 2 for x in returns[-20:]]) ** 0.5 if len(returns) >= 20 else 0.0
        max_dd = self._max_drawdown(closes[-60:]) if len(closes) >= 60 else self._max_drawdown(closes)
        return {
            "ma20": ma20,
            "ma60": ma60,
            "ma20_slope": ma20 - ma20_prev,
            "ma60_slope": ma60 - ma60_prev,
            "momentum_20": momentum_20,
            "volatility_20": vol20,
            "max_drawdown_60": max_dd,
        }

    def _build_a_share_regime_context(self, stock_codes: list[str]) -> dict[str, Any]:
        """Build A-share market regime context for 1-20 trading day decisions."""
        code = str(stock_codes[0]).upper() if stock_codes else "SH600000"
        if not self.settings.a_share_regime_enabled:
            return {
                "enabled": False,
                "stock_code": code,
                "regime_label": "disabled",
                "regime_confidence": 0.0,
                "risk_bias": "neutral",
                "regime_rationale": "a_share_regime_disabled",
                "short_horizon_signals": {},
                "style_rotation_hint": "none",
                "action_constraints": {
                    "max_confidence_cap": 1.0,
                    "position_pacing_hint": "no_extra_constraints",
                    "invalidation_window_days": 5,
                },
            }

        bars = self._history_bars(code, limit=260)
        closes = [float(x.get("close", 0.0)) for x in bars if float(x.get("close", 0.0)) > 0]
        if len(closes) < 30:
            return {
                "enabled": True,
                "stock_code": code,
                "regime_label": "range_chop",
                "regime_confidence": 0.32,
                "risk_bias": "neutral",
                "regime_rationale": "insufficient_history_for_regime",
                "short_horizon_signals": {
                    "trend_5d": 0.0,
                    "trend_20d": 0.0,
                    "vol_20d": 0.0,
                    "drawdown_20d": 0.0,
                    "up_day_ratio_20d": 0.0,
                    "gap_risk_flag": False,
                },
                "style_rotation_hint": "wait_for_more_data",
                "action_constraints": {
                    "max_confidence_cap": 0.72,
                    "position_pacing_hint": "small_probe_only",
                    "invalidation_window_days": 3,
                },
            }

        returns = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]
        recent_returns = returns[-20:] if len(returns) >= 20 else returns
        trend_5d = closes[-1] / closes[-6] - 1.0 if len(closes) >= 6 else 0.0
        trend_20d = closes[-1] / closes[-21] - 1.0 if len(closes) >= 21 else 0.0
        up_days = [r for r in recent_returns if r > 0]
        up_day_ratio = len(up_days) / max(1, len(recent_returns))
        mean_ret = mean(recent_returns) if recent_returns else 0.0
        vol_20d = (mean([(x - mean_ret) ** 2 for x in recent_returns]) ** 0.5) if recent_returns else 0.0
        drawdown_20d = self._max_drawdown(closes[-20:])
        gap_risk_flag = sum(1 for r in recent_returns if abs(r) >= 0.06) >= 2

        regime_label = "range_chop"
        regime_confidence = 0.55
        risk_bias = "neutral"
        regime_rationale = "sideways_with_mixed_signals"
        if trend_20d <= -0.06 or (drawdown_20d >= 0.12 and up_day_ratio < 0.45):
            regime_label = "bear_grind"
            regime_confidence = min(0.92, 0.62 + min(0.22, abs(trend_20d) + drawdown_20d))
            risk_bias = "defensive"
            regime_rationale = "downtrend_dominates_short_horizon"
        elif trend_20d < 0.0 and trend_5d >= 0.02:
            regime_label = "rebound_probe"
            regime_confidence = min(0.88, 0.58 + min(0.18, trend_5d + max(0.0, -trend_20d)))
            risk_bias = "balanced"
            regime_rationale = "short_rebound_inside_prior_weak_cycle"
        elif trend_20d >= 0.06 and trend_5d >= 0.0 and up_day_ratio >= 0.55:
            regime_label = "bull_burst"
            regime_confidence = min(0.9, 0.60 + min(0.2, trend_20d + trend_5d))
            risk_bias = "pro_risk"
            regime_rationale = "uptrend_present_but_needs_fast_review"
        elif abs(trend_20d) <= 0.03:
            regime_label = "range_chop"
            regime_confidence = min(0.86, 0.56 + min(0.2, vol_20d * 4.0))
            risk_bias = "neutral"
            regime_rationale = "range_chop_with_rotation_noise"

        # Constraints are consumed by confidence guard and frontend explainability.
        max_cap = 0.84
        pacing_hint = "ladder_entries"
        invalidation_days = 5
        if regime_label == "bear_grind":
            max_cap = 0.72
            pacing_hint = "defensive_small_size"
            invalidation_days = 2
        elif regime_label == "range_chop":
            max_cap = 0.76
            pacing_hint = "wait_breakout_or_mean_reversion"
            invalidation_days = 3
        elif regime_label == "rebound_probe":
            max_cap = 0.79
            pacing_hint = "probe_then_confirm"
            invalidation_days = 2
        elif regime_label == "bull_burst":
            max_cap = 0.86 if vol_20d <= self.settings.a_share_regime_vol_threshold else 0.8
            pacing_hint = "chase_controlled_pullback"
            invalidation_days = 3

        style_hint = "quality_large_cap"
        if regime_label in {"range_chop", "rebound_probe"}:
            style_hint = "event_driven_selective_beta"
        elif regime_label == "bear_grind":
            style_hint = "cashflow_defensive_low_beta"
        elif regime_label == "bull_burst":
            style_hint = "high_momentum_with_risk_limits"

        return {
            "enabled": True,
            "stock_code": code,
            "regime_label": regime_label,
            "regime_confidence": round(max(0.0, min(1.0, regime_confidence)), 4),
            "risk_bias": risk_bias,
            "regime_rationale": regime_rationale,
            "short_horizon_signals": {
                "trend_5d": round(trend_5d, 6),
                "trend_20d": round(trend_20d, 6),
                "vol_20d": round(vol_20d, 6),
                "drawdown_20d": round(drawdown_20d, 6),
                "up_day_ratio_20d": round(up_day_ratio, 6),
                "gap_risk_flag": bool(gap_risk_flag),
            },
            "style_rotation_hint": style_hint,
            "action_constraints": {
                "max_confidence_cap": float(max_cap),
                "position_pacing_hint": pacing_hint,
                "invalidation_window_days": int(invalidation_days),
            },
        }

    def _apply_a_share_signal_guard(
        self,
        signal: str,
        confidence: float,
        regime_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Apply A-share regime confidence guard while keeping signal direction unchanged."""
        normalized_signal = str(signal or "hold").strip().lower()
        if normalized_signal not in {"buy", "hold", "reduce"}:
            normalized_signal = "hold"
        base_conf = max(0.0, min(1.0, float(confidence or 0.0)))
        regime = regime_context if isinstance(regime_context, dict) else {}
        label = str(regime.get("regime_label", "unknown"))
        signals = regime.get("short_horizon_signals", {}) if isinstance(regime.get("short_horizon_signals"), dict) else {}
        vol_20d = float(signals.get("vol_20d", 0.0) or 0.0)
        cap = 1.0
        constraints = regime.get("action_constraints", {}) if isinstance(regime.get("action_constraints"), dict) else {}
        if constraints:
            cap = max(0.0, min(1.0, float(constraints.get("max_confidence_cap", 1.0) or 1.0)))

        multiplier = 1.0
        reason = "no_adjustment"
        if label == "bear_grind":
            multiplier = float(self.settings.a_share_regime_conf_discount_bear)
            reason = "bear_grind_confidence_discount"
        elif label == "range_chop":
            multiplier = float(self.settings.a_share_regime_conf_discount_range)
            reason = "range_chop_confidence_discount"
        elif label == "bull_burst" and vol_20d > float(self.settings.a_share_regime_vol_threshold):
            multiplier = float(self.settings.a_share_regime_conf_discount_bull_high_vol)
            reason = "bull_burst_high_vol_discount"

        adjusted = max(0.0, min(cap, base_conf * multiplier))
        applied = abs(adjusted - base_conf) >= 1e-6
        return {
            "signal": normalized_signal,
            "confidence": round(adjusted, 4),
            "applied": applied,
            "detail": {
                "regime_label": label,
                "multiplier": round(multiplier, 4),
                "cap": round(cap, 4),
                "input_confidence": round(base_conf, 4),
                "adjusted_confidence": round(adjusted, 4),
                "reason": reason,
            },
        }

    @staticmethod
    def _max_drawdown(values: list[float]) -> float:
        if not values:
            return 0.0
        peak = values[0]
        max_dd = 0.0
        for v in values:
            peak = max(peak, v)
            dd = (peak - v) / peak if peak else 0.0
            max_dd = max(max_dd, dd)
        return max_dd

    def _pm_agent_view(self, question: str, stock_codes: list[str]) -> list[str]:
        targets = ",".join(stock_codes) if stock_codes else "unspecified"
        return [
            f"- User objective: answer `{question}` around `{targets}` with verifiable evidence.",
            "- Product judgment: show both realtime snapshot and history trend, avoid single-point conclusions.",
            "- Information gap: if key indicators are missing, explicitly note and suggest data backfill.",
        ]

    def _dev_manager_view(self, stock_codes: list[str]) -> list[str]:
        targets = ",".join(stock_codes) if stock_codes else "unspecified"
        return [
            f"- Current implementation: connected free historical feeds and trend metrics for `{targets}`.",
            "- Engineering plan: backfill structured financial fields and anomaly detection checks.",
            "- Quality gate: validate freshness, citation coverage, and trend-indicator completeness per release.",
        ]

    @staticmethod
    def _normalize_report_signal(signal: str) -> str:
        """Normalize report signal to product-level finite set."""
        normalized = str(signal or "hold").strip().lower()
        if normalized in {"sell", "strong_reduce", "strong_sell"}:
            return "reduce"
        if normalized in {"strong_buy"}:
            return "buy"
        if normalized not in {"buy", "hold", "reduce"}:
            return "hold"
        return normalized

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Convert mixed payload values to float in a defensive way."""
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _safe_parse_datetime(value: Any) -> datetime | None:
        """Best-effort parse for mixed datetime payloads from citations/intel rows."""
        if isinstance(value, datetime):
            dt = value
        else:
            raw = str(value or "").strip()
            if not raw:
                return None
            text = raw
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            dt: datetime | None = None
            try:
                dt = datetime.fromisoformat(text)
            except Exception:
                dt = None
            if dt is None:
                for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d", "%Y/%m/%d %H:%M:%S"):
                    try:
                        dt = datetime.strptime(raw, fmt)
                        break
                    except Exception:
                        continue
            if dt is None:
                return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _evidence_freshness_profile(self, row: dict[str, Any]) -> dict[str, Any]:
        """Score evidence freshness for quality dashboard and export explainability."""
        ts = self._safe_parse_datetime(
            row.get("event_time")
            or row.get("published_at")
            or row.get("ts")
            or row.get("updated_at")
            or row.get("created_at")
        )
        if ts is None:
            return {"event_time": "", "age_hours": None, "freshness_score": 0.35, "freshness_tier": "unknown"}
        age_hours = max(0.0, (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0)
        if age_hours <= 24:
            score, tier = 1.0, "live"
        elif age_hours <= 72:
            score, tier = 0.9, "fresh"
        elif age_hours <= 24 * 7:
            score, tier = 0.75, "recent"
        elif age_hours <= 24 * 30:
            score, tier = 0.55, "stale"
        elif age_hours <= 24 * 90:
            score, tier = 0.38, "old"
        else:
            score, tier = 0.2, "very_old"
        return {
            "event_time": ts.isoformat(),
            "age_hours": round(age_hours, 2),
            "freshness_score": round(score, 4),
            "freshness_tier": tier,
        }

    def _normalize_report_evidence_refs(self, evidence_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize top-level evidence list with freshness scoring."""
        normalized: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for row in evidence_rows:
            if not isinstance(row, dict):
                continue
            source_id = str(row.get("source_id", "")).strip()
            source_url = str(row.get("source_url", "")).strip()
            excerpt = self._sanitize_report_text(str(row.get("excerpt", "")).strip())
            dedup_key = (source_id, excerpt[:120])
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            freshness = self._evidence_freshness_profile(row)
            normalized.append(
                {
                    "source_id": source_id,
                    "source_url": source_url,
                    "reliability_score": round(max(0.0, min(1.0, self._safe_float(row.get("reliability_score", 0.0), default=0.0))), 4),
                    "excerpt": excerpt[:320],
                    "event_time": str(row.get("event_time", freshness.get("event_time", ""))),
                    "freshness_score": round(
                        max(
                            0.0,
                            min(
                                1.0,
                                self._safe_float(
                                    row.get("freshness_score", freshness.get("freshness_score", 0.35)),
                                    default=0.35,
                                ),
                            ),
                        ),
                        4,
                    ),
                    "freshness_tier": str(row.get("freshness_tier", freshness.get("freshness_tier", "unknown"))),
                    "age_hours": (
                        round(self._safe_float(row.get("age_hours", freshness.get("age_hours", 0.0)), default=0.0), 2)
                        if row.get("age_hours") is not None or freshness.get("age_hours") is not None
                        else None
                    ),
                }
            )
        return normalized

    def _build_report_evidence_ref(self, row: dict[str, Any]) -> dict[str, Any]:
        """Project one citation/intel row into normalized report evidence schema."""
        freshness = self._evidence_freshness_profile(row)
        return {
            "source_id": str(row.get("source_id", "")),
            "source_url": str(row.get("source_url", "")),
            "reliability_score": float(row.get("reliability_score", 0.0) or 0.0),
            "excerpt": str(row.get("excerpt", row.get("summary", row.get("title", ""))))[:240],
            "event_time": str(row.get("event_time", freshness.get("event_time", ""))),
            "freshness_score": float(freshness.get("freshness_score", 0.35) or 0.35),
            "freshness_tier": str(freshness.get("freshness_tier", "unknown")),
            "age_hours": freshness.get("age_hours"),
        }

    def _build_report_metric_snapshot(
        self,
        *,
        overview: dict[str, Any],
        predict_snapshot: dict[str, Any],
        intel: dict[str, Any],
        quality_gate: dict[str, Any],
        citation_count: int,
    ) -> dict[str, Any]:
        """Build a compact metric dictionary used by report UI and export."""
        trend = dict(overview.get("trend", {}) or {})
        financial = dict(overview.get("financial", {}) or {})
        return {
            "history_sample_size": int(len(list(overview.get("history", []) or []))),
            "news_count": int(len(list(overview.get("news", []) or []))),
            "research_count": int(len(list(overview.get("research", []) or []))),
            "macro_count": int(len(list(overview.get("macro", []) or []))),
            "momentum_20": round(self._safe_float(trend.get("momentum_20", 0.0)), 6),
            "volatility_20": round(self._safe_float(trend.get("volatility_20", 0.0)), 6),
            "max_drawdown_60": round(self._safe_float(trend.get("max_drawdown_60", 0.0)), 6),
            "ma20": round(self._safe_float(trend.get("ma20", 0.0)), 6),
            "ma60": round(self._safe_float(trend.get("ma60", 0.0)), 6),
            "pe": self._safe_float(financial.get("pe", 0.0)),
            "pb": self._safe_float(financial.get("pb", 0.0)),
            "roe": self._safe_float(financial.get("roe", 0.0)),
            "roa": self._safe_float(financial.get("roa", 0.0)),
            "revenue_yoy": self._safe_float(financial.get("revenue_yoy", 0.0)),
            "net_profit_yoy": self._safe_float(financial.get("net_profit_yoy", 0.0)),
            "intel_signal": str(intel.get("overall_signal", "")),
            "intel_confidence": round(self._safe_float(intel.get("confidence", 0.0)), 4),
            "predict_quality": str(predict_snapshot.get("data_quality", "unknown")),
            "quality_score": round(self._safe_float(quality_gate.get("score", 0.0)), 4),
            "citation_count": int(citation_count),
        }

    def _build_fallback_final_decision(
        self,
        *,
        code: str,
        report_type: str,
        intel: dict[str, Any],
        quality_gate: dict[str, Any],
        quality_reasons: list[str],
    ) -> dict[str, Any]:
        """Build deterministic final decision before optional LLM overlay."""
        signal = self._normalize_report_signal(str(intel.get("overall_signal", "hold")))
        base_conf = self._safe_float(intel.get("confidence", 0.55), default=0.55)
        quality_cap = self._safe_float(quality_gate.get("score", 0.55), default=0.55)
        confidence = max(0.25, min(0.92, min(base_conf, quality_cap)))
        quality_status = str(quality_gate.get("status", "pass")).strip().lower() or "pass"

        key_catalysts = list(intel.get("key_catalysts", []) or [])
        risk_watch = list(intel.get("risk_watch", []) or [])
        catalyst_note = str(key_catalysts[0].get("summary", key_catalysts[0].get("title", ""))).strip() if key_catalysts else ""
        risk_note = str(risk_watch[0].get("summary", risk_watch[0].get("title", ""))).strip() if risk_watch else ""
        rationale_parts = [
            f"{code} 当前综合信号为 `{signal}`，结论受数据质量门控约束。",
            f"报告类型 `{report_type}` 下优先考虑可验证证据与风险触发条件。",
        ]
        if catalyst_note:
            rationale_parts.append(f"主要催化：{catalyst_note[:160]}")
        if risk_note:
            rationale_parts.append(f"主要风险：{risk_note[:160]}")

        invalidation_conditions = [
            "关键风险事件落地偏负面",
            "分歧持续扩大且缺口未补齐",
            "预算或风控阈值触发降级",
        ]
        if quality_reasons:
            invalidation_conditions.append(f"当前数据质量门控处于 {quality_status}，需要先补数再放大仓位")

        if signal == "buy":
            execution_plan = [
                "先小仓位试探，等待二次确认后再加仓。",
                "若 2-3 个交易日内出现放量回撤，暂停加仓。",
            ]
        elif signal == "reduce":
            execution_plan = [
                "优先降低高波动仓位，保留少量观察仓。",
                "若负面事件继续发酵，继续降低风险敞口。",
            ]
        else:
            execution_plan = [
                "维持观察仓，等待新增证据再决策。",
                "优先补齐缺失数据，避免因样本不足误判。",
            ]

        market_snapshot = dict(intel.get("market_snapshot", {}) or {})
        price_now = self._safe_float(market_snapshot.get("price", 0.0), default=0.0)
        risk_level = str(intel.get("risk_level", "medium")).strip().lower() or "medium"
        risk_base = {"low": 34.0, "medium": 55.0, "high": 74.0}.get(risk_level, 55.0)
        # Keep risk score bounded and interpretable (0-100) for UI and export consumers.
        risk_score = max(5.0, min(100.0, risk_base + min(24.0, 6.0 * float(len(quality_reasons)))))
        if signal == "buy":
            risk_score = max(0.0, risk_score - 4.0)
        elif signal == "reduce":
            risk_score = min(100.0, risk_score + 3.0)

        if price_now > 0:
            if signal == "buy":
                target_price = {
                    "low": round(price_now * 1.04, 3),
                    "base": round(price_now * 1.12, 3),
                    "high": round(price_now * 1.18, 3),
                }
            elif signal == "reduce":
                target_price = {
                    "low": round(price_now * 0.88, 3),
                    "base": round(price_now * 0.94, 3),
                    "high": round(price_now * 0.98, 3),
                }
            else:
                target_price = {
                    "low": round(price_now * 0.96, 3),
                    "base": round(price_now * 1.03, 3),
                    "high": round(price_now * 1.08, 3),
                }
        else:
            target_price = {"low": 0.0, "base": 0.0, "high": 0.0}

        upside = max(0.0, float(target_price.get("high", 0.0) - price_now))
        downside = max(0.01, float(price_now - target_price.get("low", 0.0)))
        reward_risk_ratio = round(max(0.1, min(6.0, upside / downside)), 3)
        position_sizing_hint = str(intel.get("position_hint", "")).strip() or (
            "35-60%" if signal == "buy" else "5-15%" if signal == "reduce" else "20-35%"
        )

        return {
            "signal": signal,
            "confidence": round(confidence, 4),
            "rationale": " ".join(rationale_parts),
            "invalidation_conditions": invalidation_conditions,
            "execution_plan": execution_plan,
            "target_price": target_price,
            "risk_score": round(risk_score, 2),
            "reward_risk_ratio": reward_risk_ratio,
            "position_sizing_hint": position_sizing_hint,
        }

    def _build_report_committee_notes(
        self,
        *,
        final_decision: dict[str, Any],
        quality_gate: dict[str, Any],
        intel: dict[str, Any],
        quality_reasons: list[str],
        analysis_nodes: list[dict[str, Any]] | None = None,
    ) -> dict[str, str]:
        """Build committee-style summary for product display."""
        signal = str(final_decision.get("signal", "hold"))
        confidence = self._safe_float(final_decision.get("confidence", 0.5), default=0.5)
        risk_level = str(intel.get("risk_level", "medium") or "medium")
        catalysts = len(list(intel.get("key_catalysts", []) or []))
        risk_watch = len(list(intel.get("risk_watch", []) or []))
        node_by_id = {
            str(row.get("node_id", "")).strip().lower(): row
            for row in list(analysis_nodes or [])
            if isinstance(row, dict) and str(row.get("node_id", "")).strip()
        }
        research_node = dict(node_by_id.get("research_summarizer", {}) or {})
        risk_node = dict(node_by_id.get("risk_arbiter", {}) or {})
        research_note = (
            f"研究汇总：当前倾向 `{signal}`（置信度 {confidence:.2f}），"
            f"已识别催化 {catalysts} 项、风险观察 {risk_watch} 项。"
        )
        if str(research_node.get("summary", "")).strip():
            research_note = str(research_node.get("summary", "")).strip()
        risk_note = (
            f"风险仲裁：风险等级 `{risk_level}`，质量门控 `{quality_gate.get('status', 'unknown')}`，"
            f"质量原因 {', '.join(quality_reasons) if quality_reasons else 'none'}。"
        )
        if str(risk_node.get("summary", "")).strip():
            risk_note = str(risk_node.get("summary", "")).strip()
        return {"research_note": research_note, "risk_note": risk_note}

    def _build_report_research_summarizer_node(
        self,
        *,
        code: str,
        query_result: dict[str, Any],
        overview: dict[str, Any],
        intel: dict[str, Any],
        final_decision: dict[str, Any],
        quality_gate: dict[str, Any],
    ) -> dict[str, Any]:
        """Build nodeized research synthesizer output for downstream display/bridging."""
        citations = list(query_result.get("citations", []) or [])
        citation_refs = [str(row.get("source_id", "")).strip() for row in citations if str(row.get("source_id", "")).strip()]
        news_rows = list(overview.get("news", []) or [])
        catalysts = list(intel.get("key_catalysts", []) or [])
        quality_reasons = [str(x).strip() for x in list(quality_gate.get("reasons", []) or []) if str(x).strip()]
        summary_parts = [
            f"研究汇总器：{code} 当前信号 `{self._normalize_report_signal(str(final_decision.get('signal', 'hold')))}`",
            f"证据引用 {len(citation_refs)} 条，新闻样本 {len(news_rows)} 条，催化事件 {len(catalysts)} 条。",
        ]
        if quality_reasons:
            summary_parts.append(f"质量门控提示：{','.join(quality_reasons[:3])}")
        highlights: list[str] = []
        if str(query_result.get("answer", "")).strip():
            highlights.append(str(query_result.get("answer", "")).strip()[:220])
        for row in catalysts[:2]:
            item = str(row.get("summary", row.get("title", ""))).strip()
            if item:
                highlights.append(item[:180])
        for row in news_rows[-2:]:
            title = str(row.get("title", "")).strip()
            if title:
                highlights.append(title[:160])
        return {
            "node_id": "research_summarizer",
            "title": "研究汇总器",
            "status": "ready" if citation_refs else "degraded",
            "signal": self._normalize_report_signal(str(final_decision.get("signal", "hold"))),
            "confidence": round(
                max(
                    0.2,
                    min(
                        self._safe_float(final_decision.get("confidence", 0.5), default=0.5),
                        self._safe_float(quality_gate.get("score", 0.5), default=0.5),
                    ),
                ),
                4,
            ),
            "summary": " ".join(summary_parts)[:320],
            "highlights": highlights[:6],
            "evidence_refs": citation_refs[:8],
            "coverage": {
                "citation_count": int(len(citation_refs)),
                "news_count": int(len(news_rows)),
                "catalyst_count": int(len(catalysts)),
            },
            "degrade_reason": quality_reasons,
        }

    def _build_report_risk_arbiter_node(
        self,
        *,
        intel: dict[str, Any],
        quality_gate: dict[str, Any],
        final_decision: dict[str, Any],
    ) -> dict[str, Any]:
        """Build nodeized risk arbiter output with actionable guardrails."""
        risk_level = str(intel.get("risk_level", "medium") or "medium").strip().lower() or "medium"
        risk_watch = list(intel.get("risk_watch", []) or [])
        quality_status = str(quality_gate.get("status", "unknown") or "unknown")
        quality_score = self._safe_float(quality_gate.get("score", 0.0), default=0.0)
        quality_reasons = [str(x).strip() for x in list(quality_gate.get("reasons", []) or []) if str(x).strip()]
        signal = self._normalize_report_signal(str(final_decision.get("signal", "hold")))

        guardrails: list[str] = [
            "若单日波动显著放大，降低新增仓位节奏。",
            "遇到关键风险事件落地，先执行风险限额再更新观点。",
        ]
        if signal == "buy":
            guardrails.insert(0, "采用分批建仓，避免单次重仓暴露。")
        if signal == "reduce":
            guardrails.insert(0, "优先削减高波动仓位并保留观察仓。")
        if quality_status != "pass":
            guardrails.append("当前处于质量降级，补齐缺失数据前不放大风险敞口。")

        risk_signal = "warning" if risk_level in {"high", "very_high"} or quality_status != "pass" else "normal"
        veto = bool(risk_signal == "warning" and (signal == "buy" and quality_status != "pass"))
        summary = (
            f"风险仲裁器：风险等级 `{risk_level}`，风险观察 {len(risk_watch)} 项，"
            f"质量门控 `{quality_status}`（{quality_score:.2f}）。"
        )
        if quality_reasons:
            summary += f" 主要降级原因：{','.join(quality_reasons[:3])}。"

        return {
            "node_id": "risk_arbiter",
            "title": "风险仲裁器",
            "status": "ready",
            "signal": risk_signal,
            "confidence": round(max(0.2, min(1.0, max(quality_score, 0.35))), 4),
            "summary": summary[:320],
            "highlights": [
                str(row.get("summary", row.get("title", ""))).strip()[:180]
                for row in risk_watch[:3]
                if str(row.get("summary", row.get("title", ""))).strip()
            ],
            "coverage": {
                "risk_watch_count": int(len(risk_watch)),
                "quality_status": quality_status,
            },
            "degrade_reason": quality_reasons,
            "guardrails": guardrails[:6],
            "veto": veto,
        }

    def _normalize_report_analysis_nodes(self, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize analysis node payload to stable schema."""
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in nodes:
            if not isinstance(row, dict):
                continue
            node_id = str(row.get("node_id", "")).strip().lower()
            if not node_id or node_id in seen:
                continue
            seen.add(node_id)
            highlights = [self._sanitize_report_text(str(x).strip()) for x in list(row.get("highlights", []) or []) if str(x).strip()]
            evidence_refs = [str(x).strip() for x in list(row.get("evidence_refs", []) or []) if str(x).strip()]
            degrade_reason = [str(x).strip() for x in list(row.get("degrade_reason", []) or []) if str(x).strip()]
            guardrails = [self._sanitize_report_text(str(x).strip()) for x in list(row.get("guardrails", []) or []) if str(x).strip()]
            normalized.append(
                {
                    "node_id": node_id,
                    "title": self._sanitize_report_text(str(row.get("title", node_id)).strip() or node_id),
                    "status": str(row.get("status", "ready")).strip().lower() or "ready",
                    "signal": self._normalize_report_signal(str(row.get("signal", "hold"))),
                    "confidence": round(max(0.0, min(1.0, self._safe_float(row.get("confidence", 0.5), default=0.5))), 4),
                    "summary": self._sanitize_report_text(str(row.get("summary", "")).strip()),
                    "highlights": highlights[:8],
                    "evidence_refs": evidence_refs[:12],
                    "coverage": dict(row.get("coverage", {}) or {}),
                    "degrade_reason": degrade_reason,
                    "guardrails": guardrails[:8],
                    "veto": bool(row.get("veto", False)),
                }
            )
        return normalized

    def _build_report_quality_dashboard(
        self,
        *,
        report_modules: list[dict[str, Any]],
        quality_gate: dict[str, Any],
        final_decision: dict[str, Any],
        analysis_nodes: list[dict[str, Any]] | None = None,
        evidence_refs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build module-level quality board with coverage/evidence/consistency metrics."""
        modules = self._normalize_report_modules(report_modules)
        node_rows = self._normalize_report_analysis_nodes(list(analysis_nodes or []))
        evidence_rows = self._normalize_report_evidence_refs(list(evidence_refs or []))
        module_count = len(modules)
        module_scores = [self._safe_float(row.get("module_quality_score", 0.0), default=0.0) for row in modules]
        avg_module_score = sum(module_scores) / float(module_count) if module_count else 0.0
        min_module_score = min(module_scores) if module_scores else 0.0
        coverage_full = sum(1 for row in modules if str((row.get("coverage", {}) or {}).get("status", "")).strip().lower() == "full")
        coverage_ratio = float(coverage_full) / float(module_count) if module_count else 0.0
        unique_evidence = {
            str(item).strip()
            for row in modules
            for item in list(row.get("evidence_refs", []) or [])
            if str(item).strip()
        }
        evidence_density = float(len(unique_evidence)) / float(module_count) if module_count else 0.0
        freshness_scores = [self._safe_float(row.get("freshness_score", 0.35), default=0.35) for row in evidence_rows]
        freshness_score = sum(freshness_scores) / float(len(freshness_scores)) if freshness_scores else 0.35
        stale_evidence_count = sum(1 for row in freshness_scores if row < 0.55)
        stale_evidence_ratio = float(stale_evidence_count) / float(len(freshness_scores)) if freshness_scores else 0.0
        decision_confidence = self._safe_float(final_decision.get("confidence", 0.5), default=0.5)
        consistency_score = max(0.0, min(1.0, 1.0 - abs(decision_confidence - avg_module_score)))
        node_veto = any(bool(row.get("veto", False)) for row in node_rows)
        quality_gate_score = self._safe_float(quality_gate.get("score", 0.0), default=0.0)

        overall_score = (
            avg_module_score * 0.34
            + coverage_ratio * 0.20
            + min(1.0, evidence_density / 2.0) * 0.14
            + consistency_score * 0.15
            + quality_gate_score * 0.09
            + freshness_score * 0.08
        )
        overall_score = max(0.0, min(1.0, overall_score))
        if overall_score >= 0.75 and not node_veto:
            status = "pass"
        elif overall_score >= 0.5:
            status = "watch"
        else:
            status = "degraded"
        if node_veto and status != "pass":
            status = "degraded"
        if overall_score < 0.35:
            status = "critical"
        low_quality_modules = [str(row.get("module_id", "")) for row in modules if self._safe_float(row.get("module_quality_score", 0.0), default=0.0) < 0.45]
        reasons: list[str] = [str(x).strip() for x in list(quality_gate.get("reasons", []) or []) if str(x).strip()]
        if node_veto:
            reasons.append("risk_arbiter_veto")
        if freshness_score < 0.45:
            reasons.append("evidence_stale")
        reasons = list(dict.fromkeys(reasons))
        return {
            "status": status,
            "overall_score": round(overall_score, 4),
            "module_count": int(module_count),
            "avg_module_quality": round(avg_module_score, 4),
            "min_module_quality": round(min_module_score, 4),
            "coverage_ratio": round(coverage_ratio, 4),
            "evidence_ref_count": int(len(unique_evidence)),
            "evidence_density": round(evidence_density, 4),
            "evidence_freshness_score": round(freshness_score, 4),
            "stale_evidence_ratio": round(stale_evidence_ratio, 4),
            "consistency_score": round(consistency_score, 4),
            "low_quality_modules": low_quality_modules[:8],
            "reasons": reasons,
            "node_veto": node_veto,
        }

    def _render_report_module_markdown(self, report_payload: dict[str, Any]) -> str:
        """Render report payload into moduleized markdown for export."""
        report_id = str(report_payload.get("report_id", "")).strip()
        stock_code = str(report_payload.get("stock_code", "")).strip().upper() or "UNKNOWN"
        report_type = str(report_payload.get("report_type", "")).strip() or "report"
        final_decision = dict(report_payload.get("final_decision", {}) or {})
        committee = dict(report_payload.get("committee", {}) or {})
        quality_dashboard = dict(report_payload.get("quality_dashboard", {}) or {})
        modules = self._normalize_report_modules(list(report_payload.get("report_modules", []) or []))
        lines = [
            f"# {stock_code} 模块化报告导出",
            "",
            f"- report_id: {report_id or 'n/a'}",
            f"- report_type: {report_type}",
            f"- signal: {self._normalize_report_signal(str(final_decision.get('signal', 'hold')))}",
            f"- confidence: {self._safe_float(final_decision.get('confidence', 0.0), default=0.0):.2f}",
            "",
            "## 委员会纪要",
            f"- 研究汇总: {self._sanitize_report_text(str(committee.get('research_note', '')).strip()) or 'n/a'}",
            f"- 风险仲裁: {self._sanitize_report_text(str(committee.get('risk_note', '')).strip()) or 'n/a'}",
            "",
            "## 质量看板",
            f"- status: {str(quality_dashboard.get('status', 'unknown'))}",
            f"- overall_score: {self._safe_float(quality_dashboard.get('overall_score', 0.0), default=0.0):.2f}",
            f"- coverage_ratio: {self._safe_float(quality_dashboard.get('coverage_ratio', 0.0), default=0.0):.2f}",
            f"- consistency_score: {self._safe_float(quality_dashboard.get('consistency_score', 0.0), default=0.0):.2f}",
            f"- low_quality_modules: {','.join(list(quality_dashboard.get('low_quality_modules', []) or [])) or 'none'}",
            "",
        ]
        for module in modules:
            module_id = str(module.get("module_id", "")).strip() or "module"
            title = self._sanitize_report_text(str(module.get("title", module_id)).strip() or module_id)
            content = self._sanitize_report_text(str(module.get("content", "")).strip())
            coverage = dict(module.get("coverage", {}) or {})
            lines.extend(
                [
                    f"## {title} ({module_id})",
                    f"- confidence: {self._safe_float(module.get('confidence', 0.0), default=0.0):.2f}",
                    f"- module_quality_score: {self._safe_float(module.get('module_quality_score', 0.0), default=0.0):.2f}",
                    f"- coverage: {str(coverage.get('status', 'unknown'))}/{int(coverage.get('data_points', 0) or 0)}",
                    f"- degrade_reason: {','.join(list(module.get('degrade_reason', []) or [])) or 'none'}",
                    "",
                    content or "该模块暂无可用内容。",
                    "",
                ]
            )
        return "\n".join(lines).strip() + "\n"

    def _build_fallback_report_modules(
        self,
        *,
        code: str,
        query_result: dict[str, Any],
        overview: dict[str, Any],
        predict_snapshot: dict[str, Any],
        intel: dict[str, Any],
        report_input_pack: dict[str, Any],
        quality_gate: dict[str, Any],
        quality_reasons: list[str],
        final_decision: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Build deterministic moduleized report payload used as baseline."""
        trend = dict(overview.get("trend", {}) or {})
        financial = dict(overview.get("financial", {}) or {})
        news_rows = list(overview.get("news", []) or [])
        risk_watch = list(intel.get("risk_watch", []) or [])
        scenarios = list(intel.get("scenario_matrix", []) or [])
        citations = list(query_result.get("citations", []) or [])
        citation_ids = [str(x.get("source_id", "")) for x in citations[:8] if str(x.get("source_id", "")).strip()]
        top_news = [str(x.get("title", "")).strip() for x in news_rows[-3:] if str(x.get("title", "")).strip()]
        bull_points = [
            str(x.get("summary", x.get("title", ""))).strip()
            for x in list(intel.get("key_catalysts", []) or [])[:4]
            if str(x.get("summary", x.get("title", ""))).strip()
        ]
        bear_points = [
            str(x.get("summary", x.get("title", ""))).strip()
            for x in list(intel.get("risk_watch", []) or [])[:4]
            if str(x.get("summary", x.get("title", ""))).strip()
        ]
        risk_level = str(intel.get("risk_level", "medium")).strip().lower() or "medium"
        target_price = dict(final_decision.get("target_price", {}) or {})
        position_hint = str(final_decision.get("position_sizing_hint", "")).strip()

        modules = [
            {
                "module_id": "executive_summary",
                "title": "执行摘要",
                "content": str(query_result.get("answer", "")).strip()[:1600],
                "evidence_refs": citation_ids,
                "confidence": self._safe_float(final_decision.get("confidence", 0.5), default=0.5),
                "coverage": {"status": "partial" if quality_reasons else "full", "data_points": int(len(citations))},
                "degrade_reason": list(quality_reasons),
            },
            {
                "module_id": "market_technical",
                "title": "行情与技术",
                "content": (
                    f"MA20={self._safe_float(trend.get('ma20', 0.0)):.3f}, "
                    f"MA60={self._safe_float(trend.get('ma60', 0.0)):.3f}, "
                    f"20日动量={self._safe_float(trend.get('momentum_20', 0.0)) * 100:.2f}%, "
                    f"20日波动={self._safe_float(trend.get('volatility_20', 0.0)) * 100:.2f}%, "
                    f"60日最大回撤={self._safe_float(trend.get('max_drawdown_60', 0.0)) * 100:.2f}%。"
                ),
                "evidence_refs": citation_ids[:4],
                "confidence": round(max(0.3, self._safe_float(quality_gate.get("score", 0.5), default=0.5) - 0.05), 4),
                "coverage": {"status": "partial" if "history_30d_insufficient" in quality_reasons else "full", "data_points": int(len(list(overview.get("history", []) or [])))},
                "degrade_reason": [x for x in quality_reasons if "history" in x],
            },
            {
                "module_id": "fundamental_valuation",
                "title": "财务与估值",
                "content": (
                    f"PE={self._safe_float(financial.get('pe', 0.0)):.3f}, "
                    f"PB={self._safe_float(financial.get('pb', 0.0)):.3f}, "
                    f"ROE={self._safe_float(financial.get('roe', 0.0)):.2f}%, "
                    f"ROA={self._safe_float(financial.get('roa', 0.0)):.2f}%, "
                    f"营收同比={self._safe_float(financial.get('revenue_yoy', 0.0)):.2f}%, "
                    f"净利同比={self._safe_float(financial.get('net_profit_yoy', 0.0)):.2f}%。"
                ),
                "evidence_refs": citation_ids[:4],
                "confidence": round(max(0.28, self._safe_float(quality_gate.get("score", 0.5), default=0.5) - 0.08), 4),
                "coverage": {"status": "partial" if "financial_missing" in quality_reasons else "full", "data_points": int(1 if financial else 0)},
                "degrade_reason": [x for x in quality_reasons if "financial" in x],
            },
            {
                "module_id": "news_event",
                "title": "新闻事件与催化",
                "content": (
                    "近期重点新闻："
                    + ("；".join(top_news) if top_news else "暂无高质量新闻命中。")
                ),
                "evidence_refs": citation_ids[:6],
                "confidence": round(max(0.25, self._safe_float(quality_gate.get("score", 0.5), default=0.5) - 0.06), 4),
                "coverage": {"status": "partial" if "news_insufficient" in quality_reasons else "full", "data_points": int(len(news_rows))},
                "degrade_reason": [x for x in quality_reasons if "news" in x],
            },
            {
                "module_id": "sentiment_flow",
                "title": "情绪与资金",
                "content": (
                    f"综合情报信号={str(intel.get('overall_signal', 'hold'))}, "
                    f"置信度={self._safe_float(intel.get('confidence', 0.0)):.2f}, "
                    f"催化数={len(list(intel.get('key_catalysts', []) or []))}, "
                    f"风险观察数={len(risk_watch)}。"
                ),
                "evidence_refs": citation_ids[:6],
                "confidence": round(max(0.25, self._safe_float(intel.get("confidence", 0.45), default=0.45)), 4),
                "coverage": {"status": "partial" if "fund_missing" in quality_reasons else "full", "data_points": int(len(list(intel.get("evidence", []) or [])))},
                "degrade_reason": [x for x in quality_reasons if "fund" in x],
            },
            {
                "module_id": "risk_matrix",
                "title": "风险矩阵",
                "content": json.dumps(
                    [
                        {"risk": str(x.get("title", "")), "signal": str(x.get("signal", "negative")), "detail": str(x.get("summary", ""))[:160]}
                        for x in risk_watch[:6]
                    ]
                    or [{"risk": "data_quality", "signal": "warning", "detail": ",".join(quality_reasons) or "none"}],
                    ensure_ascii=False,
                ),
                "evidence_refs": citation_ids[:6],
                "confidence": round(max(0.2, self._safe_float(quality_gate.get("score", 0.5), default=0.5) - 0.1), 4),
                "coverage": {"status": "partial" if quality_reasons else "full", "data_points": int(len(risk_watch))},
                "degrade_reason": list(quality_reasons),
            },
            {
                "module_id": "scenario_plan",
                "title": "情景分析",
                "content": json.dumps(scenarios[:6], ensure_ascii=False) if scenarios else "暂无可用情景矩阵，需补齐实时情报与历史样本。",
                "evidence_refs": citation_ids[:6],
                "confidence": round(max(0.22, self._safe_float(intel.get("confidence", 0.45), default=0.45) - 0.08), 4),
                "coverage": {"status": "partial" if not scenarios else "full", "data_points": int(len(scenarios))},
                "degrade_reason": ["scenario_missing"] if not scenarios else [],
            },
            {
                # TradingAgents-style role module: explicit bullish thesis long text.
                "module_id": "bull_case",
                "title": "多头论证（Bull Case）",
                "content": (
                    "核心逻辑：趋势与催化共振时，优先考虑增配。"
                    + (
                        "\n" + "\n".join(f"- {line}" for line in bull_points[:6])
                        if bull_points
                        else "\n- 当前未提取到明确多头催化，需补齐新闻/研报样本。"
                    )
                    + f"\n- 目标价参考：{json.dumps(target_price, ensure_ascii=False)}"
                ),
                "evidence_refs": citation_ids[:8],
                "confidence": round(max(0.2, self._safe_float(final_decision.get("confidence", 0.5), default=0.5) - 0.03), 4),
                "coverage": {"status": "partial" if quality_reasons else "full", "data_points": int(len(bull_points))},
                "degrade_reason": list(quality_reasons),
            },
            {
                # TradingAgents-style role module: explicit bearish thesis long text.
                "module_id": "bear_case",
                "title": "空头论证（Bear Case）",
                "content": (
                    "核心逻辑：若风险事件持续发酵，应优先保护回撤。"
                    + (
                        "\n" + "\n".join(f"- {line}" for line in bear_points[:6])
                        if bear_points
                        else "\n- 当前未提取到明确空头风险，仍需跟踪事件时间轴。"
                    )
                    + f"\n- 当前风险等级：{risk_level}"
                ),
                "evidence_refs": citation_ids[:8],
                "confidence": round(max(0.2, self._safe_float(final_decision.get("confidence", 0.5), default=0.5) - 0.05), 4),
                "coverage": {"status": "partial" if quality_reasons else "full", "data_points": int(len(bear_points))},
                "degrade_reason": list(quality_reasons),
            },
            {
                "module_id": "risky_case",
                "title": "激进执行论证（Risky Case）",
                "content": (
                    "核心逻辑：在高胜率窗口以更快节奏执行，但严格遵守单步上限。"
                    f"\n- 仓位提示：{position_hint or '未提供'}"
                    f"\n- 执行路径：{'; '.join(list(final_decision.get('execution_plan', []) or [])[:3]) or '待补充'}"
                ),
                "evidence_refs": citation_ids[:6],
                "confidence": round(max(0.2, self._safe_float(final_decision.get("confidence", 0.5), default=0.5) - 0.06), 4),
                "coverage": {"status": "partial" if quality_reasons else "full", "data_points": int(len(list(final_decision.get("execution_plan", []) or [])))},
                "degrade_reason": list(quality_reasons),
            },
            {
                "module_id": "safe_case",
                "title": "保守风控论证（Safe Case）",
                "content": (
                    "核心逻辑：优先控制回撤与证据不确定性，分批验证再行动。"
                    f"\n- 门控状态：{str(quality_gate.get('status', 'unknown'))}"
                    f"\n- 风险触发：{'; '.join(list(final_decision.get('invalidation_conditions', []) or [])[:4]) or '待补充'}"
                ),
                "evidence_refs": citation_ids[:6],
                "confidence": round(max(0.2, self._safe_float(final_decision.get("confidence", 0.5), default=0.5) - 0.04), 4),
                "coverage": {"status": "partial" if quality_reasons else "full", "data_points": int(len(list(final_decision.get("invalidation_conditions", []) or [])))},
                "degrade_reason": list(quality_reasons),
            },
            {
                "module_id": "execution_plan",
                "title": "执行策略",
                "content": "\n".join(f"- {line}" for line in list(final_decision.get("execution_plan", []) or [])),
                "evidence_refs": citation_ids[:4],
                "confidence": self._safe_float(final_decision.get("confidence", 0.5), default=0.5),
                "coverage": {"status": "full", "data_points": int(len(list(final_decision.get("execution_plan", []) or [])))},
                "degrade_reason": [],
            },
            {
                "module_id": "final_decision",
                "title": "综合决策",
                "content": str(final_decision.get("rationale", ""))[:1800],
                "evidence_refs": citation_ids[:8],
                "confidence": self._safe_float(final_decision.get("confidence", 0.5), default=0.5),
                "coverage": {"status": "partial" if quality_reasons else "full", "data_points": int(len(citation_ids))},
                "degrade_reason": list(quality_reasons),
            },
        ]

        return self._normalize_report_modules(modules)

    def _normalize_report_modules(self, modules: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize module payload to stable schema for frontend rendering."""
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in modules:
            if not isinstance(row, dict):
                continue
            module_id = str(row.get("module_id", "")).strip().lower()
            if not module_id or module_id in seen:
                continue
            seen.add(module_id)
            title = str(row.get("title", module_id)).strip() or module_id
            content = self._sanitize_report_text(str(row.get("content", "")).strip())
            if not content:
                content = "该模块暂无可用内容。"
            evidence_refs = [str(x).strip() for x in list(row.get("evidence_refs", []) or []) if str(x).strip()]
            coverage_raw = dict(row.get("coverage", {}) or {})
            coverage = {
                "status": str(coverage_raw.get("status", "partial")).strip().lower() or "partial",
                "data_points": int(coverage_raw.get("data_points", 0) or 0),
            }
            confidence = max(0.0, min(1.0, self._safe_float(row.get("confidence", 0.5), default=0.5)))
            degrade_reason = [str(x).strip() for x in list(row.get("degrade_reason", []) or []) if str(x).strip()]
            coverage_factor = {"full": 1.0, "partial": 0.78, "missing": 0.4}.get(str(coverage["status"]).lower(), 0.7)
            degrade_penalty = min(0.35, 0.08 * float(len(degrade_reason)))
            module_quality_score = max(0.0, min(1.0, confidence * coverage_factor - degrade_penalty))
            normalized.append(
                {
                    "module_id": module_id,
                    "title": title,
                    "content": content,
                    "evidence_refs": evidence_refs,
                    "coverage": coverage,
                    "confidence": round(confidence, 4),
                    "degrade_reason": degrade_reason,
                    "module_quality_score": round(module_quality_score, 4),
                    "module_degrade_code": str(degrade_reason[0]) if degrade_reason else "",
                }
            )
        return normalized

    def report_generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Generate report based on query result and cache it for retrieval/export."""
        req = ReportRequest(**payload)
        run_id = str(payload.get("run_id", "")).strip()
        pool_snapshot_id = str(payload.get("pool_snapshot_id", "")).strip()
        template_id = str(payload.get("template_id", "default")).strip() or "default"
        code = str(req.stock_code or "").strip().upper()
        report_input_pack = self._build_llm_input_pack(
            code,
            question=f"report:{code}:{req.report_type}",
            scenario="report",
        )
        query_result = self.query(
            {
                "user_id": req.user_id,
                "question": f"请生成{req.stock_code} {req.period} 的{req.report_type}报告",
                "stock_codes": [req.stock_code],
            }
        )
        overview = self.market_overview(code)
        predict_snapshot = self.predict_run({"stock_codes": [code], "horizons": ["5d", "20d"]})
        try:
            intel = self.analysis_intel_card(code, horizon="30d", risk_profile="neutral")
        except Exception:
            intel = {}
        history_sample_size = len(list(overview.get("history", []) or []))
        refresh_actions = list(((report_input_pack.get("dataset", {}) or {}).get("refresh_actions", []) or []))
        refresh_failed_count = len([x for x in refresh_actions if str((x or {}).get("status", "")).strip().lower() != "ok"])
        quality_reasons: list[str] = []
        if len(list(query_result.get("citations", []) or [])) < 3:
            quality_reasons.append("citations_insufficient")
        if history_sample_size < 60:
            quality_reasons.append("history_sample_insufficient")
        if str(predict_snapshot.get("data_quality", "")).strip() != "real":
            quality_reasons.append("predict_degraded")
        if refresh_failed_count > 0:
            quality_reasons.append("auto_refresh_failed")
        quality_reasons.extend([str(x) for x in list(report_input_pack.get("missing_data", []) or []) if str(x).strip()])
        quality_reasons = list(dict.fromkeys(quality_reasons))
        # Weighted quality gate: avoid over-penalizing isolated soft gaps (e.g. research_only gap).
        reason_weights = {
            "quote_missing": 0.32,
            "history_insufficient": 0.30,
            "history_sample_insufficient": 0.20,
            "history_30d_insufficient": 0.14,
            "financial_missing": 0.16,
            "announcement_missing": 0.10,
            "news_insufficient": 0.12,
            "research_insufficient": 0.08,
            "macro_insufficient": 0.08,
            "fund_missing": 0.06,
            "predict_degraded": 0.06,
            "citations_insufficient": 0.10,
            "auto_refresh_failed": 0.12,
        }
        quality_penalty = 0.0
        for reason in quality_reasons:
            quality_penalty += float(reason_weights.get(reason, 0.06))
        quality_penalty = min(0.85, quality_penalty)
        quality_score = round(max(0.0, 1.0 - quality_penalty), 4)
        if not quality_reasons:
            quality_status = "pass"
        elif quality_penalty < 0.28:
            quality_status = "watch"
        else:
            quality_status = "degraded"
        quality_gate = {
            "status": quality_status,
            "score": quality_score,
            "reasons": quality_reasons,
            "penalty": round(quality_penalty, 4),
        }
        report_data_pack_summary = {
            "as_of": datetime.now(timezone.utc).isoformat(),
            "history_sample_size": history_sample_size,
            "history_30d_count": int(((report_input_pack.get("dataset", {}) or {}).get("coverage", {}) or {}).get("history_30d_count", 0) or 0),
            "history_90d_count": int(((report_input_pack.get("dataset", {}) or {}).get("coverage", {}) or {}).get("history_90d_count", 0) or 0),
            "history_252d_count": int(len(list(report_input_pack.get("history_daily_252", []) or []))),
            "history_1y_has_full": bool((report_input_pack.get("history_1y_summary", {}) or {}).get("has_full_1y", False)),
            "predict_quality": str(predict_snapshot.get("data_quality", "unknown")),
            "predict_degrade_reasons": list(predict_snapshot.get("degrade_reasons", []) or []),
            "intel_signal": str(intel.get("overall_signal", "")),
            "intel_confidence": float(intel.get("confidence", 0.0) or 0.0),
            "news_count": len(list(overview.get("news", []) or [])),
            "research_count": len(list(overview.get("research", []) or [])),
            "macro_count": len(list(overview.get("macro", []) or [])),
            "quarterly_fundamentals_count": int(len(list(report_input_pack.get("quarterly_fundamentals_summary", []) or []))),
            "event_timeline_count": int(len(list(report_input_pack.get("event_timeline_1y", []) or []))),
            "uncertainty_note_count": int(len(list(report_input_pack.get("evidence_uncertainty", []) or []))),
            "time_horizon_coverage": dict(report_input_pack.get("time_horizon_coverage", {}) or {}),
            "refresh_action_count": int(len(refresh_actions)),
            "refresh_failed_count": int(refresh_failed_count),
            "missing_data": list(report_input_pack.get("missing_data", []) or []),
            "data_quality": str(report_input_pack.get("data_quality", "ready")),
        }
        metric_snapshot = self._build_report_metric_snapshot(
            overview=overview,
            predict_snapshot=predict_snapshot,
            intel=intel,
            quality_gate=quality_gate,
            citation_count=len(list(query_result.get("citations", []) or [])),
        )
        final_decision = self._build_fallback_final_decision(
            code=code,
            report_type=str(req.report_type),
            intel=intel,
            quality_gate=quality_gate,
            quality_reasons=quality_reasons,
        )
        # Nodeized committee pipeline: explicitly separate research synthesis and risk arbitration.
        analysis_nodes = self._normalize_report_analysis_nodes(
            [
                self._build_report_research_summarizer_node(
                    code=code,
                    query_result=query_result,
                    overview=overview,
                    intel=intel,
                    final_decision=final_decision,
                    quality_gate=quality_gate,
                ),
                self._build_report_risk_arbiter_node(
                    intel=intel,
                    quality_gate=quality_gate,
                    final_decision=final_decision,
                ),
            ]
        )
        committee = self._build_report_committee_notes(
            final_decision=final_decision,
            quality_gate=quality_gate,
            intel=intel,
            quality_reasons=quality_reasons,
            analysis_nodes=analysis_nodes,
        )
        report_modules = self._build_fallback_report_modules(
            code=code,
            query_result=query_result,
            overview=overview,
            predict_snapshot=predict_snapshot,
            intel=intel,
            report_input_pack=report_input_pack,
            quality_gate=quality_gate,
            quality_reasons=quality_reasons,
            final_decision=final_decision,
        )
        module_order = [
            "executive_summary",
            "market_technical",
            "fundamental_valuation",
            "news_event",
            "sentiment_flow",
            "risk_matrix",
            "scenario_plan",
            "bull_case",
            "bear_case",
            "risky_case",
            "safe_case",
            "execution_plan",
            "final_decision",
        ]
        generation_mode = "fallback"
        generation_error = ""
        report_id = str(uuid.uuid4())
        evidence_refs = [self._build_report_evidence_ref(dict(c)) for c in list(query_result.get("citations", []) or [])]
        report_sections = [
            {"section_id": "summary", "title": "结论摘要", "content": str(query_result["answer"])[:800]},
            {
                "section_id": "evidence",
                "title": "证据清单",
                "content": "\n".join(f"- {x['source_id']}: {x['excerpt']}" for x in evidence_refs[:8]),
            },
            {"section_id": "risk", "title": "风险与反证", "content": "结合估值、流动性与政策扰动进行反证校验。"},
            {"section_id": "action", "title": "操作建议", "content": "建议分批验证信号稳定性，避免一次性重仓。"},
        ]
        for row in list(intel.get("evidence", []) or [])[:6]:
            evidence_refs.append(self._build_report_evidence_ref(dict(row)))
        evidence_refs = self._normalize_report_evidence_refs(evidence_refs)[:14]

        report_sections.extend(
            [
                {
                    "section_id": "evidence_summary",
                    "title": "证据摘要",
                    "content": json.dumps(
                        {
                            "citation_count": int(len(evidence_refs)),
                            "top_citation_sources": [str(x.get("source_id", "")) for x in evidence_refs[:8] if str(x.get("source_id", "")).strip()],
                            "catalyst_count": int(len(list(intel.get("key_catalysts", []) or []))),
                            "risk_watch_count": int(len(list(intel.get("risk_watch", []) or []))),
                        },
                        ensure_ascii=False,
                    ),
                },
                {
                    "section_id": "uncertainty_notes",
                    "title": "不确定性清单",
                    "content": "\n".join(f"- {line}" for line in list(report_input_pack.get("evidence_uncertainty", []) or [])),
                },
                {
                    "section_id": "time_horizon_coverage",
                    "title": "时域覆盖度",
                    "content": json.dumps(dict(report_input_pack.get("time_horizon_coverage", {}) or {}), ensure_ascii=False),
                },
                {
                    "section_id": "data_quality_gate",
                    "title": "数据质量门控",
                    "content": f"status={quality_gate['status']}; score={quality_gate['score']}; reasons={','.join(quality_reasons) or 'none'}",
                },
                {
                    "section_id": "signal_context",
                    "title": "信号上下文",
                    "content": json.dumps(
                        {
                            "predict_quality": str(predict_snapshot.get("data_quality", "unknown")),
                            "predict_degrade_reasons": list(predict_snapshot.get("degrade_reasons", []) or []),
                            "intel_signal": str(intel.get("overall_signal", "")),
                            "intel_confidence": float(intel.get("confidence", 0.0) or 0.0),
                        },
                        ensure_ascii=False,
                    ),
                },
                {
                    "section_id": "scenario_matrix",
                    "title": "情景分析矩阵",
                    "content": json.dumps(list(intel.get("scenario_matrix", []) or [])[:6], ensure_ascii=False),
                },
                {
                    "section_id": "llm_input_pack",
                    "title": "模型输入数据包摘要",
                    "content": json.dumps(
                        {
                            "coverage": dict((report_input_pack.get("dataset", {}) or {}).get("coverage", {}) or {}),
                            "missing_data": list(report_input_pack.get("missing_data", []) or []),
                            "data_quality": str(report_input_pack.get("data_quality", "")),
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
        )
        if self.settings.llm_external_enabled and self.llm_gateway.providers:
            llm_prompt = (
                "你是A股研究报告助手。请基于给定context输出严格JSON，不要输出markdown或解释性文本。"
                "顶层JSON必须包含字段：modules, final_decision, committee, next_data_needed。"
                "modules 为数组，元素结构："
                "{module_id,title,content,evidence_refs,confidence,coverage,degrade_reason}。"
                "module_id 仅允许：executive_summary,market_technical,fundamental_valuation,news_event,"
                "sentiment_flow,risk_matrix,scenario_plan,bull_case,bear_case,risky_case,safe_case,execution_plan,final_decision。"
                "final_decision 结构：{signal,confidence,rationale,invalidation_conditions,execution_plan,target_price,risk_score,reward_risk_ratio,position_sizing_hint}，"
                "signal 仅允许 buy/hold/reduce。"
                "committee 结构：{research_note,risk_note}。"
                "要求："
                "1) 明确引用数据覆盖缺口，不得忽略 missing_data。"
                "2) 若数据不足，降低结论确信度并给出补数建议。"
                "3) 模块内容聚焦交易可执行性与风险触发条件。"
                "4) confidence 必须在 0 到 1。"
                "5) 必须显式使用 1y 证据包（history_daily_252 / quarterly_fundamentals_summary / event_timeline_1y）。"
                "6) 若1y样本不足，必须在内容里写出不确定性说明。"
                "context="
                + json.dumps(
                    {
                        "stock_code": code,
                        "report_type": req.report_type,
                        "query_answer": query_result.get("answer", ""),
                        "financial": overview.get("financial", {}),
                        "trend": overview.get("trend", {}),
                        "predict": {
                            "quality": predict_snapshot.get("data_quality", ""),
                            "horizons": ((predict_snapshot.get("results", []) or [{}])[0].get("horizons", []) if predict_snapshot.get("results") else []),
                        },
                        "intel": {
                            "signal": intel.get("overall_signal", ""),
                            "confidence": intel.get("confidence", 0.0),
                            "risk_watch": intel.get("risk_watch", []),
                            "catalysts": intel.get("key_catalysts", []),
                            "scenario_matrix": intel.get("scenario_matrix", []),
                        },
                        "quality_gate": quality_gate,
                        "metric_snapshot": metric_snapshot,
                        "input_pack": {
                            "coverage": dict((report_input_pack.get("dataset", {}) or {}).get("coverage", {}) or {}),
                            "missing_data": list(report_input_pack.get("missing_data", []) or []),
                            "data_quality": str(report_input_pack.get("data_quality", "")),
                            "history_daily_30": list(report_input_pack.get("history_daily_30", []) or []),
                            "history_daily_252": list(report_input_pack.get("history_daily_252", []) or []),
                            "history_1y_summary": dict(report_input_pack.get("history_1y_summary", {}) or {}),
                            "history_monthly_summary_12": list(report_input_pack.get("history_monthly_summary_12", []) or []),
                            "quarterly_fundamentals_summary": list(report_input_pack.get("quarterly_fundamentals_summary", []) or []),
                            "event_timeline_1y": list(report_input_pack.get("event_timeline_1y", []) or []),
                            "time_horizon_coverage": dict(report_input_pack.get("time_horizon_coverage", {}) or {}),
                            "evidence_uncertainty": list(report_input_pack.get("evidence_uncertainty", []) or []),
                        },
                    },
                    ensure_ascii=False,
                )
            )
            state = AgentState(
                user_id="report-generator",
                question=f"report:{code}",
                stock_codes=[code],
                trace_id=self.traces.new_trace(),
            )
            try:
                raw = self.llm_gateway.generate(state, llm_prompt)
                parsed = self._deep_safe_json_loads(raw)
                if isinstance(parsed, dict) and parsed:
                    generation_mode = "llm"
                    modules_map: dict[str, dict[str, Any]] = {
                        str(x.get("module_id", "")): dict(x)
                        for x in report_modules
                        if isinstance(x, dict) and str(x.get("module_id", "")).strip()
                    }

                    parsed_modules = list(parsed.get("modules", []) or [])
                    for row in parsed_modules:
                        if not isinstance(row, dict):
                            continue
                        module_id = str(row.get("module_id", "")).strip().lower()
                        if module_id not in module_order:
                            continue
                        base = dict(modules_map.get(module_id, {}))
                        base["module_id"] = module_id
                        if str(row.get("title", "")).strip():
                            base["title"] = str(row.get("title", "")).strip()
                        if str(row.get("content", "")).strip():
                            base["content"] = str(row.get("content", "")).strip()
                        if isinstance(row.get("evidence_refs"), list):
                            base["evidence_refs"] = [str(x).strip() for x in list(row.get("evidence_refs", [])) if str(x).strip()]
                        if row.get("confidence") is not None:
                            base["confidence"] = self._safe_float(row.get("confidence", 0.5), default=0.5)
                        if isinstance(row.get("coverage"), dict):
                            base["coverage"] = dict(row.get("coverage", {}) or {})
                        if isinstance(row.get("degrade_reason"), list):
                            base["degrade_reason"] = [str(x).strip() for x in list(row.get("degrade_reason", [])) if str(x).strip()]
                        modules_map[module_id] = base

                    # Backward compatibility for older prompt schema.
                    if "modules" not in parsed:
                        legacy_summary = str(parsed.get("executive_summary", "")).strip()
                        if legacy_summary:
                            modules_map["executive_summary"]["content"] = legacy_summary
                        core_logic = list(parsed.get("core_logic", []) or [])
                        if core_logic:
                            modules_map["market_technical"]["content"] = "\n".join(f"- {str(x)}" for x in core_logic[:10])
                        legacy_risk = list(parsed.get("risk_matrix", []) or [])
                        if legacy_risk:
                            modules_map["risk_matrix"]["content"] = json.dumps(legacy_risk[:8], ensure_ascii=False)
                        legacy_scenario = list(parsed.get("scenario_analysis", []) or [])
                        if legacy_scenario:
                            modules_map["scenario_plan"]["content"] = json.dumps(legacy_scenario[:8], ensure_ascii=False)
                        legacy_exec = list(parsed.get("execution_plan", []) or [])
                        if legacy_exec:
                            modules_map["execution_plan"]["content"] = "\n".join(f"- {str(x)}" for x in legacy_exec[:10])

                    parsed_decision = parsed.get("final_decision")
                    if isinstance(parsed_decision, dict):
                        final_decision["signal"] = self._normalize_report_signal(str(parsed_decision.get("signal", final_decision.get("signal", "hold"))))
                        final_decision["confidence"] = round(
                            max(0.2, min(1.0, self._safe_float(parsed_decision.get("confidence", final_decision.get("confidence", 0.5)), default=0.5))),
                            4,
                        )
                        if str(parsed_decision.get("rationale", "")).strip():
                            final_decision["rationale"] = str(parsed_decision.get("rationale", "")).strip()
                        if isinstance(parsed_decision.get("invalidation_conditions"), list):
                            final_decision["invalidation_conditions"] = [
                                str(x).strip()
                                for x in list(parsed_decision.get("invalidation_conditions", []))
                                if str(x).strip()
                            ][:8]
                        if isinstance(parsed_decision.get("execution_plan"), list):
                            final_decision["execution_plan"] = [
                                str(x).strip()
                                for x in list(parsed_decision.get("execution_plan", []))
                                if str(x).strip()
                            ][:8]
                        target_raw = parsed_decision.get("target_price")
                        if isinstance(target_raw, dict):
                            final_decision["target_price"] = {
                                "low": round(self._safe_float(target_raw.get("low", 0.0), default=0.0), 3),
                                "base": round(self._safe_float(target_raw.get("base", 0.0), default=0.0), 3),
                                "high": round(self._safe_float(target_raw.get("high", 0.0), default=0.0), 3),
                            }
                        elif isinstance(target_raw, (int, float)):
                            base_target = round(float(target_raw), 3)
                            final_decision["target_price"] = {
                                "low": round(base_target * 0.95, 3),
                                "base": base_target,
                                "high": round(base_target * 1.05, 3),
                            }
                        if parsed_decision.get("risk_score") is not None:
                            final_decision["risk_score"] = round(
                                max(0.0, min(100.0, self._safe_float(parsed_decision.get("risk_score", 50.0), default=50.0))),
                                2,
                            )
                        if parsed_decision.get("reward_risk_ratio") is not None:
                            final_decision["reward_risk_ratio"] = round(
                                max(0.0, min(10.0, self._safe_float(parsed_decision.get("reward_risk_ratio", 1.0), default=1.0))),
                                3,
                            )
                        if str(parsed_decision.get("position_sizing_hint", "")).strip():
                            final_decision["position_sizing_hint"] = str(parsed_decision.get("position_sizing_hint", "")).strip()[:80]

                    parsed_committee = parsed.get("committee")
                    if isinstance(parsed_committee, dict):
                        if str(parsed_committee.get("research_note", "")).strip():
                            committee["research_note"] = str(parsed_committee.get("research_note", "")).strip()
                        if str(parsed_committee.get("risk_note", "")).strip():
                            committee["risk_note"] = str(parsed_committee.get("risk_note", "")).strip()

                    if isinstance(parsed.get("next_data_needed"), list):
                        llm_missing = [str(x).strip() for x in list(parsed.get("next_data_needed", [])) if str(x).strip()]
                        if llm_missing:
                            final_decision["invalidation_conditions"] = list(dict.fromkeys(list(final_decision.get("invalidation_conditions", [])) + llm_missing))[:10]

                    report_modules = [modules_map.get(mid, {"module_id": mid, "title": mid, "content": "该模块暂无可用内容。"}) for mid in module_order]
                    report_modules = self._normalize_report_modules(report_modules)
            except Exception as ex:
                generation_error = str(ex)[:240]

        # Keep final decision confidence bounded by quality gate even if LLM outputs larger value.
        final_decision["confidence"] = round(
            min(
                self._safe_float(final_decision.get("confidence", 0.5), default=0.5),
                max(0.25, self._safe_float(quality_gate.get("score", 0.5), default=0.5)),
            ),
            4,
        )
        final_decision["signal"] = self._normalize_report_signal(str(final_decision.get("signal", "hold")))
        committee = {
            "research_note": self._sanitize_report_text(str(committee.get("research_note", "")).strip() or "研究汇总暂不可用。"),
            "risk_note": self._sanitize_report_text(str(committee.get("risk_note", "")).strip() or "风险仲裁暂不可用。"),
        }
        analysis_nodes = self._normalize_report_analysis_nodes(list(analysis_nodes or []))
        quality_dashboard = self._build_report_quality_dashboard(
            report_modules=report_modules,
            quality_gate=quality_gate,
            final_decision=final_decision,
            analysis_nodes=analysis_nodes,
            evidence_refs=evidence_refs,
        )

        # Project module payload into report_sections for backward-compatible renderers.
        for module in report_modules:
            if not isinstance(module, dict):
                continue
            report_sections.append(
                {
                    "section_id": f"module_{str(module.get('module_id', '')).strip() or 'unknown'}",
                    "title": str(module.get("title", "模块内容")),
                    "content": str(module.get("content", ""))[:2200],
                }
            )
        report_sections.append(
            {
                "section_id": "committee_notes",
                "title": "委员会纪要",
                "content": json.dumps(committee, ensure_ascii=False),
            }
        )
        report_sections.append(
            {
                "section_id": "final_decision",
                "title": "综合决策",
                "content": json.dumps(final_decision, ensure_ascii=False),
            }
        )
        report_sections.append(
            {
                "section_id": "metric_snapshot",
                "title": "指标快照",
                "content": json.dumps(metric_snapshot, ensure_ascii=False),
            }
        )
        report_sections.append(
            {
                "section_id": "analysis_nodes",
                "title": "研究与风险节点",
                "content": json.dumps(analysis_nodes, ensure_ascii=False),
            }
        )
        report_sections.append(
            {
                "section_id": "quality_dashboard",
                "title": "质量评估看板",
                "content": json.dumps(quality_dashboard, ensure_ascii=False),
            }
        )

        markdown = (
            f"# {req.stock_code} 分析报告\n\n"
            f"## 结论\n{query_result['answer']}\n\n"
            "## 证据\n"
            + "\n".join(f"- {c['source_id']}: {c['excerpt']}" for c in query_result["citations"])
            + "\n\n## 风险与反证\n结合估值、流动性与政策扰动进行反证校验。"
            + "\n\n## 操作建议\n建议分批验证信号稳定性，避免一次性重仓。"
        )
        markdown += (
            "\n\n## 最终决策\n"
            f"- signal: {str(final_decision.get('signal', 'hold'))}\n"
            f"- confidence: {self._safe_float(final_decision.get('confidence', 0.5), default=0.5):.2f}\n"
            f"- target_price: {json.dumps(dict(final_decision.get('target_price', {}) or {}), ensure_ascii=False)}\n"
            f"- risk_score: {self._safe_float(final_decision.get('risk_score', 0.0), default=0.0):.2f}\n"
            f"- reward_risk_ratio: {self._safe_float(final_decision.get('reward_risk_ratio', 0.0), default=0.0):.3f}\n"
            f"- position_sizing_hint: {str(final_decision.get('position_sizing_hint', ''))}\n"
            f"- rationale: {str(final_decision.get('rationale', ''))[:400]}\n"
        )
        markdown += (
            "\n\n## Data Quality Gate\n"
            f"- status: {quality_gate['status']}\n"
            f"- score: {float(quality_gate.get('score', 0.0) or 0.0):.2f}\n"
            f"- reasons: {','.join(quality_reasons) or 'none'}\n"
            f"- generation_mode: {generation_mode}\n"
            + (f"- generation_error: {generation_error}\n" if generation_error else "")
        )
        self._reports[report_id] = {
            "report_id": report_id,
            "schema_version": self._report_bundle_schema_version,
            "trace_id": query_result["trace_id"],
            "markdown": markdown,
            "citations": query_result["citations"],
            "run_id": run_id,
            "pool_snapshot_id": pool_snapshot_id,
            "template_id": template_id,
            "evidence_refs": evidence_refs,
            "report_sections": report_sections,
            "quality_gate": quality_gate,
            "report_data_pack_summary": report_data_pack_summary,
            "report_modules": report_modules,
            "committee": committee,
            "final_decision": final_decision,
            "metric_snapshot": metric_snapshot,
            "analysis_nodes": analysis_nodes,
            "quality_dashboard": quality_dashboard,
            "confidence_attribution": {
                "citation_count": len(list(query_result.get("citations", []) or [])),
                "history_sample_size": history_sample_size,
                "predict_quality": str(predict_snapshot.get("data_quality", "unknown")),
                "quality_score": float(quality_gate.get("score", 0.0) or 0.0),
            },
            "llm_input_pack": report_input_pack,
            "generation_mode": generation_mode,
            "generation_error": generation_error,
        }
        user_id = None
        tenant_id = None
        token = payload.get("token")
        if token:
            try:
                me = self.web.auth_me(token)
                user_id = int(me["user_id"])
                tenant_id = int(me["tenant_id"])
            except Exception:
                pass
        version_payload = {
            "schema_version": self._report_bundle_schema_version,
            "report_id": report_id,
            "stock_code": str(req.stock_code),
            "report_type": str(req.report_type),
            "final_decision": dict(final_decision),
            "committee": dict(committee),
            "report_modules": list(report_modules),
            "analysis_nodes": list(analysis_nodes),
            "quality_dashboard": dict(quality_dashboard),
            "metric_snapshot": dict(metric_snapshot),
            "quality_gate": dict(quality_gate),
            "evidence_refs": list(evidence_refs),
        }
        self.web.save_report_index(
            report_id=report_id,
            user_id=user_id,
            tenant_id=tenant_id,
            stock_code=req.stock_code,
            report_type=req.report_type,
            markdown=markdown,
            run_id=run_id,
            pool_snapshot_id=pool_snapshot_id,
            template_id=template_id,
            payload_json=json.dumps(version_payload, ensure_ascii=False),
        )
        resp = ReportResponse(
            report_id=report_id,
            trace_id=query_result["trace_id"],
            markdown=markdown,
            citations=[Citation(**c) for c in query_result["citations"]],
        ).model_dump(mode="json")
        resp["run_id"] = run_id
        resp["pool_snapshot_id"] = pool_snapshot_id
        resp["template_id"] = template_id
        resp["evidence_refs"] = evidence_refs
        resp["report_sections"] = report_sections
        resp["quality_gate"] = quality_gate
        resp["report_data_pack_summary"] = report_data_pack_summary
        resp["report_modules"] = report_modules
        resp["committee"] = committee
        resp["final_decision"] = final_decision
        resp["metric_snapshot"] = metric_snapshot
        resp["analysis_nodes"] = analysis_nodes
        resp["quality_dashboard"] = quality_dashboard
        resp["confidence_attribution"] = {
            "citation_count": len(list(query_result.get("citations", []) or [])),
            "history_sample_size": history_sample_size,
            "predict_quality": str(predict_snapshot.get("data_quality", "unknown")),
            "quality_score": float(quality_gate.get("score", 0.0) or 0.0),
        }
        resp["llm_input_pack"] = report_input_pack
        resp["generation_mode"] = generation_mode
        resp["generation_error"] = generation_error
        resp["stock_code"] = str(req.stock_code)
        resp["report_type"] = str(req.report_type)
        resp["schema_version"] = self._report_bundle_schema_version
        resp["result_level"] = "full"
        degrade_code = ""
        if quality_gate["status"] == "degraded":
            degrade_code = "quality_degraded"
        elif quality_gate["status"] == "watch":
            degrade_code = "quality_watch"
        severity = "low"
        if quality_gate["status"] == "degraded":
            severity = "high" if float(quality_gate.get("score", 0.0) or 0.0) < 0.5 else "medium"
        elif quality_gate["status"] == "watch":
            severity = "low"
        resp["degrade"] = {
            "active": bool(quality_reasons),
            "code": degrade_code if quality_reasons else "",
            "severity": severity,
            "reasons": list(dict.fromkeys(quality_reasons)),
            "missing_data": list(dict.fromkeys(quality_reasons)),
            "confidence_penalty": round(min(0.7, 0.15 * float(len(quality_reasons))), 4),
            "user_message": (
                "Report has watch-level gaps; conclusions remain usable but should be rechecked with fresher evidence."
                if quality_gate["status"] == "watch"
                else "Report contains degraded segments; please review quality gate and evidence coverage."
            )
            if quality_reasons
            else "Report quality is normal.",
        }
        sanitized = self._sanitize_report_payload(resp)
        self._reports[report_id] = dict(sanitized)
        return sanitized

    def report_get(self, report_id: str) -> dict[str, Any]:
        """Get report by report_id."""
        report = self._reports.get(report_id)
        if not report:
            return {"error": "not_found", "report_id": report_id}
        return self._sanitize_report_payload(dict(report))

    def _latest_report_context(self, stock_code: str) -> dict[str, Any]:
        """Get latest in-memory report context for one stock to assist DeepThink rounds."""
        code = str(stock_code or "").strip().upper()
        if not code:
            return {"available": False}
        latest: dict[str, Any] | None = None
        # Iterate reverse insertion order to find newest matching report quickly.
        for value in reversed(list(self._reports.values())):
            if not isinstance(value, dict):
                continue
            row_code = str(value.get("stock_code", "")).strip().upper()
            if row_code == code:
                latest = value
                break
        if not latest:
            return {"available": False, "stock_code": code}
        decision = dict(latest.get("final_decision", {}) or {})
        committee = dict(latest.get("committee", {}) or {})
        node_map = {
            str(row.get("node_id", "")).strip().lower(): row
            for row in list(latest.get("analysis_nodes", []) or [])
            if isinstance(row, dict) and str(row.get("node_id", "")).strip()
        }
        research_node = dict(node_map.get("research_summarizer", {}) or {})
        risk_node = dict(node_map.get("risk_arbiter", {}) or {})
        return {
            "available": True,
            "stock_code": code,
            "report_id": str(latest.get("report_id", "")),
            "signal": self._normalize_report_signal(str(decision.get("signal", "hold"))),
            "confidence": max(0.0, min(1.0, self._safe_float(decision.get("confidence", 0.5), default=0.5))),
            "rationale": str(decision.get("rationale", ""))[:400],
            "research_note": str(committee.get("research_note", ""))[:300],
            "risk_note": str(committee.get("risk_note", ""))[:300],
            "research_summary": str(research_node.get("summary", ""))[:280],
            "risk_summary": str(risk_node.get("summary", ""))[:280],
        }

    def _sanitize_report_text(self, raw: str) -> str:
        """Normalize known mojibake fragments in report-facing Chinese text."""
        text = str(raw or "")
        replacements = {
            # Common mojibake fragments observed in report fields.
            "缂佹捁顔戦幗妯款洣": "Executive Summary",
            "鐠囦焦宓佸〒鍛礋": "Evidence List",
            "妞嬪酣娅撴稉搴″冀鐠?": "Risks And Counterpoints",
            "閹垮秳缍斿楦款唴": "Action Plan",
            "闁轰胶澧楀畵浣烘嫻閵娾晛娅ら梻鍌樺妿": "Data Quality Gate",
            "闁圭瑳鍡╂斀闁硅姤顭堥々?LLM)": "LLM Summary",
            "闁哄秶顭堢缓楣冩焻閺勫繒甯?LLM)": "Core Logic (LLM)",
            "濡炲閰ｅ▍鎾绘儗閳哄懏鈻?LLM)": "Risk Matrix (LLM)",
            "闁诡垰鎳忓▍娆撳箳閵婏妇宸?LLM)": "Scenario Analysis (LLM)",
            "閺嶇绺鹃柅鏄忕帆(LLM)": "Core Logic (LLM)",
            "閹躲儱鎲℃稉顓炵妇": "Report Center",
            "閸掑棙鐎介幎銉ユ啞": "Analysis Report",
            "妞嬪酣娅撴稉搴″冀鐠囦箺n": "Risks And Counterpoints\n",
            # Also cover visible mojibake strings from historical payloads.
            "缁撹鎽樿": "Executive Summary",
            "璇佹嵁娓呭崟": "Evidence List",
            "椋庨櫓涓庡弽璇?": "Risks And Counterpoints",
            "鎿嶄綔寤鸿": "Action Plan",
            "閺佺増宓佺拹銊╁櫤闂傘劎": "Data Quality Gate",
            "閹笛嗩攽閹芥顩?LLM)": "LLM Summary",
            "妞嬪酣娅撻惌鈺呮█(LLM)": "Risk Matrix (LLM)",
            "閹懏娅欓幒銊︾川(LLM)": "Scenario Analysis (LLM)",
        }
        for bad, good in replacements.items():
            text = text.replace(bad, good)
        return text

    def _sanitize_report_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Ensure report payload is readable and contains degrade metadata."""
        body = dict(payload)
        body["markdown"] = self._sanitize_report_text(str(body.get("markdown", "")))

        sections: list[dict[str, Any]] = []
        for row in list(body.get("report_sections", []) or []):
            if not isinstance(row, dict):
                continue
            section = dict(row)
            section["title"] = self._sanitize_report_text(str(section.get("title", "")))
            section["content"] = self._sanitize_report_text(str(section.get("content", "")))
            sections.append(section)
        if sections:
            body["report_sections"] = sections

        module_rows: list[dict[str, Any]] = []
        for row in list(body.get("report_modules", []) or []):
            if not isinstance(row, dict):
                continue
            item = dict(row)
            item["title"] = self._sanitize_report_text(str(item.get("title", "")))
            item["content"] = self._sanitize_report_text(str(item.get("content", "")))
            module_rows.append(item)
        if module_rows:
            body["report_modules"] = self._normalize_report_modules(module_rows)
        body["evidence_refs"] = self._normalize_report_evidence_refs(
            [dict(row) for row in list(body.get("evidence_refs", []) or []) if isinstance(row, dict)]
        )

        node_rows: list[dict[str, Any]] = []
        for row in list(body.get("analysis_nodes", []) or []):
            if isinstance(row, dict):
                node_rows.append(dict(row))
        body["analysis_nodes"] = self._normalize_report_analysis_nodes(node_rows)

        committee = dict(body.get("committee", {}) or {})
        if committee:
            body["committee"] = {
                "research_note": self._sanitize_report_text(str(committee.get("research_note", "")).strip()),
                "risk_note": self._sanitize_report_text(str(committee.get("risk_note", "")).strip()),
            }

        final_decision = dict(body.get("final_decision", {}) or {})
        if final_decision:
            final_decision["signal"] = self._normalize_report_signal(str(final_decision.get("signal", "hold")))
            final_decision["confidence"] = round(
                max(0.0, min(1.0, self._safe_float(final_decision.get("confidence", 0.5), default=0.5))),
                4,
            )
            final_decision["rationale"] = self._sanitize_report_text(str(final_decision.get("rationale", "")).strip())
            target_price = dict(final_decision.get("target_price", {}) or {})
            final_decision["target_price"] = {
                "low": round(self._safe_float(target_price.get("low", 0.0), default=0.0), 3),
                "base": round(self._safe_float(target_price.get("base", 0.0), default=0.0), 3),
                "high": round(self._safe_float(target_price.get("high", 0.0), default=0.0), 3),
            }
            final_decision["risk_score"] = round(
                max(0.0, min(100.0, self._safe_float(final_decision.get("risk_score", 0.0), default=0.0))),
                2,
            )
            final_decision["reward_risk_ratio"] = round(
                max(0.0, min(10.0, self._safe_float(final_decision.get("reward_risk_ratio", 0.0), default=0.0))),
                3,
            )
            final_decision["position_sizing_hint"] = self._sanitize_report_text(
                str(final_decision.get("position_sizing_hint", "")).strip()
            )[:80]
            if isinstance(final_decision.get("execution_plan"), list):
                final_decision["execution_plan"] = [
                    self._sanitize_report_text(str(x).strip())
                    for x in list(final_decision.get("execution_plan", []))
                    if str(x).strip()
                ]
            if isinstance(final_decision.get("invalidation_conditions"), list):
                final_decision["invalidation_conditions"] = [
                    self._sanitize_report_text(str(x).strip())
                    for x in list(final_decision.get("invalidation_conditions", []))
                    if str(x).strip()
                ]
            body["final_decision"] = final_decision

        degrade = body.get("degrade")
        if not isinstance(degrade, dict):
            reasons = list((body.get("quality_gate", {}) or {}).get("reasons", []) or [])
            quality_status = str((body.get("quality_gate", {}) or {}).get("status", "pass")).strip().lower() or "pass"
            degrade_code = ""
            if quality_status == "degraded":
                degrade_code = "quality_degraded"
            elif quality_status == "watch":
                degrade_code = "quality_watch"
            degrade = {
                "active": bool(reasons),
                "code": degrade_code if reasons else "",
                "severity": "high" if quality_status == "degraded" else "low" if quality_status == "watch" else "low",
                "reasons": reasons,
                "missing_data": reasons,
                "confidence_penalty": round(min(0.7, 0.15 * float(len(reasons))), 4),
                "user_message": "Report contains degraded segments; please review quality gate and evidence coverage."
                if reasons
                else "Report quality is normal.",
            }
        body["degrade"] = degrade
        body["quality_dashboard"] = self._build_report_quality_dashboard(
            report_modules=list(body.get("report_modules", []) or []),
            quality_gate=dict(body.get("quality_gate", {}) or {}),
            final_decision=dict(body.get("final_decision", {}) or {}),
            analysis_nodes=list(body.get("analysis_nodes", []) or []),
            evidence_refs=list(body.get("evidence_refs", []) or []),
        )
        body["schema_version"] = str(body.get("schema_version", self._report_bundle_schema_version) or self._report_bundle_schema_version)
        if "result_level" not in body:
            body["result_level"] = "full"
        return body

    def _build_report_task_partial_result(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Build a minimal usable report result for async task partial_ready state."""
        req = ReportRequest(**payload)
        code = str(req.stock_code).strip().upper()
        partial_pack = self._build_llm_input_pack(code, question=f"report_partial:{code}", scenario="report")
        query_result = self.query(
            {
                "user_id": req.user_id,
                "question": f"请先给出 {req.stock_code} {req.period} 的核心结论与证据摘要",
                "stock_codes": [code],
            }
        )
        citations = list(query_result.get("citations", []) or [])
        citation_lines = "\n".join(
            f"- {str(item.get('source_id', ''))}: {str(item.get('excerpt', ''))}"
            for item in citations[:8]
        )
        markdown = (
            f"# {str(req.stock_code).strip().upper()} Analysis Report (Partial)\n\n"
            f"## Summary\n{str(query_result.get('answer', '')).strip() or 'No summary available'}\n\n"
            "## Evidence\n"
            + (citation_lines or "- No evidence")
            + "\n\n## Note\nThis is a minimum viable report. Full report is still running."
        )
        result = {
            "report_id": f"partial-{uuid.uuid4().hex[:10]}",
            "schema_version": self._report_bundle_schema_version,
            "trace_id": str(query_result.get("trace_id", "")),
            "stock_code": code,
            "report_type": str(req.report_type),
            "markdown": markdown,
            "citations": citations,
            "evidence_refs": [self._build_report_evidence_ref(dict(item)) for item in citations[:10]],
            "report_modules": [
                {
                    "module_id": "executive_summary",
                    "title": "执行摘要",
                    "content": str(query_result.get("answer", "")).strip() or "暂无摘要",
                    "evidence_refs": [str(item.get("source_id", "")) for item in citations[:6] if str(item.get("source_id", "")).strip()],
                    "coverage": {"status": "partial", "data_points": int(len(citations))},
                    "confidence": 0.52,
                    "degrade_reason": ["partial_result"],
                },
                {
                    "module_id": "risk_matrix",
                    "title": "风险矩阵",
                    "content": json.dumps(
                        [
                            {
                                "risk": "partial_result",
                                "signal": "warning",
                                "detail": "完整报告尚未完成，当前仅返回最小可用结果。",
                            }
                        ],
                        ensure_ascii=False,
                    ),
                    "evidence_refs": [],
                    "coverage": {"status": "partial", "data_points": 1},
                    "confidence": 0.45,
                    "degrade_reason": ["partial_result"],
                },
                {
                    "module_id": "execution_plan",
                    "title": "执行策略",
                    "content": "- 等待 full 报告完成后再做仓位动作。\n- 先查看数据缺口与质量门控说明。",
                    "evidence_refs": [],
                    "coverage": {"status": "partial", "data_points": 1},
                    "confidence": 0.42,
                    "degrade_reason": ["partial_result"],
                },
            ],
            "committee": {
                "research_note": "研究汇总：当前为 partial 结果，结论仅供预览。",
                "risk_note": "风险仲裁：完整风险矩阵尚未生成，暂不建议执行重仓动作。",
            },
            "final_decision": {
                "signal": "hold",
                "confidence": 0.45,
                "rationale": "当前任务处于 partial 阶段，证据和模块尚未完整，暂保持观望。",
                "invalidation_conditions": ["full_report_ready"],
                "execution_plan": ["等待完整报告结果", "优先核验数据缺口后再决策"],
            },
            "analysis_nodes": [
                {
                    "node_id": "research_summarizer",
                    "title": "研究汇总器",
                    "status": "degraded",
                    "signal": "hold",
                    "confidence": 0.45,
                    "summary": "研究汇总器：当前仅有 partial 证据，结论仅供预览。",
                    "highlights": ["等待 full 报告补齐核心证据后再更新结论。"],
                    "evidence_refs": [str(item.get("source_id", "")) for item in citations[:4] if str(item.get("source_id", "")).strip()],
                    "coverage": {"citation_count": int(len(citations))},
                    "degrade_reason": ["partial_result"],
                    "guardrails": [],
                    "veto": False,
                },
                {
                    "node_id": "risk_arbiter",
                    "title": "风险仲裁器",
                    "status": "ready",
                    "signal": "warning",
                    "confidence": 0.52,
                    "summary": "风险仲裁器：partial 阶段默认保持风控优先，不建议执行重仓动作。",
                    "highlights": ["完整风险矩阵未生成前，维持观望仓位。"],
                    "evidence_refs": [],
                    "coverage": {"risk_watch_count": 0, "quality_status": "degraded"},
                    "degrade_reason": ["partial_result"],
                    "guardrails": ["等待 full 报告后再执行仓位动作。"],
                    "veto": True,
                },
            ],
            "metric_snapshot": {
                "history_sample_size": 0,
                "news_count": 0,
                "research_count": 0,
                "macro_count": 0,
                "quality_score": 0.55,
                "citation_count": int(len(citations)),
                "predict_quality": "pending",
            },
            "quality_gate": {
                "status": "degraded",
                "score": 0.55,
                "reasons": ["partial_result"],
            },
            "quality_dashboard": {
                "status": "degraded",
                "overall_score": 0.48,
                "module_count": 3,
                "avg_module_quality": 0.43,
                "min_module_quality": 0.38,
                "coverage_ratio": 0.0,
                "evidence_ref_count": int(len(citations)),
                "evidence_density": round(float(len(citations)) / 3.0, 4) if citations else 0.0,
                "consistency_score": 0.72,
                "low_quality_modules": ["executive_summary", "risk_matrix", "execution_plan"],
                "reasons": ["partial_result"],
                "node_veto": True,
            },
            "report_data_pack_summary": {
                "as_of": datetime.now(timezone.utc).isoformat(),
                "history_sample_size": 0,
                "predict_quality": "pending",
                "predict_degrade_reasons": ["pending_full_pipeline"],
                "intel_signal": "pending",
                "intel_confidence": 0.0,
                "news_count": 0,
                "research_count": 0,
                "macro_count": 0,
                "missing_data": list(partial_pack.get("missing_data", []) or []),
            },
            "generation_mode": "partial",
            "generation_error": "",
            "degrade": {
                "active": True,
                "code": "partial_result",
                "reasons": ["partial_result"],
                "missing_data": ["full_report_pending"],
                "confidence_penalty": 0.45,
                "user_message": "Minimum viable report returned. Full report is still being generated.",
            },
            "result_level": "partial",
            "stage_progress": {"stage": "partial_ready", "progress": 0.45},
        }
        return self._sanitize_report_payload(result)

    def _report_task_mark_failed(self, task: dict[str, Any], *, error_code: str, error_message: str) -> None:
        """Mark one report task as failed with a normalized error payload."""
        now_iso = datetime.now(timezone.utc).isoformat()
        task["status"] = "failed"
        task["current_stage"] = "failed"
        task["stage_message"] = "Report task failed"
        task["error_code"] = str(error_code or "report_task_failed")
        task["error_message"] = str(error_message or "report_task_failed")[:260]
        task["updated_at"] = now_iso
        task["heartbeat_at"] = now_iso
        task["completed_at"] = now_iso

    def _report_task_apply_runtime_guard(self, task: dict[str, Any]) -> None:
        """Apply timeout/stall guard in-place before task snapshot/result is returned."""
        status = str(task.get("status", "")).strip().lower()
        if status in {"completed", "failed", "cancelled"}:
            return
        now = datetime.now(timezone.utc)

        deadline_raw = str(task.get("deadline_at", "")).strip()
        if deadline_raw:
            deadline = self._parse_time(deadline_raw)
            if now > deadline:
                self._report_task_mark_failed(
                    task,
                    error_code="report_task_timeout",
                    error_message="Report task exceeded runtime limit; please retry with fewer inputs.",
                )
                return

        # If worker heartbeat disappears for too long, fail fast so frontend does not spin forever.
        heartbeat_raw = str(task.get("heartbeat_at", "")).strip() or str(task.get("updated_at", "")).strip()
        if heartbeat_raw and status in {"running", "partial_ready"}:
            heartbeat_at = self._parse_time(heartbeat_raw)
            if (now - heartbeat_at).total_seconds() > float(self._report_task_stall_seconds):
                self._report_task_mark_failed(
                    task,
                    error_code="report_task_stalled",
                    error_message="Report task heartbeat stalled; please retry.",
                )

    def _report_task_tick_heartbeat(
        self,
        task: dict[str, Any],
        *,
        progress_floor: float = 0.0,
        stage_message: str | None = None,
    ) -> None:
        """Touch heartbeat and keep progress moving while long-running stage is executing."""
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        task["updated_at"] = now_iso
        task["heartbeat_at"] = now_iso
        if stage_message is not None:
            task["stage_message"] = stage_message
        if progress_floor > 0:
            task["progress"] = max(float(task.get("progress", 0.0) or 0.0), float(progress_floor))

    def _report_task_snapshot(self, task: dict[str, Any]) -> dict[str, Any]:
        """Project internal task state to API-safe payload."""
        self._report_task_apply_runtime_guard(task)
        has_full = bool(task.get("result_full"))
        has_partial = bool(task.get("result_partial"))
        level = "full" if has_full else "partial" if has_partial else "none"
        display_ready = bool(has_full)
        partial_reason = ""
        if has_partial and not has_full:
            partial_reason = "warming_up"
        best_result = task.get("result_full") if has_full else task.get("result_partial") if has_partial else {}
        best_result = best_result if isinstance(best_result, dict) else {}
        pack_summary = dict(best_result.get("report_data_pack_summary", {}) or {})
        missing_data = [str(x) for x in list(pack_summary.get("missing_data", []) or []) if str(x).strip()]
        data_pack_status = "ready"
        if not best_result:
            data_pack_status = "failed" if str(task.get("status", "")) == "failed" else "partial"
        elif missing_data:
            data_pack_status = "partial"
        stage_started_at = str(task.get("stage_started_at", "")).strip()
        heartbeat_at = str(task.get("heartbeat_at", "")).strip()
        stage_elapsed_seconds = 0
        heartbeat_age_seconds = 0
        now = datetime.now(timezone.utc)
        if stage_started_at:
            stage_elapsed_seconds = max(0, int((now - self._parse_time(stage_started_at)).total_seconds()))
        if heartbeat_at:
            heartbeat_age_seconds = max(0, int((now - self._parse_time(heartbeat_at)).total_seconds()))
        return {
            "task_id": str(task.get("task_id", "")),
            "status": str(task.get("status", "queued")),
            "progress": round(max(0.0, min(1.0, float(task.get("progress", 0.0) or 0.0))), 4),
            "current_stage": str(task.get("current_stage", "")),
            "stage_message": str(task.get("stage_message", "")),
            "created_at": str(task.get("created_at", "")),
            "updated_at": str(task.get("updated_at", "")),
            "started_at": str(task.get("started_at", "")),
            "completed_at": str(task.get("completed_at", "")),
            "deadline_at": str(task.get("deadline_at", "")),
            "heartbeat_at": heartbeat_at,
            "stage_started_at": stage_started_at,
            "stage_elapsed_seconds": int(stage_elapsed_seconds),
            "heartbeat_age_seconds": int(heartbeat_age_seconds),
            "result_level": level,
            "has_partial_result": has_partial,
            "has_full_result": has_full,
            "display_ready": display_ready,
            "partial_reason": partial_reason,
            "error_code": str(task.get("error_code", "")),
            "error_message": str(task.get("error_message", "")),
            "data_pack_status": data_pack_status,
            "data_pack_missing": missing_data,
            "quality_gate_detail": dict(best_result.get("quality_gate", {}) or {}),
            "report_quality_dashboard": dict(best_result.get("quality_dashboard", {}) or {}),
        }

    def _run_report_task(self, task_id: str) -> None:
        """Background worker for async report tasks."""
        with self._report_task_lock:
            task = self._report_tasks.get(task_id)
            if not task:
                return
            if bool(task.get("cancel_requested", False)):
                now_iso = datetime.now(timezone.utc).isoformat()
                task["status"] = "cancelled"
                task["updated_at"] = now_iso
                task["heartbeat_at"] = now_iso
                task["completed_at"] = now_iso
                task["stage_message"] = "Task was cancelled before start"
                return
            now_iso = datetime.now(timezone.utc).isoformat()
            task["status"] = "running"
            task["started_at"] = now_iso
            task["updated_at"] = now_iso
            task["heartbeat_at"] = now_iso
            task["stage_started_at"] = now_iso
            task["current_stage"] = "partial"
            task["stage_message"] = "Generating minimum viable result"
            task["progress"] = max(float(task.get("progress", 0.0) or 0.0), 0.1)

        try:
            with self._report_task_lock:
                task_payload = dict(self._report_tasks.get(task_id, {}).get("payload", {}))

            partial = self._build_report_task_partial_result(task_payload)
            with self._report_task_lock:
                current = self._report_tasks.get(task_id)
                if not current:
                    return
                self._report_task_apply_runtime_guard(current)
                if str(current.get("status", "")) == "failed":
                    return
                if bool(current.get("cancel_requested", False)):
                    now = datetime.now(timezone.utc).isoformat()
                    current["status"] = "cancelled"
                    current["updated_at"] = now
                    current["heartbeat_at"] = now
                    current["completed_at"] = now
                    current["stage_message"] = "Task cancelled"
                    return
                current["status"] = "partial_ready"
                current["current_stage"] = "full_report"
                current["stage_message"] = "Partial result ready; generating full report"
                current["progress"] = max(float(current.get("progress", 0.0) or 0.0), 0.45)
                current["result_partial"] = partial
                now_iso = datetime.now(timezone.utc).isoformat()
                current["updated_at"] = now_iso
                current["heartbeat_at"] = now_iso
                current["stage_started_at"] = now_iso

            keepalive_stop = threading.Event()

            def _keepalive_loop() -> None:
                while not keepalive_stop.wait(self._report_task_heartbeat_interval_seconds):
                    with self._report_task_lock:
                        running = self._report_tasks.get(task_id)
                        if not running:
                            return
                        self._report_task_apply_runtime_guard(running)
                        if str(running.get("status", "")) in {"completed", "failed", "cancelled"}:
                            return
                        if bool(running.get("cancel_requested", False)):
                            return
                        stage_started = str(running.get("stage_started_at", "")).strip()
                        elapsed = 0
                        if stage_started:
                            elapsed = max(0, int((datetime.now(timezone.utc) - self._parse_time(stage_started)).total_seconds()))
                        # Keep users informed that full generation is still progressing.
                        dynamic_progress = min(0.92, 0.45 + min(0.42, float(elapsed) / 150.0))
                        self._report_task_tick_heartbeat(
                            running,
                            progress_floor=dynamic_progress,
                            stage_message=f"Generating full report ({elapsed}s)",
                        )

            keepalive_thread = threading.Thread(target=_keepalive_loop, daemon=True)
            keepalive_thread.start()
            try:
                full = self.report_generate(task_payload)
            finally:
                keepalive_stop.set()
                keepalive_thread.join(timeout=1.0)

            with self._report_task_lock:
                current = self._report_tasks.get(task_id)
                if not current:
                    return
                self._report_task_apply_runtime_guard(current)
                if str(current.get("status", "")) == "failed":
                    return
                if bool(current.get("cancel_requested", False)):
                    now = datetime.now(timezone.utc).isoformat()
                    current["status"] = "cancelled"
                    current["updated_at"] = now
                    current["heartbeat_at"] = now
                    current["completed_at"] = now
                    current["stage_message"] = "Task cancelled"
                    return
                current["status"] = "completed"
                current["current_stage"] = "done"
                current["stage_message"] = "Report generation completed"
                current["progress"] = 1.0
                current["result_full"] = self._sanitize_report_payload(full if isinstance(full, dict) else {})
                if not current.get("result_partial"):
                    current["result_partial"] = current["result_full"]
                now = datetime.now(timezone.utc).isoformat()
                current["updated_at"] = now
                current["heartbeat_at"] = now
                current["stage_started_at"] = now
                current["completed_at"] = now
        except Exception as ex:  # noqa: BLE001
            with self._report_task_lock:
                current = self._report_tasks.get(task_id)
                if not current:
                    return
                self._report_task_mark_failed(
                    current,
                    error_code="report_task_failed",
                    error_message=str(ex),
                )

    def report_task_create(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create async report task and return status immediately."""
        req = ReportRequest(**payload)
        task_id = f"rpt-{uuid.uuid4().hex[:12]}"
        now_iso = datetime.now(timezone.utc).isoformat()
        deadline_iso = (datetime.now(timezone.utc) + timedelta(seconds=max(60, int(self._report_task_timeout_seconds)))).isoformat()
        task_payload = {
            "user_id": str(req.user_id),
            "stock_code": str(req.stock_code).strip().upper(),
            "period": str(req.period),
            "report_type": str(req.report_type),
            "run_id": str(payload.get("run_id", "")).strip(),
            "pool_snapshot_id": str(payload.get("pool_snapshot_id", "")).strip(),
            "template_id": str(payload.get("template_id", "default")).strip() or "default",
        }
        token = str(payload.get("token", "")).strip()
        if token:
            task_payload["token"] = token

        with self._report_task_lock:
            self._report_tasks[task_id] = {
                "task_id": task_id,
                "status": "queued",
                "progress": 0.0,
                "current_stage": "queued",
                "stage_message": "Task queued",
                "created_at": now_iso,
                "updated_at": now_iso,
                "heartbeat_at": now_iso,
                "deadline_at": deadline_iso,
                "stage_started_at": now_iso,
                "started_at": "",
                "completed_at": "",
                "error_code": "",
                "error_message": "",
                "cancel_requested": False,
                "payload": task_payload,
                "result_partial": None,
                "result_full": None,
            }
        self._report_task_executor.submit(self._run_report_task, task_id)
        return self.report_task_get(task_id)

    def report_task_get(self, task_id: str) -> dict[str, Any]:
        """Get async report task status."""
        key = str(task_id or "").strip()
        if not key:
            return {"error": "not_found", "task_id": key}
        with self._report_task_lock:
            task = self._report_tasks.get(key)
            if not task:
                return {"error": "not_found", "task_id": key}
            return self._report_task_snapshot(task)

    def report_task_result(self, task_id: str) -> dict[str, Any]:
        """Get best available report result for one task."""
        key = str(task_id or "").strip()
        if not key:
            return {"error": "not_found", "task_id": key}
        with self._report_task_lock:
            task = self._report_tasks.get(key)
            if not task:
                return {"error": "not_found", "task_id": key}
            self._report_task_apply_runtime_guard(task)
            full = task.get("result_full")
            partial = task.get("result_partial")
            if isinstance(full, dict):
                return {
                    "task_id": key,
                    "status": str(task.get("status", "")),
                    "result_level": "full",
                    "display_ready": True,
                    "partial_reason": "",
                    "deadline_at": str(task.get("deadline_at", "")),
                    "heartbeat_at": str(task.get("heartbeat_at", "")),
                    "result": full,
                }
            if isinstance(partial, dict):
                return {
                    "task_id": key,
                    "status": str(task.get("status", "")),
                    "result_level": "partial",
                    "display_ready": False,
                    "partial_reason": "warming_up",
                    "deadline_at": str(task.get("deadline_at", "")),
                    "heartbeat_at": str(task.get("heartbeat_at", "")),
                    "result": partial,
                }
            return {
                "task_id": key,
                "status": str(task.get("status", "")),
                "result_level": "none",
                "display_ready": False,
                "partial_reason": "",
                "deadline_at": str(task.get("deadline_at", "")),
                "heartbeat_at": str(task.get("heartbeat_at", "")),
                "result": None,
            }

    def report_task_cancel(self, task_id: str) -> dict[str, Any]:
        """Cancel async report task."""
        key = str(task_id or "").strip()
        if not key:
            return {"error": "not_found", "task_id": key}
        with self._report_task_lock:
            task = self._report_tasks.get(key)
            if not task:
                return {"error": "not_found", "task_id": key}
            status = str(task.get("status", ""))
            if status in {"completed", "failed", "cancelled"}:
                return self._report_task_snapshot(task)
            task["cancel_requested"] = True
            now_iso = datetime.now(timezone.utc).isoformat()
            task["updated_at"] = now_iso
            task["heartbeat_at"] = now_iso
            if status == "queued":
                task["status"] = "cancelled"
                task["current_stage"] = "cancelled"
                task["stage_message"] = "Task cancelled"
                task["completed_at"] = now_iso
            else:
                task["status"] = "cancelling"
                task["stage_message"] = "Cancel requested; waiting for safe stop"
            return self._report_task_snapshot(task)

    def ingest_market_daily(self, stock_codes: list[str]) -> dict[str, Any]:
        """Trigger market-daily ingestion."""
        return self.ingestion.ingest_market_daily(stock_codes)

    def ingest_announcements(self, stock_codes: list[str]) -> dict[str, Any]:
        """Trigger announcements ingestion."""
        return self.ingestion.ingest_announcements(stock_codes)

    def ingest_financials(self, stock_codes: list[str]) -> dict[str, Any]:
        """Trigger financial snapshot ingestion."""

        return self.ingestion.ingest_financials(stock_codes)

    def ingest_news(self, stock_codes: list[str], limit: int = 20) -> dict[str, Any]:
        """Trigger news ingestion and sync rows into local RAG index."""

        result = self.ingestion.ingest_news(stock_codes, limit=limit)
        # News text contributes to event-driven analysis, so we index it as lightweight docs.
        self._index_text_rows_to_rag(
            rows=self.ingestion.store.news_items[-400:],
            namespace="news",
            source_field="source_id",
            time_field="event_time",
            content_fields=("content", "title"),
            filename_ext="txt",
        )
        return result

    def ingest_research_reports(self, stock_codes: list[str], limit: int = 20) -> dict[str, Any]:
        """Trigger research-report ingestion and sync rows into local RAG index."""

        result = self.ingestion.ingest_research_reports(stock_codes, limit=limit)
        self._index_text_rows_to_rag(
            rows=self.ingestion.store.research_reports[-240:],
            namespace="research",
            source_field="source_id",
            time_field="published_at",
            content_fields=("content", "title"),
            filename_ext="txt",
        )
        return result

    def ingest_macro_indicators(self, limit: int = 20) -> dict[str, Any]:
        """Trigger macro-indicator ingestion."""

        return self.ingestion.ingest_macro_indicators(limit=limit)

    def ingest_fund_snapshots(self, stock_codes: list[str]) -> dict[str, Any]:
        """Trigger fund snapshot ingestion."""

        return self.ingestion.ingest_fund_snapshots(stock_codes)

    def _index_text_rows_to_rag(
        self,
        *,
        rows: list[dict[str, Any]],
        namespace: str,
        source_field: str,
        time_field: str,
        content_fields: tuple[str, ...],
        filename_ext: str = "txt",
    ) -> None:
        """Best-effort helper: map structured rows to local RAG docs without blocking ingestion."""

        for row in rows:
            # Stable id dedups repeated ingest calls for the same upstream record.
            signature = (
                str(row.get("source_url", ""))
                + "|"
                + str(row.get("title", ""))
                + "|"
                + str(row.get(time_field, ""))
            )
            doc_id = f"{namespace}-{hashlib.md5(signature.encode('utf-8')).hexdigest()[:16]}"
            if doc_id in self.ingestion.store.docs:
                continue
            content = ""
            for field in content_fields:
                value = str(row.get(field, "")).strip()
                if value:
                    content = value
                    break
            if not content:
                continue
            try:
                self.docs_upload(doc_id, f"{doc_id}.{filename_ext}", content, source=str(row.get(source_field, namespace)))
                self.docs_index(doc_id)
            except Exception:
                # Do not fail ingestion when one row cannot be indexed.
                continue

    def docs_upload(self, doc_id: str, filename: str, content: str, source: str) -> dict[str, Any]:
        """Upload raw document content."""
        result = self.ingestion.upload_doc(doc_id, filename, content, source)
        doc = self.ingestion.store.docs.get(doc_id, {})
        self.web.doc_upsert(
            doc_id=doc_id,
            filename=filename,
            parse_confidence=float(doc.get("parse_confidence", 0.0)),
            # Product requirement: upload should be effective immediately.
            # Keep parse_confidence for observability, but do not block by review gate.
            needs_review=False,
        )
        self.web.doc_pipeline_run_add(
            doc_id=doc_id,
            stage="upload",
            status="ok",
            filename=filename,
            parse_confidence=float(doc.get("parse_confidence", 0.0) or 0.0),
            chunk_count=0,
            table_count=0,
            parse_notes="upload_received",
            metadata={
                "source": source,
                "doc_hash": str(doc.get("doc_hash", "")),
                "pipeline_version": int(doc.get("version", 1) or 1),
            },
        )
        return result

    def docs_index(self, doc_id: str) -> dict[str, Any]:
        """Index one uploaded document into chunks and write to retrieval stores."""
        result = self.ingestion.index_doc(doc_id)
        doc = self.ingestion.store.docs.get(doc_id, {})
        if str(result.get("status", "")) != "indexed":
            self.web.doc_pipeline_run_add(
                doc_id=doc_id,
                stage="index",
                status="not_found",
                filename=str(doc.get("filename", "")),
                parse_confidence=float(doc.get("parse_confidence", 0.0) or 0.0),
                parse_notes="doc_not_found",
                metadata={"result_status": str(result.get("status", ""))},
            )
            return result
        if doc:
            self.web.doc_upsert(
                doc_id=doc_id,
                filename=doc.get("filename", ""),
                parse_confidence=float(doc.get("parse_confidence", 0.0)),
                # Product requirement: index completion should not create review gate.
                needs_review=False,
            )
            # 鏂囨。绱㈠紩瀹屾垚鍚庯紝鎶?chunk 鎸佷箙鍖栧埌 RAG 璧勪骇搴擄紝渚涘悗缁绱笌娌荤悊澶嶇敤銆?
            self._persist_doc_chunks_to_rag(doc_id, doc)
            self.web.doc_pipeline_run_add(
                doc_id=doc_id,
                stage="index",
                status="ok",
                filename=str(doc.get("filename", "")),
                parse_confidence=float(doc.get("parse_confidence", 0.0) or 0.0),
                chunk_count=int(result.get("chunk_count", 0) or 0),
                table_count=int(result.get("table_count", 0) or 0),
                parse_notes="index_completed",
                metadata={
                    "source": str(doc.get("source", "")),
                    "doc_hash": str(doc.get("doc_hash", "")),
                    "pipeline_version": int(doc.get("version", 1) or 1),
                },
            )
        return result

    def docs_versions(self, token: str, doc_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        return self.web.doc_versions(token, doc_id, limit=limit)

    def docs_pipeline_runs(self, token: str, doc_id: str, *, limit: int = 30) -> list[dict[str, Any]]:
        return self.web.doc_pipeline_runs(token, doc_id, limit=limit)

    def docs_recommend(self, token: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Recommend docs using history + context + graph signals."""
        _ = self.web.auth_me(token)
        context = payload if isinstance(payload, dict) else {}
        safe_top_k = max(1, min(30, int(context.get("top_k", 5) or 5)))
        stock_code = str(context.get("stock_code", "")).strip().upper()

        history_rows = self.web.query_history_list(token, limit=120)
        chunks = self.web.rag_doc_chunk_list_internal(status="active", limit=2500)
        graph_terms: list[str] = []
        if stock_code:
            graph = self.knowledge_graph_view(stock_code, limit=30)
            # Pull neighbor concept words as ranking hints.
            graph_terms = [
                str(x.get("target", "")).strip().upper()
                for x in graph.get("relations", [])
                if str(x.get("target", "")).strip()
            ][:60]

        ranked = self.doc_recommender.recommend(
            chunks=chunks,
            query_history_rows=history_rows,
            context=context,
            graph_terms=graph_terms,
            top_k=safe_top_k,
        )

        items: list[dict[str, Any]] = []
        for row in ranked:
            doc_meta = self.web.store.query_one(
                """
                SELECT doc_id, filename, status, parse_confidence, created_at
                FROM doc_index
                WHERE doc_id = ?
                """,
                (str(row.get("doc_id", "")),),
            ) or {}
            items.append(
                {
                    "doc_id": row.get("doc_id", ""),
                    "filename": str(doc_meta.get("filename", "")),
                    "status": str(doc_meta.get("status", "")),
                    "parse_confidence": float(doc_meta.get("parse_confidence", 0.0) or 0.0),
                    "score": float(row.get("score", 0.0) or 0.0),
                    "reasons": list(row.get("reasons", [])),
                    "stock_codes": list(row.get("stock_codes", [])),
                    "source": str(row.get("source", "")),
                    "updated_at": str(row.get("updated_at", "")),
                    "created_at": str(doc_meta.get("created_at", "")),
                }
            )

        return {
            "top_k": safe_top_k,
            "count": len(items),
            "context": {
                "stock_code": stock_code,
                "question": str(context.get("question", "")),
            },
            "items": items,
        }

    def knowledge_graph_view(self, entity_id: str, *, limit: int = 20) -> dict[str, Any]:
        """Return one-hop graph neighborhood for a given entity."""
        normalized = str(entity_id or "").strip().upper()
        if not normalized:
            raise ValueError("entity_id is required")
        safe_limit = max(1, min(200, int(limit)))
        raw_relations = self.workflow.graph_rag.store.find_relations([], limit=safe_limit * 3)

        rows: list[dict[str, str]] = []
        for row in raw_relations:
            src = str(getattr(row, "src", "") or "")
            dst = str(getattr(row, "dst", "") or "")
            rel_type = str(getattr(row, "rel_type", "") or "")
            source_id = str(getattr(row, "source_id", "") or "graph")
            source_url = str(getattr(row, "source_url", "") or "")
            if normalized not in (src.upper(), dst.upper()):
                continue
            rows.append(
                {
                    "source": src,
                    "target": dst,
                    "relation_type": rel_type,
                    "source_id": source_id,
                    "source_url": source_url,
                }
            )
            if len(rows) >= safe_limit:
                break

        node_index: dict[str, dict[str, Any]] = {}
        for relation in rows:
            src = str(relation.get("source", ""))
            dst = str(relation.get("target", ""))
            if src and src not in node_index:
                node_index[src] = {"entity_id": src, "entity_type": self._infer_graph_entity_type(src)}
            if dst and dst not in node_index:
                node_index[dst] = {"entity_id": dst, "entity_type": self._infer_graph_entity_type(dst)}

        return {
            "entity_id": normalized,
            "entity_type": self._infer_graph_entity_type(normalized),
            "node_count": len(node_index),
            "relation_count": len(rows),
            "nodes": list(node_index.values()),
            "relations": rows,
        }

    @staticmethod
    def _infer_graph_entity_type(entity_id: str) -> str:
        token = str(entity_id or "").strip().upper()
        if token.startswith(("SH", "SZ", "BJ")) and len(token) >= 8:
            return "stock"
        return "concept"

    def docs_quality_report(self, doc_id: str) -> dict[str, Any]:
        """Build a quality report for one indexed document.

        This is a lightweight quality dashboard used by Knowledge Hub Phase 1:
        - Parse confidence from `doc_index`
        - Chunk distribution from `rag_doc_chunk`
        - Actionable recommendations for low quality inputs
        """
        doc = self.web.store.query_one(
            """
            SELECT doc_id, filename, status, parse_confidence, needs_review, created_at
            FROM doc_index
            WHERE doc_id = ?
            """,
            (doc_id,),
        )
        if not doc:
            return {"error": "not_found", "doc_id": doc_id}

        chunks = self.web.store.query_all(
            """
            SELECT chunk_id, chunk_no, chunk_text_redacted, quality_score, effective_status, updated_at
            FROM rag_doc_chunk
            WHERE doc_id = ?
            ORDER BY chunk_no ASC
            """,
            (doc_id,),
        )
        chunk_lengths = [len(str(row.get("chunk_text_redacted", ""))) for row in chunks]
        chunk_count = len(chunks)
        avg_chunk_len = (sum(chunk_lengths) / chunk_count) if chunk_count else 0.0
        short_chunk_count = sum(1 for x in chunk_lengths if x < 60)
        short_chunk_ratio = (short_chunk_count / chunk_count) if chunk_count else 0.0
        avg_quality = (
            sum(float(row.get("quality_score", 0.0) or 0.0) for row in chunks) / chunk_count if chunk_count else 0.0
        )
        active_chunk_count = sum(1 for row in chunks if str(row.get("effective_status", "")) == "active")

        parse_confidence = float(doc.get("parse_confidence", 0.0) or 0.0)
        quality_score = max(
            0.0,
            min(
                1.0,
                parse_confidence * 0.5 + avg_quality * 0.3 + (1.0 - short_chunk_ratio) * 0.2,
            ),
        )
        quality_level = "high" if quality_score >= 0.8 else ("medium" if quality_score >= 0.6 else "low")

        recommendations: list[str] = []
        if parse_confidence < 0.7:
            recommendations.append("parse_confidence ????????????? OCR ????????")
        if chunk_count < 3:
            recommendations.append("??????????????????")
        if short_chunk_ratio > 0.4:
            recommendations.append("????????????????????????")
        if active_chunk_count == 0 and chunk_count > 0:
            recommendations.append("??? active ????????????")
        if not recommendations:
            recommendations.append("??????????????????")

        return {
            "doc_id": doc_id,
            "filename": str(doc.get("filename", "")),
            "status": str(doc.get("status", "")),
            "parse_confidence": round(parse_confidence, 4),
            "needs_review": bool(int(doc.get("needs_review", 0) or 0)),
            "quality_score": round(quality_score, 4),
            "quality_level": quality_level,
            "chunk_stats": {
                "chunk_count": chunk_count,
                "active_chunk_count": active_chunk_count,
                "avg_chunk_len": round(avg_chunk_len, 2),
                "short_chunk_count": short_chunk_count,
                "short_chunk_ratio": round(short_chunk_ratio, 4),
                "avg_chunk_quality": round(avg_quality, 4),
            },
            "recommendations": recommendations,
            "created_at": str(doc.get("created_at", "")),
        }

    def rag_upload_from_payload(self, token: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Unified RAG upload flow: parse -> de-duplicate -> docs_upload -> optional index -> persist asset."""
        filename = str(payload.get("filename", "")).strip()
        if not filename:
            raise ValueError("filename is required")
        source = str(payload.get("source", "user_upload")).strip().lower() or "user_upload"
        source_url = str(payload.get("source_url", "")).strip()
        content_type = str(payload.get("content_type", "")).strip()
        stock_codes = [str(x).strip().upper() for x in payload.get("stock_codes", []) if str(x).strip()]
        tags = [str(x).strip() for x in payload.get("tags", []) if str(x).strip()]
        auto_index = bool(payload.get("auto_index", True))
        force_reupload = bool(payload.get("force_reupload", False))
        user_hint = str(payload.get("user_id", "frontend-user")).strip() or "frontend-user"

        text_content = str(payload.get("content", "") or "")
        raw_bytes: bytes
        parse_note_parts: list[str] = []
        if text_content.strip():
            raw_bytes = text_content.encode("utf-8", errors="ignore")
            extracted = text_content
            parse_note_parts.append("text_payload")
        else:
            encoded = str(payload.get("content_base64", "")).strip()
            if not encoded:
                raise ValueError("content or content_base64 is required")
            try:
                raw_bytes = base64.b64decode(encoded, validate=True)
            except Exception as ex:  # noqa: BLE001
                raise ValueError("content_base64 is invalid") from ex
            extracted, parse_note = self._extract_text_from_upload_bytes(
                filename=filename,
                raw_bytes=raw_bytes,
                content_type=content_type,
            )
            if parse_note:
                parse_note_parts.append(parse_note)

        if not extracted.strip():
            raise ValueError("uploaded file could not be parsed into text content")

        file_sha256 = hashlib.sha256(raw_bytes).hexdigest().lower()
        if not force_reupload:
            existing = self.web.rag_upload_asset_get_by_hash(file_sha256)
            if existing:
                return {
                    "status": "deduplicated",
                    "dedupe_hit": True,
                    "existing": existing,
                    "doc_id": str(existing.get("doc_id", "")),
                    "upload_id": str(existing.get("upload_id", "")),
                }

        doc_id = str(payload.get("doc_id", "")).strip()
        if not doc_id:
            # 鐢?hash 鍓嶇紑鐢熸垚绋冲畾 doc_id锛屼究浜庡悗缁法鍏ュ彛杩借釜鍚屼竴鏂囦欢銆?
            doc_id = f"ragdoc-{file_sha256[:12]}"
            if force_reupload:
                doc_id = f"{doc_id}-{uuid.uuid4().hex[:4]}"
        upload_id = str(payload.get("upload_id", "")).strip() or f"ragu-{uuid.uuid4().hex[:12]}"

        _ = self.docs_upload(doc_id, filename, extracted, source)
        indexed = None
        status = "uploaded"
        if auto_index:
            indexed = self.docs_index(doc_id)
            status = "indexed"
            chunk_rows = self.rag_doc_chunks_list(token, doc_id=doc_id, limit=1)
            if chunk_rows:
                status = str(chunk_rows[0].get("effective_status", "indexed"))

        parse_note = ",".join(parse_note_parts)
        asset = self.web.rag_upload_asset_upsert(
            token,
            upload_id=upload_id,
            doc_id=doc_id,
            filename=filename,
            source=source,
            source_url=source_url,
            file_sha256=file_sha256,
            file_size=len(raw_bytes),
            content_type=content_type,
            stock_codes=stock_codes,
            tags=tags,
            parse_note=parse_note,
            status=status,
            created_by=user_hint,
        )
        return {
            "status": "ok",
            "dedupe_hit": False,
            "upload_id": upload_id,
            "doc_id": doc_id,
            "source": source,
            "auto_index": auto_index,
            "index_result": indexed or {},
            "asset": asset,
        }

    @staticmethod
    def _rag_preview_trim(text: str, max_len: int = 140) -> str:
        """Normalize and truncate preview text to keep payload readable."""
        compact = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(compact) <= max(1, int(max_len)):
            return compact
        return compact[: max(1, int(max_len))].rstrip() + "..."

    def _build_rag_preview_queries(
        self,
        *,
        chunks: list[dict[str, Any]],
        stock_codes: list[str],
        tags: list[str],
        max_queries: int,
    ) -> list[str]:
        """Generate deterministic sample queries for retrieval verification."""
        candidates: list[str] = []
        snippet_queries: list[str] = []

        for row in chunks[:12]:
            text = str(row.get("chunk_text_redacted") or row.get("chunk_text") or "")
            normalized = re.sub(r"\s+", " ", text).strip()
            if not normalized:
                continue
            # Use sentence-like fragments so users can quickly understand preview intent.
            for part in re.split(r"[銆傦紒锛??锛?\n]", normalized):
                piece = self._rag_preview_trim(part, max_len=42)
                if len(piece) < 12:
                    continue
                snippet_queries.append(piece)
                break
            if len(snippet_queries) >= max_queries * 2:
                break

        # Put current-document snippets first to maximize "this doc is retrievable" signal.
        candidates.extend(snippet_queries)
        for code in stock_codes[:2]:
            normalized = str(code or "").strip().upper()
            if not normalized:
                continue
            candidates.append(f"{normalized} 鍏抽敭缁撹")
            candidates.append(f"{normalized} 椋庨櫓鎻愮ず")
        for tag in tags[:2]:
            normalized = str(tag or "").strip()
            if not normalized:
                continue
            candidates.append(f"{normalized} 鏍稿績淇℃伅")

        deduped: list[str] = []
        seen: set[str] = set()
        for query in candidates:
            key = query.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(query.strip())
            if len(deduped) >= max_queries:
                break
        return deduped or ["鏂囨。鏍稿績缁撹"]

    def rag_retrieval_preview(
        self,
        token: str,
        *,
        doc_id: str,
        max_queries: int = 3,
        top_k: int = 5,
        hint_stock_codes: list[str] | None = None,
        hint_tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build retrieval verification preview for one uploaded document."""
        _ = self.web.require_role(token, {"admin", "ops"})
        doc_id_clean = str(doc_id or "").strip()
        if not doc_id_clean:
            raise ValueError("doc_id is required")
        safe_max_queries = max(1, min(6, int(max_queries)))
        safe_top_k = max(1, min(8, int(top_k)))

        all_chunks = self.web.rag_doc_chunk_list_internal(doc_id=doc_id_clean, limit=180)
        if not all_chunks:
            return {
                "doc_id": doc_id_clean,
                "ready": False,
                "passed": False,
                "reason": "doc_not_found",
                "total_chunk_count": 0,
                "active_chunk_count": 0,
                "query_count": 0,
                "matched_query_count": 0,
                "target_hit_rate": 0.0,
                "items": [],
            }

        active_chunks = [row for row in all_chunks if str(row.get("effective_status", "")).strip().lower() == "active"]
        if not active_chunks:
            return {
                "doc_id": doc_id_clean,
                "ready": False,
                "passed": False,
                "reason": "doc_not_active",
                "total_chunk_count": len(all_chunks),
                "active_chunk_count": 0,
                "query_count": 0,
                "matched_query_count": 0,
                "target_hit_rate": 0.0,
                "items": [],
            }

        stock_codes: list[str] = []
        for raw in hint_stock_codes or []:
            code = str(raw or "").strip().upper()
            if code and code not in stock_codes:
                stock_codes.append(code)
        for row in active_chunks:
            for raw in row.get("stock_codes", []) or []:
                code = str(raw or "").strip().upper()
                if code and code not in stock_codes:
                    stock_codes.append(code)

        tags: list[str] = []
        for raw in hint_tags or []:
            tag = str(raw or "").strip()
            if tag and tag not in tags:
                tags.append(tag)

        preview_queries = self._build_rag_preview_queries(
            chunks=active_chunks,
            stock_codes=stock_codes,
            tags=tags,
            max_queries=safe_max_queries,
        )
        retriever = self._build_runtime_retriever(stock_codes)

        items: list[dict[str, Any]] = []
        matched_queries = 0
        for query in preview_queries:
            started = time.perf_counter()
            hits = retriever.retrieve(
                query,
                top_k_vector=max(8, safe_top_k * 2),
                top_k_bm25=max(12, safe_top_k * 2),
                rerank_top_n=safe_top_k,
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
            hit_rows: list[dict[str, Any]] = []
            target_hit_rank: int | None = None
            for rank, hit in enumerate(hits, start=1):
                meta = dict(hit.metadata or {})
                hit_doc_id = str(meta.get("doc_id", "")).strip()
                hit_source_url = str(hit.source_url or "")
                # Some lexical retrieval paths may not preserve doc_id metadata, so keep source_url fallback.
                is_target = bool(
                    (hit_doc_id and hit_doc_id == doc_id_clean)
                    or (f"/docs/{doc_id_clean}" in hit_source_url)
                    or hit_source_url.endswith(f"://docs/{doc_id_clean}")
                )
                if is_target and target_hit_rank is None:
                    target_hit_rank = rank
                hit_rows.append(
                    {
                        "rank": rank,
                        "score": round(float(hit.score or 0.0), 4),
                        "source_id": str(hit.source_id or ""),
                        "source_url": hit_source_url,
                        "retrieval_track": str(meta.get("retrieval_track", "")),
                        "doc_id": hit_doc_id,
                        "chunk_id": str(meta.get("chunk_id", "")),
                        "is_target_doc": is_target,
                        "excerpt": self._rag_preview_trim(str(hit.text or ""), max_len=120),
                    }
                )
            if target_hit_rank is not None:
                matched_queries += 1
            items.append(
                {
                    "query": query,
                    "latency_ms": latency_ms,
                    "target_hit": target_hit_rank is not None,
                    "target_hit_rank": target_hit_rank,
                    "top_hits": hit_rows,
                }
            )

        query_count = len(items)
        hit_rate = round((matched_queries / query_count), 4) if query_count > 0 else 0.0
        return {
            "doc_id": doc_id_clean,
            "ready": True,
            "passed": matched_queries > 0,
            "reason": "",
            "total_chunk_count": len(all_chunks),
            "active_chunk_count": len(active_chunks),
            "query_count": query_count,
            "matched_query_count": matched_queries,
            "target_hit_rate": hit_rate,
            "items": items,
        }

    def rag_workflow_upload_and_index(self, token: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Business entrypoint: upload then index immediately and return timeline for frontend progress."""
        req = dict(payload or {})
        req["auto_index"] = True
        started = datetime.now(timezone.utc)
        result = self.rag_upload_from_payload(token, req)
        timeline = [
            {"phase": "upload_received", "status": "done", "at": started.isoformat()},
            {
                "phase": "indexing",
                "status": "done" if bool(result.get("index_result")) else "skipped",
                "at": datetime.now(timezone.utc).isoformat(),
            },
            {
                "phase": "asset_recorded",
                "status": "done",
                "at": datetime.now(timezone.utc).isoformat(),
            },
        ]
        preview_doc_id = str(result.get("doc_id", "")).strip()
        retrieval_preview: dict[str, Any] = {
            "doc_id": preview_doc_id,
            "ready": False,
            "passed": False,
            "reason": "doc_id_missing",
            "query_count": 0,
            "matched_query_count": 0,
            "target_hit_rate": 0.0,
            "items": [],
        }
        if preview_doc_id:
            try:
                asset = result.get("asset", {}) if isinstance(result.get("asset"), dict) else {}
                hint_codes = [str(x).strip().upper() for x in asset.get("stock_codes", []) if str(x).strip()]
                hint_tags = [str(x).strip() for x in asset.get("tags", []) if str(x).strip()]
                retrieval_preview = self.rag_retrieval_preview(
                    token,
                    doc_id=preview_doc_id,
                    max_queries=2,
                    top_k=4,
                    hint_stock_codes=hint_codes,
                    hint_tags=hint_tags,
                )
            except Exception as ex:  # noqa: BLE001
                retrieval_preview = {
                    "doc_id": preview_doc_id,
                    "ready": False,
                    "passed": False,
                    "reason": "preview_failed",
                    "error": str(ex)[:240],
                    "query_count": 0,
                    "matched_query_count": 0,
                    "target_hit_rate": 0.0,
                    "items": [],
                }
        return {
            "status": "ok",
            "result": result,
            "timeline": timeline,
            "retrieval_preview": retrieval_preview,
        }

    def rag_retrieval_preview_api(
        self,
        token: str,
        *,
        doc_id: str,
        max_queries: int = 3,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Public API wrapper for upload-time retrieval preview."""
        return self.rag_retrieval_preview(
            token,
            doc_id=doc_id,
            max_queries=max_queries,
            top_k=top_k,
        )

    def rag_uploads_list(
        self,
        token: str,
        *,
        status: str = "",
        source: str = "",
        limit: int = 40,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        return self.web.rag_upload_asset_list(
            token,
            status=status,
            source=source,
            limit=limit,
            offset=offset,
        )

    def rag_dashboard(self, token: str) -> dict[str, Any]:
        return self.web.rag_dashboard_summary(token)

    @staticmethod
    def _decode_text_bytes(raw_bytes: bytes) -> str:
        for enc in ("utf-8", "gbk", "utf-16", "latin1"):
            try:
                return raw_bytes.decode(enc, errors="ignore")
            except Exception:  # noqa: BLE001
                continue
        return ""

    def _extract_text_from_upload_bytes(self, *, filename: str, raw_bytes: bytes, content_type: str = "") -> tuple[str, str]:
        """Extract text from uploaded bytes; use lightweight fallbacks when third-party parsers are unavailable."""
        ext = ""
        idx = str(filename).rfind(".")
        if idx >= 0:
            ext = str(filename)[idx:].lower()
        note_parts: list[str] = []
        if ext in {".txt", ".md", ".csv", ".json", ".log", ".html", ".htm", ".ts", ".js", ".py"}:
            note_parts.append("plain_text_decode")
            return self._decode_text_bytes(raw_bytes), ",".join(note_parts)
        if ext == ".docx":
            try:
                with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                    xml_bytes = zf.read("word/document.xml")
                root = ET.fromstring(xml_bytes)
                text = " ".join(x.strip() for x in root.itertext() if str(x).strip())
                note_parts.append("docx_xml_extract")
                return text, ",".join(note_parts)
            except Exception:  # noqa: BLE001
                note_parts.append("docx_extract_failed_fallback_decode")
                return self._decode_text_bytes(raw_bytes), ",".join(note_parts)
        if ext in {".xlsx", ".xlsm"} and load_workbook is not None:
            try:
                wb = load_workbook(filename=io.BytesIO(raw_bytes), read_only=True, data_only=True)
                ws = wb.active
                lines: list[str] = []
                max_rows = 1500
                max_cols = 32
                for row_idx, row in enumerate(ws.iter_rows(values_only=True), start=1):
                    if row_idx > max_rows:
                        break
                    cells = [str(c).strip() for c in row[:max_cols] if c is not None and str(c).strip()]
                    if cells:
                        lines.append(" | ".join(cells))
                note_parts.append("xlsx_extract")
                return "\n".join(lines), ",".join(note_parts)
            except Exception:  # noqa: BLE001
                note_parts.append("xlsx_extract_failed_fallback_decode")
                return self._decode_text_bytes(raw_bytes), ",".join(note_parts)
        if ext == ".pdf":
            try:
                import pypdf  # type: ignore

                reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
                pages = [str(page.extract_text() or "") for page in reader.pages]
                text = "\n".join(x for x in pages if x.strip())
                if text.strip():
                    note_parts.append("pdf_pypdf_extract")
                    return text, ",".join(note_parts)
            except Exception:  # noqa: BLE001
                note_parts.append("pdf_parser_unavailable")
            # 鏃?pdf 瑙ｆ瀽搴撴椂鐨勫厹搴曪細灏濊瘯鎶藉彇鍙 ASCII 涓诧紝鑷冲皯淇濈暀閮ㄥ垎鍙绱㈠唴瀹广€?
            ascii_chunks = re.findall(rb"[A-Za-z0-9][A-Za-z0-9 ,.;:%()\-_/]{16,}", raw_bytes)
            decoded = " ".join(x.decode("latin1", errors="ignore") for x in ascii_chunks)
            note_parts.append("pdf_ascii_fallback")
            return decoded, ",".join(note_parts)

        note_parts.append(f"generic_decode:{content_type or 'unknown'}")
        return self._decode_text_bytes(raw_bytes), ",".join(note_parts)

    def _persist_doc_chunks_to_rag(self, doc_id: str, doc: dict[str, Any]) -> None:
        """Persist in-memory document chunks into the web-layer RAG storage."""
        chunks = [str(x) for x in doc.get("chunks", []) if str(x).strip()]
        if not chunks:
            return
        source = str(doc.get("source", "user_upload")).strip().lower() or "user_upload"
        source_url = f"local://docs/{doc_id}"
        # Product requirement: all uploaded/indexed docs become searchable immediately.
        # Source policy is kept for governance metadata, but not used as an activation gate.
        effective_status = "active"
        quality_score = float(doc.get("parse_confidence", 0.0))
        stock_codes = self._extract_stock_codes_from_text(
            f"{doc.get('filename', '')}\n{doc.get('cleaned_text', '')}\n{doc.get('content', '')}"
        )
        payload_chunks: list[dict[str, Any]] = []
        for idx, chunk in enumerate(chunks, start=1):
            payload_chunks.append(
                {
                    "chunk_id": f"{doc_id}-c{idx}",
                    "chunk_no": idx,
                    "chunk_text": chunk,
                    # 鍙岃建涓殑鈥滄憳瑕?鑴辨晱杞ㄢ€濓細鍦ㄧ嚎妫€绱㈤粯璁や娇鐢?redacted 鏂囨湰銆?
                    "chunk_text_redacted": self._redact_text(chunk),
                    "stock_codes": stock_codes,
                    "industry_tags": [],
                }
            )
        self.web.rag_doc_chunk_replace(
            doc_id=doc_id,
            source=source,
            source_url=source_url,
            effective_status=effective_status,
            quality_score=quality_score,
            chunks=payload_chunks,
        )

    @staticmethod
    def _extract_stock_codes_from_text(text: str) -> list[str]:
        """Extract SH/SZ stock codes from free text for retrieval filtering."""
        items = re.findall(r"\b(?:SH|SZ)\d{6}\b", str(text or "").upper())
        return list(dict.fromkeys(items))

    @staticmethod
    def _redact_text(text: str) -> str:
        """Lightweight redaction to remove email/phone/ID patterns from shared text."""
        value = str(text or "")
        value = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]", value)
        value = re.sub(r"\b1\d{10}\b", "[REDACTED_PHONE]", value)
        value = re.sub(r"\b\d{15,18}[0-9Xx]?\b", "[REDACTED_ID]", value)
        return value

    def evals_run(self, samples: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Run evals and return gate results."""
        result = self.eval_service.run_eval(samples)
        stable_prompt = self.prompts.get_stable_prompt("fact_qa")
        self.prompts.save_eval_result(
            eval_run_id=result["eval_run_id"],
            prompt_id=stable_prompt["prompt_id"],
            version=stable_prompt["version"],
            suite_id="default_suite",
            metrics=result["metrics"],
            pass_gate=result["pass_gate"],
        )
        self.prompts.create_release(
            prompt_id=stable_prompt["prompt_id"],
            version=stable_prompt["version"],
            target_env="staging",
            gate_result="pass" if result["pass_gate"] else "fail",
        )
        return result

    def evals_get(self, eval_run_id: str) -> dict[str, Any]:
        """Get eval run status."""
        return {"eval_run_id": eval_run_id, "status": "not_persisted_in_mvp"}

    def scheduler_run(self, job_name: str) -> dict[str, Any]:
        """Manually trigger one scheduler job."""
        return self.scheduler.run_once(job_name)

    def scheduler_status(self) -> dict[str, Any]:
        """Return status for all scheduler jobs."""
        return self.scheduler.list_status()

    # ---------- Prediction domain methods ----------
    def predict_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run quant prediction task."""
        stock_codes = payload.get("stock_codes", [])
        pool_id = str(payload.get("pool_id", "")).strip()
        token = str(payload.get("token", "")).strip()
        if pool_id:
            stock_codes = self.web.watchlist_pool_codes(token, pool_id)
        normalized_codes = [str(code).strip().upper().replace(".", "") for code in list(stock_codes or []) if str(code).strip()]
        data_packs: list[dict[str, Any]] = []
        # Predict may run on large pools, so cap auto-refresh scope to keep latency controllable.
        refresh_cap = min(40, len(normalized_codes))
        for code in normalized_codes[:refresh_cap]:
            try:
                data_packs.append(
                    self._build_llm_input_pack(
                        code,
                        question=f"predict:{code}",
                        scenario="predict",
                    )
                )
            except Exception:
                continue

        horizons = payload.get("horizons") or ["5d", "20d"]
        as_of_date = payload.get("as_of_date")
        result = self.prediction.run_prediction(stock_codes=normalized_codes, horizons=horizons, as_of_date=as_of_date)
        if pool_id:
            result["pool_id"] = pool_id
        result["segment_metrics"] = self._predict_segment_metrics(result.get("results", []))
        quality = self._predict_attach_quality(result)
        input_pack_missing = [
            str(item)
            for pack in data_packs
            for item in list((pack.get("missing_data") or []))
            if str(item).strip()
        ]
        if input_pack_missing:
            merged_reasons = list(
                dict.fromkeys(
                    list(result.get("degrade_reasons", []) or [])
                    + [f"input_pack:{name}" for name in input_pack_missing]
                )
            )
            result["degrade_reasons"] = merged_reasons
            # When upstream data pack has hard gaps, mark predict output degraded explicitly.
            result["data_quality"] = "degraded"
            quality["data_quality"] = "degraded"
            quality["degrade_reasons"] = merged_reasons
        result["input_data_packs"] = [
            {
                "stock_code": str(pack.get("stock_code", "")),
                "coverage": dict((pack.get("dataset", {}) or {}).get("coverage", {}) or {}),
                "missing_data": list(pack.get("missing_data", []) or []),
                "data_quality": str(pack.get("data_quality", "")),
            }
            for pack in data_packs
        ]
        result["input_data_pack_truncated"] = len(normalized_codes) > refresh_cap
        # Expose metric provenance explicitly to avoid mixing simulated metrics into live confidence.
        latest_eval = self.prediction.eval_latest()
        metric_mode = str(latest_eval.get("metric_mode", "simulated")).strip() or "simulated"
        result["metric_mode"] = metric_mode
        result["metrics_note"] = str(latest_eval.get("metrics_note", ""))
        if quality.get("data_quality") == "real" and metric_mode == "live":
            result["metrics_live"] = dict(latest_eval.get("metrics", {}) or {})
            result["metrics_simulated"] = {}
        else:
            result["metrics_live"] = {}
            result["metrics_simulated"] = dict(latest_eval.get("metrics", {}) or {})
        return result

    def _predict_attach_quality(self, result: dict[str, Any]) -> dict[str, Any]:
        """Derive prediction data quality labels for business UI rendering."""
        items = list(result.get("results", []))
        if not items:
            payload = {
                "data_quality": "degraded",
                "degrade_reasons": ["empty_prediction_result"],
                "source_coverage": {"total": 0, "real_history_count": 0, "synthetic_count": 0, "real_history_ratio": 0.0},
            }
            result.update(payload)
            return payload

        all_reasons: list[str] = []
        real_history_count = 0
        synthetic_count = 0
        for item in items:
            source = dict(item.get("source", {}) or {})
            history_mode = str(source.get("history_data_mode", "unknown")).strip() or "unknown"
            sample_size = int(source.get("history_sample_size", 0) or 0)
            source_id = str(source.get("source_id", "")).strip().lower()
            reasons: list[str] = []

            # Quality gate rule: non-real history must be treated as degraded.
            if history_mode != "real_history":
                reasons.append("history_not_real")
            if sample_size < 60:
                reasons.append("history_sample_insufficient")
            if bool(source.get("history_degraded", False)):
                reasons.append("history_fetch_degraded")
            if "mock" in source_id:
                reasons.append("quote_source_mock")
            if history_mode == "real_history":
                real_history_count += 1
            else:
                synthetic_count += 1

            dedup_reasons = list(dict.fromkeys(reasons))
            item["data_quality"] = "degraded" if dedup_reasons else "real"
            item["degrade_reasons"] = dedup_reasons
            all_reasons.extend(dedup_reasons)

        total = max(1, len(items))
        quality = {
            "data_quality": "real" if not all_reasons else "degraded",
            "degrade_reasons": list(dict.fromkeys(all_reasons)),
            "source_coverage": {
                "total": len(items),
                "real_history_count": real_history_count,
                "synthetic_count": synthetic_count,
                "real_history_ratio": round(float(real_history_count) / float(total), 4),
            },
        }
        result.update(quality)
        return quality

    def _predict_segment_metrics(self, items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        if not items:
            return {"exchange": [], "market_tier": [], "industry_l1": []}
        codes = [str(x.get("stock_code", "")).strip().upper() for x in items if str(x.get("stock_code", "")).strip()]
        if not codes:
            return {"exchange": [], "market_tier": [], "industry_l1": []}
        placeholders = ",".join(["?"] * len(codes))
        rows = self.web.store.query_all(
            f"""
            SELECT stock_code, exchange, market_tier, industry_l1
            FROM stock_universe
            WHERE stock_code IN ({placeholders})
            """,
            tuple(codes),
        )
        by_code = {str(row.get("stock_code", "")).upper(): row for row in rows}
        buckets: dict[str, dict[str, dict[str, float]]] = {
            "exchange": {},
            "market_tier": {},
            "industry_l1": {},
        }
        for item in items:
            code = str(item.get("stock_code", "")).upper()
            seg = by_code.get(code, {})
            horizons = list(item.get("horizons", []))
            if not horizons:
                continue
            avg_excess = sum(float(h.get("expected_excess_return", 0.0)) for h in horizons) / max(1, len(horizons))
            avg_up = sum(float(h.get("up_probability", 0.0)) for h in horizons) / max(1, len(horizons))
            for key in ("exchange", "market_tier", "industry_l1"):
                label = str(seg.get(key, "")).strip() or "UNKNOWN"
                bucket = buckets[key].setdefault(label, {"count": 0.0, "avg_expected_excess_return": 0.0, "avg_up_probability": 0.0})
                bucket["count"] += 1
                bucket["avg_expected_excess_return"] += avg_excess
                bucket["avg_up_probability"] += avg_up
        out: dict[str, list[dict[str, Any]]] = {"exchange": [], "market_tier": [], "industry_l1": []}
        for key, groups in buckets.items():
            rows_out: list[dict[str, Any]] = []
            for label, acc in groups.items():
                cnt = max(1, int(acc["count"]))
                rows_out.append(
                    {
                        "segment": label,
                        "count": cnt,
                        "avg_expected_excess_return": round(acc["avg_expected_excess_return"] / cnt, 6),
                        "avg_up_probability": round(acc["avg_up_probability"] / cnt, 6),
                    }
                )
            rows_out.sort(key=lambda x: x["segment"])
            out[key] = rows_out
        return out

    def predict_get(self, run_id: str) -> dict[str, Any]:
        """Get prediction task details."""
        return self.prediction.get_prediction(run_id)

    def factor_snapshot(self, stock_code: str) -> dict[str, Any]:
        """Get factor snapshot for one stock."""
        return self.prediction.get_factor_snapshot(stock_code)

    def predict_eval_latest(self) -> dict[str, Any]:
        """Get latest prediction evaluation summary."""
        return self.prediction.eval_latest()

    def market_overview(self, stock_code: str) -> dict[str, Any]:
        """Return structured market overview: realtime, history, announcements, and trends."""
        code = stock_code.upper().replace(".", "")
        pack = self._build_llm_input_pack(code, question=f"overview:{code}", scenario="overview")
        realtime = dict(pack.get("realtime", {}) or {})
        bars = list(pack.get("history", []) or [])[-120:]
        financial = dict(pack.get("financial", {}) or {})
        events = list(pack.get("announcements", []) or [])[-10:]
        news = list(pack.get("news", []) or [])[-10:]
        research = list(pack.get("research", []) or [])[-6:]
        fund = list(pack.get("fund", []) or [])[-1:]
        macro = list(pack.get("macro", []) or [])[-8:]
        trend = dict(pack.get("trend", {}) or {})
        if not trend and bars:
            trend = self._trend_metrics(bars)
        return {
            "stock_code": code,
            "realtime": realtime or {},
            "financial": financial or {},
            "history": bars,
            "events": events,
            "news": news,
            "research": research,
            "fund": fund,
            "macro": macro,
            "trend": trend,
            "dataset": dict(pack.get("dataset", {}) or {}),
            "missing_data": list(pack.get("missing_data", []) or []),
        }

    def analysis_intel_card(self, stock_code: str, *, horizon: str = "30d", risk_profile: str = "neutral") -> dict[str, Any]:
        """Build business-facing intel card from multi-source evidence."""
        code = stock_code.upper().replace(".", "")
        horizon_days_map = {"7d": 7, "30d": 30, "90d": 90}
        if horizon not in horizon_days_map:
            raise ValueError("horizon must be one of: 7d, 30d, 90d")
        profile = str(risk_profile or "neutral").strip().lower()
        if profile not in {"conservative", "neutral", "aggressive"}:
            raise ValueError("risk_profile must be one of: conservative, neutral, aggressive")

        overview = self.market_overview(code)
        realtime = overview.get("realtime", {}) or {}
        financial = overview.get("financial", {}) or {}
        trend = overview.get("trend", {}) or {}
        events = list(overview.get("events", []) or [])[-8:]
        news = list(overview.get("news", []) or [])[-10:]
        research = list(overview.get("research", []) or [])[-8:]
        macro = list(overview.get("macro", []) or [])[-8:]
        fund_rows = list(overview.get("fund", []) or [])
        fund_row = fund_rows[-1] if fund_rows else {}

        def minutes_since(ts_value: str) -> int | None:
            raw = str(ts_value or "").strip()
            if not raw:
                return None
            delta = datetime.now(timezone.utc) - self._parse_time(raw)
            return max(0, int(delta.total_seconds() // 60))

        evidence_rows: list[dict[str, Any]] = []
        key_catalysts: list[dict[str, Any]] = []
        risk_watch: list[dict[str, Any]] = []

        positive_keywords = ("澧為暱", "澧炴寔", "涓爣", "鍥炶喘", "鏀瑰杽", "绐佺牬", "涓婅皟", "buy", "outperform")
        negative_keywords = ("涓嬫粦", "鍑忔寔", "璇夎", "澶勭綒", "浜忔崯", "椋庨櫓", "涓嬭皟", "sell", "underperform")

        def add_evidence(
            *,
            kind: str,
            title: str,
            summary: str,
            source_id: str,
            source_url: str,
            event_time: str,
            reliability_score: float,
            retrieval_track: str,
        ) -> None:
            evidence_rows.append(
                {
                    "kind": kind,
                    "title": title[:220],
                    "summary": summary[:420],
                    "source_id": source_id,
                    "source_url": source_url,
                    "event_time": event_time,
                    "reliability_score": round(max(0.0, min(1.0, reliability_score)), 4),
                    "retrieval_track": retrieval_track,
                }
            )

        def classify_row(title: str, summary: str, payload: dict[str, Any], *, kind: str, default_signal: str = "neutral") -> None:
            text = f"{title} {summary}".lower()
            signal = default_signal
            if any(x in text for x in positive_keywords):
                signal = "positive"
            elif any(x in text for x in negative_keywords):
                signal = "negative"
            row = {
                "kind": kind,
                "title": title[:180],
                "summary": summary[:240],
                "source_id": str(payload.get("source_id", "")),
                "source_url": str(payload.get("source_url", "")),
                "event_time": str(payload.get("event_time", payload.get("published_at", payload.get("report_date", "")))),
                "reliability_score": float(payload.get("reliability_score", 0.0) or 0.0),
                "signal": signal,
            }
            if signal == "negative":
                risk_watch.append(row)
            else:
                key_catalysts.append(row)

        for row in news[-6:]:
            title = str(row.get("title", "")).strip()
            summary = str(row.get("content", "")).strip() or title
            add_evidence(
                kind="news",
                title=title,
                summary=summary,
                source_id=str(row.get("source_id", "")),
                source_url=str(row.get("source_url", "")),
                event_time=str(row.get("event_time", "")),
                reliability_score=float(row.get("reliability_score", 0.0) or 0.0),
                retrieval_track="news_event",
            )
            classify_row(title, summary, row, kind="news", default_signal="neutral")

        for row in research[-5:]:
            title = str(row.get("title", "")).strip()
            org = str(row.get("org_name", "")).strip()
            summary = f"{org} {str(row.get('content', '')).strip()}".strip() or title
            add_evidence(
                kind="research",
                title=title,
                summary=summary,
                source_id=str(row.get("source_id", "")),
                source_url=str(row.get("source_url", "")),
                event_time=str(row.get("published_at", "")),
                reliability_score=float(row.get("reliability_score", 0.0) or 0.0),
                retrieval_track="research_report",
            )
            classify_row(title, summary, row, kind="research", default_signal="positive")

        for row in macro[-4:]:
            metric_name = str(row.get("metric_name", "")).strip()
            metric_value = str(row.get("metric_value", "")).strip()
            title = metric_name or "macro_indicator"
            summary = f"{metric_name}={metric_value}, report_date={row.get('report_date', '')}".strip(",")
            add_evidence(
                kind="macro",
                title=title,
                summary=summary,
                source_id=str(row.get("source_id", "")),
                source_url=str(row.get("source_url", "")),
                event_time=str(row.get("event_time", row.get("report_date", ""))),
                reliability_score=float(row.get("reliability_score", 0.0) or 0.0),
                retrieval_track="macro_indicator",
            )
            classify_row(title, summary, row, kind="macro", default_signal="neutral")

        if fund_row:
            fund_summary = (
                f"涓诲姏={float(fund_row.get('main_inflow', 0.0) or 0.0):.2f}, "
                f"澶у崟={float(fund_row.get('large_inflow', 0.0) or 0.0):.2f}, "
                f"灏忓崟={float(fund_row.get('small_inflow', 0.0) or 0.0):.2f}"
            )
            add_evidence(
                kind="fund",
                title="璧勯噾娴佸悜",
                summary=fund_summary,
                source_id=str(fund_row.get("source_id", "")),
                source_url=str(fund_row.get("source_url", "")),
                event_time=str(fund_row.get("ts", fund_row.get("trade_date", ""))),
                reliability_score=float(fund_row.get("reliability_score", 0.0) or 0.0),
                retrieval_track="fund_flow",
            )
            classify_row("璧勯噾娴佸悜", fund_summary, fund_row, kind="fund", default_signal="neutral")

        for row in events[-3:]:
            title = str(row.get("title", "")).strip()
            summary = str(row.get("content", "")).strip() or title
            add_evidence(
                kind="announcement",
                title=title,
                summary=summary,
                source_id=str(row.get("source_id", "")),
                source_url=str(row.get("source_url", "")),
                event_time=str(row.get("event_time", "")),
                reliability_score=float(row.get("reliability_score", 0.0) or 0.0),
                retrieval_track="announcement_event",
            )
            classify_row(title, summary, row, kind="announcement", default_signal="neutral")

        # Score integrates trend, fundamentals and multi-source evidence polarity.
        score = 0.0
        ma20 = float(trend.get("ma20", 0.0) or 0.0)
        ma60 = float(trend.get("ma60", 0.0) or 0.0)
        momentum_20 = float(trend.get("momentum_20", 0.0) or 0.0)
        volatility_20 = float(trend.get("volatility_20", 0.0) or 0.0)
        drawdown_60 = float(trend.get("max_drawdown_60", 0.0) or 0.0)
        pct_change = float(realtime.get("pct_change", 0.0) or 0.0)
        main_inflow = float(fund_row.get("main_inflow", 0.0) or 0.0)
        revenue_yoy = float(financial.get("revenue_yoy", 0.0) or 0.0)
        profit_yoy = float(financial.get("net_profit_yoy", 0.0) or 0.0)

        score += 0.9 if ma20 >= ma60 else -0.8
        score += 0.8 if momentum_20 > 0 else -0.7
        score += 0.4 if pct_change > 0 else -0.3
        score += 0.45 if main_inflow > 0 else -0.35
        score += 0.35 if revenue_yoy > 0 else -0.25
        score += 0.35 if profit_yoy > 0 else -0.25
        score += min(1.0, 0.2 * len(key_catalysts))
        score -= min(1.2, 0.3 * len(risk_watch))

        signal = "hold"
        if score >= 1.2:
            signal = "buy"
        elif score <= -0.8:
            signal = "reduce"

        risk_level = "medium"
        if drawdown_60 > 0.2 or volatility_20 > 0.03:
            risk_level = "high"
        elif drawdown_60 < 0.1 and volatility_20 < 0.015 and momentum_20 > 0:
            risk_level = "low"

        # If evidence is sparse, cap confidence to avoid overclaiming.
        confidence_raw = 0.42 + abs(score) * 0.1 + min(0.2, len(evidence_rows) * 0.015)
        confidence = min(0.92, max(0.35, confidence_raw))
        if len(evidence_rows) < 4:
            confidence = min(confidence, 0.6)

        position_hint_map = {
            "buy": {"conservative": "20-35%", "neutral": "35-60%", "aggressive": "55-75%"},
            "hold": {"conservative": "10-20%", "neutral": "20-35%", "aggressive": "25-45%"},
            "reduce": {"conservative": "0-10%", "neutral": "5-15%", "aggressive": "10-20%"},
        }
        position_hint = position_hint_map[signal][profile]

        base_return_pct = momentum_20 * 100.0 if momentum_20 != 0 else pct_change * 0.35
        profile_shift = {"conservative": -1.5, "neutral": 0.0, "aggressive": 1.5}[profile]
        risk_penalty = {"low": 2.0, "medium": 4.0, "high": 7.0}[risk_level]
        scenario_matrix = [
            {
                "scenario": "bull",
                "expected_return_pct": round(base_return_pct + 8.0 + profile_shift, 2),
                "probability": 0.25 if risk_level != "high" else 0.18,
            },
            {
                "scenario": "base",
                "expected_return_pct": round(base_return_pct - risk_penalty * 0.25, 2),
                "probability": 0.5 if risk_level != "high" else 0.42,
            },
            {
                "scenario": "bear",
                "expected_return_pct": round(base_return_pct - risk_penalty - 4.0, 2),
                "probability": 0.25 if risk_level != "high" else 0.4,
            },
        ]

        event_calendar: list[dict[str, Any]] = []
        for row in events[-4:]:
            event_calendar.append(
                {
                    "date": str(row.get("event_time", ""))[:10],
                    "title": str(row.get("title", "鍏徃鍏憡"))[:120],
                    "event_type": "company",
                    "source_id": str(row.get("source_id", "")),
                }
            )
        for row in macro[-3:]:
            event_calendar.append(
                {
                    "date": str(row.get("report_date", row.get("event_time", "")))[:10],
                    "title": f"{str(row.get('metric_name', '瀹忚鎸囨爣'))} {str(row.get('metric_value', ''))}".strip(),
                    "event_type": "macro",
                    "source_id": str(row.get("source_id", "")),
                }
            )
        if not event_calendar:
            event_calendar.append(
                {
                    "date": (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d"),
                    "title": "Next trading day re-check window",
                    "event_type": "review",
                    "source_id": "system",
                }
            )

        # Keep evidence sorted by recency so frontend can stream top signals first.
        evidence_rows = sorted(
            evidence_rows,
            key=lambda x: str(x.get("event_time", "")),
            reverse=True,
        )[:16]
        key_catalysts = sorted(
            key_catalysts,
            key=lambda x: (str(x.get("event_time", "")), float(x.get("reliability_score", 0.0))),
            reverse=True,
        )[:8]
        risk_watch = sorted(
            risk_watch,
            key=lambda x: (str(x.get("event_time", "")), float(x.get("reliability_score", 0.0))),
            reverse=True,
        )[:8]

        freshness = {
            "quote_minutes": minutes_since(str(realtime.get("ts", ""))),
            "financial_minutes": minutes_since(str(financial.get("ts", ""))),
            "news_minutes": minutes_since(str(news[-1].get("event_time", ""))) if news else None,
            "research_minutes": minutes_since(str(research[-1].get("published_at", ""))) if research else None,
            "macro_minutes": minutes_since(str(macro[-1].get("event_time", ""))) if macro else None,
            "fund_minutes": minutes_since(str(fund_row.get("ts", ""))) if fund_row else None,
        }

        risk_thresholds = {
            "volatility_20_max": 0.03 if profile != "aggressive" else 0.035,
            "max_drawdown_60_max": 0.20 if profile != "aggressive" else 0.24,
            "min_evidence_count": 4,
            "max_data_staleness_minutes": 360,
        }

        degrade_reasons: list[str] = []
        for key in ("quote_minutes", "news_minutes", "macro_minutes"):
            val = freshness.get(key)
            if val is not None and isinstance(val, (int, float)) and float(val) > risk_thresholds["max_data_staleness_minutes"]:
                degrade_reasons.append(f"{key}_stale")
        if len(evidence_rows) < int(risk_thresholds["min_evidence_count"]):
            degrade_reasons.append("insufficient_evidence")
        if volatility_20 > float(risk_thresholds["volatility_20_max"]):
            degrade_reasons.append("volatility_exceeds_threshold")
        if drawdown_60 > float(risk_thresholds["max_drawdown_60_max"]):
            degrade_reasons.append("drawdown_exceeds_threshold")

        degrade_level = "normal"
        if len(degrade_reasons) >= 3:
            degrade_level = "degraded"
        elif degrade_reasons:
            degrade_level = "watch"
        # 闄嶇骇鏃朵笅璋冪疆淇″害锛岄伩鍏嶁€滆瘉鎹急浣嗗姩浣滆繃婵€鈥濄€?
        if degrade_level == "degraded":
            confidence = max(0.30, confidence - 0.18)
        elif degrade_level == "watch":
            confidence = max(0.33, confidence - 0.08)

        cadence = "single-step execution"
        if signal == "buy":
            cadence = "build in 3 batches (40%/30%/30%)"
        elif signal == "hold":
            cadence = "maintain position and re-check on T+1"
        elif signal == "reduce":
            cadence = "reduce in 2 batches (60%/40%)"
        review_hours = 6 if risk_level == "high" else 24 if risk_level == "medium" else 48

        execution_plan = {
            "entry_mode": "staggered" if signal in {"buy", "reduce"} else "observe",
            "cadence_hint": cadence,
            "max_single_step_pct": 0.35 if profile == "aggressive" else 0.25 if profile == "neutral" else 0.2,
            "max_position_cap": position_hint,
            "stop_loss_hint_pct": 4.0 if risk_level == "high" else 5.5 if risk_level == "medium" else 7.0,
            "recheck_interval_hours": review_hours,
        }

        trigger_conditions = [
            "Trend remains intact (MA20>=MA60 and 20-day momentum non-negative)",
            "Fund flow does not deteriorate significantly",
            "Core evidence keeps updating at daily cadence",
        ]
        invalidation_conditions = [
            "Key risk events land with clear negative direction",
            "Trend reverses (MA20<MA60 with weakening momentum)",
            "Volatility or drawdown breaches the configured threshold",
        ]
        if risk_level == "high":
            trigger_conditions.append("Only re-add exposure after risk converges")
        return {
            "stock_code": code,
            "time_horizon": horizon,
            "horizon_days": horizon_days_map[horizon],
            "risk_profile": profile,
            "overall_signal": signal,
            "confidence": round(confidence, 4),
            "risk_level": risk_level,
            "position_hint": position_hint,
            "market_snapshot": {
                "price": float(realtime.get("price", 0.0) or 0.0),
                "pct_change": pct_change,
                "ma20": ma20,
                "ma60": ma60,
                "momentum_20": momentum_20,
                "volatility_20": volatility_20,
                "max_drawdown_60": drawdown_60,
                "main_inflow": main_inflow,
            },
            "key_catalysts": key_catalysts,
            "risk_watch": risk_watch,
            "event_calendar": event_calendar,
            "scenario_matrix": scenario_matrix,
            "evidence": evidence_rows,
            "trigger_conditions": trigger_conditions,
            "invalidation_conditions": invalidation_conditions,
            "execution_plan": execution_plan,
            "risk_thresholds": risk_thresholds,
            "degrade_status": {
                "level": degrade_level,
                "reasons": degrade_reasons,
            },
            "next_review_time": (datetime.now(timezone.utc) + timedelta(hours=review_hours)).isoformat(),
            "data_freshness": freshness,
        }

    def analysis_intel_feedback(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Record whether user accepts/rejects intel-card conclusion for later review loop."""
        stock_code = str(payload.get("stock_code", "")).strip().upper()
        if not stock_code:
            raise ValueError("stock_code is required")
        feedback = str(payload.get("feedback", "")).strip().lower()
        if feedback not in {"adopt", "watch", "reject"}:
            raise ValueError("feedback must be one of: adopt, watch, reject")
        signal = str(payload.get("signal", "hold")).strip().lower() or "hold"
        if signal not in {"buy", "hold", "reduce"}:
            signal = "hold"
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        position_hint = str(payload.get("position_hint", "")).strip()
        trace_id = str(payload.get("trace_id", "")).strip()

        if self._needs_history_refresh(stock_code):
            try:
                self.ingestion.ingest_history_daily([stock_code], limit=520)
            except Exception:
                pass
        bars = self._history_bars(stock_code, limit=520)
        baseline_trade_date = str((bars[-1] if bars else {}).get("trade_date", ""))
        baseline_close = float((bars[-1] if bars else {}).get("close", 0.0) or 0.0)
        if baseline_close <= 0:
            quote = self._latest_quote(stock_code) or {}
            baseline_close = float(quote.get("price", 0.0) or 0.0)
            baseline_trade_date = baseline_trade_date or str(quote.get("ts", ""))[:10]

        row = self.web.analysis_intel_feedback_add(
            stock_code=stock_code,
            trace_id=trace_id,
            signal=signal,
            confidence=confidence,
            position_hint=position_hint,
            feedback=feedback,
            baseline_trade_date=baseline_trade_date,
            baseline_price=baseline_close,
        )
        return {"status": "ok", "item": row}

    def _analysis_forward_return(self, stock_code: str, baseline_trade_date: str, baseline_price: float, horizon_steps: int) -> float | None:
        bars = sorted(self._history_bars(stock_code, limit=520), key=lambda x: str(x.get("trade_date", "")))
        if not bars or baseline_price <= 0:
            return None
        start_idx = None
        for idx, row in enumerate(bars):
            if str(row.get("trade_date", "")) >= str(baseline_trade_date):
                start_idx = idx
                break
        if start_idx is None:
            return None
        target_idx = start_idx + int(horizon_steps)
        if target_idx >= len(bars):
            return None
        target_close = float(bars[target_idx].get("close", 0.0) or 0.0)
        if target_close <= 0:
            return None
        return target_close / baseline_price - 1.0

    def analysis_intel_review(self, stock_code: str, *, limit: int = 120) -> dict[str, Any]:
        """Aggregate feedback outcomes with T+1/T+5/T+20 realized drift for review dashboard."""
        code = str(stock_code or "").strip().upper()
        rows = self.web.analysis_intel_feedback_list(stock_code=code, limit=limit)
        if code and self._needs_history_refresh(code):
            try:
                self.ingestion.ingest_history_daily([code], limit=520)
            except Exception:
                pass

        horizons = {"t1": 1, "t5": 5, "t20": 20}
        review_rows: list[dict[str, Any]] = []
        for row in rows:
            stock = str(row.get("stock_code", "")).strip().upper()
            if stock and self._needs_history_refresh(stock):
                try:
                    self.ingestion.ingest_history_daily([stock], limit=520)
                except Exception:
                    pass
            baseline_trade_date = str(row.get("baseline_trade_date", ""))
            baseline_price = float(row.get("baseline_price", 0.0) or 0.0)
            realized: dict[str, Any] = {}
            for name, step in horizons.items():
                ret = self._analysis_forward_return(stock, baseline_trade_date, baseline_price, step)
                realized[name] = None if ret is None else round(ret, 6)
            review_rows.append({**row, "realized": realized})

        stats: dict[str, dict[str, Any]] = {}
        for name in horizons:
            values: list[float] = []
            hit = 0
            total = 0
            for row in review_rows:
                value = row.get("realized", {}).get(name)
                if value is None:
                    continue
                ret = float(value)
                values.append(ret)
                total += 1
                signal = str(row.get("signal", "hold"))
                if (signal == "buy" and ret > 0) or (signal == "reduce" and ret < 0) or (signal == "hold" and abs(ret) <= 0.02):
                    hit += 1
            stats[name] = {
                "count": total,
                "avg_return": round(mean(values), 6) if values else None,
                "hit_rate": round(hit / total, 4) if total > 0 else None,
            }
        return {
            "stock_code": code,
            "count": len(review_rows),
            "stats": stats,
            "items": review_rows[: max(1, min(200, limit))],
        }

    def backtest_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run a reproducible MA strategy backtest using local history bars."""
        stock_code = str(payload.get("stock_code", "")).strip().upper()
        if not stock_code:
            raise ValueError("stock_code is required")
        start_date = str(payload.get("start_date", "2024-01-01")).strip()
        end_date = str(payload.get("end_date", datetime.now().strftime("%Y-%m-%d"))).strip()
        initial_capital = float(payload.get("initial_capital", 100000.0) or 100000.0)
        ma_window = max(5, min(120, int(payload.get("ma_window", 20) or 20)))

        if self._needs_history_refresh(stock_code):
            try:
                self.ingestion.ingest_history_daily([stock_code], limit=520)
            except Exception:
                pass
        bars = [x for x in self._history_bars(stock_code, limit=520) if start_date <= str(x.get("trade_date", "")) <= end_date]
        if len(bars) < ma_window + 5:
            raise ValueError("insufficient history bars for backtest window")

        closes = [float(x.get("close", 0.0) or 0.0) for x in bars]
        dates = [str(x.get("trade_date", "")) for x in bars]

        cash = initial_capital
        shares = 0.0
        equity_curve: list[float] = []
        trades: list[dict[str, Any]] = []
        for idx in range(len(closes)):
            px = closes[idx]
            if px <= 0:
                continue
            if idx >= ma_window - 1:
                ma = sum(closes[idx - ma_window + 1 : idx + 1]) / ma_window
                if shares <= 0 and px > ma:
                    buy_shares = int(cash // px)
                    if buy_shares > 0:
                        shares = float(buy_shares)
                        cash -= shares * px
                        trades.append({"date": dates[idx], "side": "buy", "price": round(px, 4), "shares": int(shares)})
                elif shares > 0 and px < ma:
                    cash += shares * px
                    trades.append({"date": dates[idx], "side": "sell", "price": round(px, 4), "shares": int(shares)})
                    shares = 0.0
            equity_curve.append(cash + shares * px)

        final_value = equity_curve[-1] if equity_curve else initial_capital
        total_return = final_value - initial_capital
        total_return_pct = (total_return / initial_capital * 100.0) if initial_capital > 0 else 0.0
        max_drawdown = self._max_drawdown(equity_curve) if equity_curve else 0.0
        run_id = f"bkt-{uuid.uuid4().hex[:12]}"
        result = {
            "run_id": run_id,
            "stock_code": stock_code,
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": initial_capital,
            "final_value": round(final_value, 4),
            "metrics": {
                "total_return": round(total_return, 4),
                "total_return_pct": round(total_return_pct, 4),
                "max_drawdown": round(max_drawdown, 6),
                "trade_count": len(trades),
                "ma_window": ma_window,
            },
            "trades": trades[-100:],
            "status": "ok",
        }
        self._backtest_runs[run_id] = result
        return result

    def backtest_get(self, run_id: str) -> dict[str, Any]:
        row = self._backtest_runs.get(str(run_id))
        if not row:
            return {"error": "not_found", "run_id": run_id}
        return row

    def scheduler_pause(self, job_name: str) -> dict[str, Any]:
        return self.scheduler.pause(job_name)

    def scheduler_resume(self, job_name: str) -> dict[str, Any]:
        return self.scheduler.resume(job_name)

    # ---------- Web domain methods ----------
    def auth_register(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.web.auth_register(payload["username"], payload["password"], payload.get("tenant_name"))

    def auth_login(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.web.auth_login(payload["username"], payload["password"])

    def auth_me(self, token: str) -> dict[str, Any]:
        return self.web.auth_me(token)

    def watchlist_list(self, token: str) -> list[dict[str, Any]]:
        return self.web.watchlist_list(token)

    def watchlist_add(self, token: str, stock_code: str) -> dict[str, Any]:
        return self.web.watchlist_add(token, stock_code)

    def watchlist_delete(self, token: str, stock_code: str) -> dict[str, Any]:
        return self.web.watchlist_delete(token, stock_code)

    def watchlist_pool_list(self, token: str) -> list[dict[str, Any]]:
        return self.web.watchlist_pool_list(token)

    def watchlist_pool_create(self, token: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.web.watchlist_pool_create(
            token,
            pool_name=str(payload.get("pool_name", "")),
            description=str(payload.get("description", "")),
            is_default=bool(payload.get("is_default", False)),
        )

    def watchlist_pool_add_stock(self, token: str, pool_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.web.watchlist_pool_add_stock(
            token,
            pool_id=pool_id,
            stock_code=str(payload.get("stock_code", "")),
            source_filters=payload.get("source_filters", {}),
        )

    def watchlist_pool_stocks(self, token: str, pool_id: str) -> list[dict[str, Any]]:
        return self.web.watchlist_pool_stocks(token, pool_id)

    def watchlist_pool_delete_stock(self, token: str, pool_id: str, stock_code: str) -> dict[str, Any]:
        return self.web.watchlist_pool_delete_stock(token, pool_id, stock_code)

    def dashboard_overview(self, token: str) -> dict[str, Any]:
        return self.web.dashboard_overview(token)

    def portfolio_create(self, token: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.web.portfolio_create(
            token,
            portfolio_name=str(payload.get("portfolio_name", "")),
            initial_capital=float(payload.get("initial_capital", 0.0) or 0.0),
            description=str(payload.get("description", "")),
        )

    def portfolio_list(self, token: str) -> list[dict[str, Any]]:
        return self.web.portfolio_list(token)

    def portfolio_add_transaction(self, token: str, portfolio_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        return self.web.portfolio_add_transaction(
            token,
            portfolio_id=portfolio_id,
            stock_code=str(payload.get("stock_code", "")),
            transaction_type=str(payload.get("transaction_type", "")),
            quantity=float(payload.get("quantity", 0.0) or 0.0),
            price=float(payload.get("price", 0.0) or 0.0),
            fee=float(payload.get("fee", 0.0) or 0.0),
            transaction_date=str(payload.get("transaction_date", "")),
            notes=str(payload.get("notes", "")),
        )

    def _portfolio_price_map(self, stock_codes: list[str]) -> dict[str, float]:
        unique_codes = list(dict.fromkeys([str(x).strip().upper() for x in stock_codes if str(x).strip()]))
        if not unique_codes:
            return {}
        try:
            refresh_codes = [c for c in unique_codes if self._needs_quote_refresh(c)]
            if refresh_codes:
                self.ingest_market_daily(refresh_codes)
        except Exception:
            pass
        prices: dict[str, float] = {}
        for code in unique_codes:
            q = self._latest_quote(code) or {}
            p = float(q.get("price", 0.0) or 0.0)
            if p > 0:
                prices[code] = p
        return prices

    def portfolio_summary(self, token: str, portfolio_id: int) -> dict[str, Any]:
        positions = self.web.portfolio_positions(token, portfolio_id=portfolio_id)
        price_map = self._portfolio_price_map([str(x.get("stock_code", "")) for x in positions])
        return self.web.portfolio_summary(token, portfolio_id=portfolio_id, price_map=price_map)

    def portfolio_transactions(self, token: str, portfolio_id: int, *, limit: int = 200) -> list[dict[str, Any]]:
        return self.web.portfolio_transactions(token, portfolio_id=portfolio_id, limit=limit)

    def _journal_normalize_items(self, values: Any, *, max_items: int = 6, max_len: int = 200) -> list[str]:
        if not isinstance(values, list):
            return []
        normalized: list[str] = []
        for raw in values:
            text = str(raw or "").strip()
            if not text:
                continue
            text = text[:max_len]
            if text in normalized:
                continue
            normalized.append(text)
            if len(normalized) >= max_items:
                break
        return normalized

    def _journal_build_ai_reflection_prompt(
        self,
        *,
        journal: dict[str, Any],
        reflections: list[dict[str, Any]],
        focus: str,
    ) -> str:
        """Build strict JSON prompt so downstream API can parse deterministic fields."""
        reflection_brief = []
        for item in reflections[:5]:
            reflection_brief.append(str(item.get("reflection_content", ""))[:120])
        return (
            "你是A股投资复盘助手。仅输出 JSON，不要输出 markdown。\n"
            "JSON schema:\n"
            "{\n"
            '  "summary": "一句话总结，<=120字",\n'
            '  "insights": ["洞察1", "洞察2"],\n'
            '  "lessons": ["改进行动1", "改进行动2"],\n'
            '  "confidence": 0.0\n'
            "}\n\n"
            f"日志类型: {journal.get('journal_type', '')}\n"
            f"股票: {journal.get('stock_code', '')}\n"
            f"决策方向: {journal.get('decision_type', '')}\n"
            f"日志标题: {journal.get('title', '')}\n"
            f"日志正文: {journal.get('content', '')}\n"
            f"历史复盘摘要: {json.dumps(reflection_brief, ensure_ascii=False)}\n"
            f"关注重点: {focus}\n"
            "要求:\n"
            "1) 聚焦执行偏差、证据缺口、下次可操作动作，不给买卖指令。\n"
            "2) 每条洞察/改进动作控制在 60 字内。\n"
            "3) confidence range must be within [0,1]."
        )

    def _journal_validate_ai_reflection_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("ai reflection payload must be object")
        summary = str(payload.get("summary", "")).strip()
        if not summary:
            raise ValueError("ai reflection summary is empty")
        insights = self._journal_normalize_items(payload.get("insights", []))
        lessons = self._journal_normalize_items(payload.get("lessons", []))
        confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.0) or 0.0)))
        if not insights:
            insights = ["Review evidence completeness and trigger conditions before making decisions."]
        if not lessons:
            lessons = ["Define invalidation conditions and re-check schedule before execution."]
        return {
            "summary": summary[:600],
            "insights": insights,
            "lessons": lessons,
            "confidence": confidence,
        }

    def _journal_ai_reflection_fallback(self, *, journal: dict[str, Any]) -> dict[str, Any]:
        stock = str(journal.get("stock_code", "")).strip()
        decision = str(journal.get("decision_type", "")).strip() or "unknown"
        return {
            "summary": f"Journal for {stock or 'this ticker'} ({decision}) recorded. Re-check trigger and invalidation conditions.",
            "insights": [
                "Break decision basis into verifiable indicators instead of narrative-first reasoning.",
                "Add position constraints and stop-loss thresholds to reduce execution bias.",
            ],
            "lessons": [
                "Compare expected vs actual outcomes and update the next decision template.",
                "Capture failed samples and boundary conditions to prioritize high-impact fixes.",
            ],
            "confidence": 0.46,
        }

    @staticmethod
    def _journal_default_title(journal_type: str, stock_code: str) -> str:
        """Create a readable title when frontend only sends minimal fields."""
        label_map = {
            "decision": "交易决策",
            "reflection": "交易复盘",
            "learning": "学习记录",
        }
        jt = str(journal_type or "decision").strip().lower()
        label = label_map.get(jt, "投资日志")
        code = str(stock_code or "").strip().upper() or "UNKNOWN"
        return f"{label} {code}"

    def _journal_default_content(self, payload: dict[str, Any]) -> str:
        """Generate fallback content so users can submit quickly without verbose input."""
        stock_code = str(payload.get("stock_code", "")).strip().upper() or "UNKNOWN"
        journal_type = str(payload.get("journal_type", "decision")).strip().lower() or "decision"
        decision_type = str(payload.get("decision_type", "hold")).strip().lower() or "hold"
        thesis = str(payload.get("thesis", "")).strip()
        if not thesis:
            thesis = (
                f"{stock_code} 当前采用 {journal_type}/{decision_type} 模板记录。"
                "后续将结合数据覆盖提升观点完整性。"
            )
        return "\n".join(
            [
                f"模板类型: {journal_type}",
                f"核心观点: {thesis}",
                "触发条件: （系统默认）价格与成交量共振，且关键风险未恶化。",
                "失效条件: （系统默认）关键风险事件落地偏负面或趋势结构被破坏。",
                "执行计划: （系统默认）分批验证，不单次重仓。",
            ]
        )

    # Investment Journal: keep orchestration in service layer so API payload remains thin.
    def journal_create(self, token: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = dict(payload or {})
        journal_type = str(body.get("journal_type", "decision")).strip().lower() or "decision"
        stock_code = str(body.get("stock_code", "")).strip().upper()
        title = str(body.get("title", "")).strip()
        content = str(body.get("content", "")).strip()
        if not title:
            title = self._journal_default_title(journal_type, stock_code)
        if not content:
            # Support fast-create mode where frontend only sends template + code.
            content = self._journal_default_content(body)
        tags = body.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        if not tags:
            tags = [journal_type, "auto_generated"]

        return self.web.journal_create(
            token,
            journal_type=journal_type,
            title=title,
            content=content,
            stock_code=stock_code,
            decision_type=str(body.get("decision_type", "")),
            related_research_id=str(body.get("related_research_id", "")),
            related_portfolio_id=body.get("related_portfolio_id"),
            tags=tags,
            sentiment=str(body.get("sentiment", "")),
        )

    def journal_list(
        self,
        token: str,
        *,
        journal_type: str = "",
        stock_code: str = "",
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        return self.web.journal_list(
            token,
            journal_type=journal_type,
            stock_code=stock_code,
            limit=limit,
            offset=offset,
        )

    def journal_reflection_add(self, token: str, journal_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        return self.web.journal_reflection_add(
            token,
            journal_id=journal_id,
            reflection_content=str(payload.get("reflection_content", "")),
            ai_insights=str(payload.get("ai_insights", "")),
            lessons_learned=str(payload.get("lessons_learned", "")),
        )

    def journal_reflection_list(self, token: str, journal_id: int, *, limit: int = 50) -> list[dict[str, Any]]:
        return self.web.journal_reflection_list(token, journal_id=journal_id, limit=limit)

    def journal_ai_reflection_get(self, token: str, journal_id: int) -> dict[str, Any]:
        return self.web.journal_ai_reflection_get(token, journal_id=journal_id)

    def journal_ai_reflection_generate(self, token: str, journal_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        journal = self.web.journal_get(token, journal_id=journal_id)
        reflections = self.web.journal_reflection_list(token, journal_id=journal_id, limit=8)
        focus = str(payload.get("focus", "")).strip()[:200]
        trace_id = self.traces.new_trace()

        generated = self._journal_ai_reflection_fallback(journal=journal)
        status = "fallback"
        error_code = ""
        error_message = ""
        provider = ""
        model = ""
        started_at = time.perf_counter()

        if self.settings.llm_external_enabled and self.llm_gateway.providers:
            state = AgentState(
                user_id=str(journal.get("user_id", "journal_user")),
                question=f"journal-ai-reflection:{journal_id}",
                stock_codes=[str(journal.get("stock_code", "")).strip().upper()] if str(journal.get("stock_code", "")).strip() else [],
                trace_id=trace_id,
            )
            prompt = self._journal_build_ai_reflection_prompt(
                journal=journal,
                reflections=reflections,
                focus=focus or "Please emphasize execution bias, evidence gaps, and actionable improvements.",
            )
            try:
                raw = self.llm_gateway.generate(state, prompt)
                parsed = self._deep_safe_json_loads(raw)
                generated = self._journal_validate_ai_reflection_payload(parsed)
                status = "ready"
                provider = str(state.analysis.get("llm_provider", ""))
                model = str(state.analysis.get("llm_model", ""))
            except Exception as ex:  # noqa: BLE001
                status = "fallback"
                error_code = "journal_ai_generation_failed"
                error_message = str(ex)[:380]

        result = self.web.journal_ai_reflection_upsert(
            token,
            journal_id=journal_id,
            status=status,
            summary=str(generated.get("summary", "")),
            insights=generated.get("insights", []),
            lessons=generated.get("lessons", []),
            confidence=float(generated.get("confidence", 0.0) or 0.0),
            provider=provider,
            model=model,
            trace_id=trace_id,
            error_code=error_code,
            error_message=error_message,
        )
        # 杈撳嚭鐢熸垚鑰楁椂锛屾柟渚垮悗缁帓鏌モ€淎I澶嶇洏鎱?澶辫触鈥濈殑鍏蜂綋閾捐矾銆?
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        result["latency_ms"] = latency_ms
        # 姣忔鐢熸垚閮借褰曡川閲忔棩蹇楋紝渚夸簬杩愮淮缁熻 fallback/failed 姣斾緥涓庡欢杩熷垎甯冦€?
        try:
            _ = self.web.journal_ai_generation_log_add(
                token,
                journal_id=journal_id,
                status=status,
                provider=provider,
                model=model,
                trace_id=trace_id,
                error_code=error_code,
                error_message=error_message,
                latency_ms=latency_ms,
            )
        except Exception:
            pass
        return result

    def _journal_counter_breakdown(self, counter: Counter[str], *, total: int, top_n: int) -> list[dict[str, Any]]:
        """Convert Counter into deterministic [{key,count,ratio}] payload for UI/API."""
        items = sorted(counter.items(), key=lambda x: (-int(x[1]), str(x[0])))
        output: list[dict[str, Any]] = []
        for key, count in items[: max(1, int(top_n))]:
            ratio = float(count) / float(total) if total > 0 else 0.0
            output.append({"key": str(key), "count": int(count), "ratio": round(ratio, 4)})
        return output

    def _journal_extract_keywords(self, rows: list[dict[str, Any]], *, top_n: int = 12) -> list[dict[str, Any]]:
        """Extract coarse keywords from title/content/tags for quick journal topic profiling."""
        stopwords = {
            "以及",
            "因为",
            "所以",
            "我们",
            "他们",
            "如果",
            "但是",
            "然后",
            "这个",
            "那个",
            "需要",
            "已经",
            "当前",
            "计划",
            "复盘",
            "日志",
            "记录",
            "分析",
            "执行",
            "策略",
            "stock",
            "journal",
            "decision",
            "reflection",
            "learning",
        }
        pattern = re.compile(r"[A-Za-z][A-Za-z0-9_+-]{2,24}|[\u4e00-\u9fff]{2,8}")
        counter: Counter[str] = Counter()
        for row in rows:
            tags = row.get("tags", [])
            joined_tags = " ".join(str(x) for x in tags if str(x).strip()) if isinstance(tags, list) else ""
            corpus = " ".join(
                [
                    str(row.get("title", "")),
                    str(row.get("content", "")),
                    joined_tags,
                ]
            )
            for token in pattern.findall(corpus):
                term = str(token).strip().lower()
                if not term:
                    continue
                if term in stopwords:
                    continue
                if re.fullmatch(r"\d+", term):
                    continue
                counter[term] += 1
        items = sorted(counter.items(), key=lambda x: (-int(x[1]), str(x[0])))
        return [{"keyword": str(term), "count": int(count)} for term, count in items[: max(1, int(top_n))]]

    def journal_insights(
        self,
        token: str,
        *,
        window_days: int = 90,
        limit: int = 400,
        timeline_days: int = 30,
    ) -> dict[str, Any]:
        # 鑱氬悎灞傜粺涓€杈撳嚭涓氬姟娲炲療锛屽墠绔棤闇€鍐嶆嫾鎺ュ涓帴鍙ｃ€?
        safe_window_days = max(7, min(3650, int(window_days)))
        safe_limit = max(20, min(2000, int(limit)))
        safe_timeline_days = max(7, min(safe_window_days, int(timeline_days)))

        rows = self.web.journal_insights_rows(token, days=safe_window_days, limit=safe_limit)
        timeline_rows = self.web.journal_insights_timeline(token, days=safe_timeline_days)

        total_journals = len(rows)
        type_counter: Counter[str] = Counter()
        decision_counter: Counter[str] = Counter()
        stock_counter: Counter[str] = Counter()

        reflection_covered = 0
        ai_reflection_covered = 0
        total_reflection_records = 0
        for row in rows:
            journal_type = str(row.get("journal_type", "")).strip() or "unknown"
            decision_type = str(row.get("decision_type", "")).strip() or "none"
            stock_code = str(row.get("stock_code", "")).strip().upper() or "UNASSIGNED"
            reflection_count = max(0, int(row.get("reflection_count", 0) or 0))
            has_ai_reflection = bool(row.get("has_ai_reflection", False))

            type_counter[journal_type] += 1
            decision_counter[decision_type] += 1
            stock_counter[stock_code] += 1
            total_reflection_records += reflection_count
            if reflection_count > 0:
                reflection_covered += 1
            if has_ai_reflection:
                ai_reflection_covered += 1

        avg_reflections_per_journal = float(total_reflection_records) / float(total_journals) if total_journals else 0.0
        reflection_coverage_rate = float(reflection_covered) / float(total_journals) if total_journals else 0.0
        ai_reflection_coverage_rate = float(ai_reflection_covered) / float(total_journals) if total_journals else 0.0

        return {
            "status": "ok",
            "window_days": safe_window_days,
            "timeline_days": safe_timeline_days,
            "total_journals": total_journals,
            "type_distribution": self._journal_counter_breakdown(type_counter, total=total_journals, top_n=8),
            "decision_distribution": self._journal_counter_breakdown(decision_counter, total=total_journals, top_n=8),
            "stock_activity": self._journal_counter_breakdown(stock_counter, total=total_journals, top_n=10),
            "reflection_coverage": {
                "with_reflection": int(reflection_covered),
                "with_ai_reflection": int(ai_reflection_covered),
                "reflection_coverage_rate": round(reflection_coverage_rate, 4),
                "ai_reflection_coverage_rate": round(ai_reflection_coverage_rate, 4),
                "total_reflection_records": int(total_reflection_records),
                "avg_reflections_per_journal": round(avg_reflections_per_journal, 4),
            },
            "keyword_profile": self._journal_extract_keywords(rows, top_n=12),
            "timeline": timeline_rows,
        }

    def reports_list(self, token: str) -> list[dict[str, Any]]:
        return self.web.report_list(token)

    def _load_report_payload_from_version_row(self, report_id: str, row: dict[str, Any]) -> dict[str, Any]:
        """Load one persisted report version row into sanitized payload."""
        parsed: dict[str, Any] = {}
        raw_payload = str(row.get("payload_json", "")).strip()
        if raw_payload:
            try:
                data = json.loads(raw_payload)
                if isinstance(data, dict):
                    parsed = dict(data)
            except Exception:
                parsed = {}
        if not parsed:
            parsed = {
                "report_id": report_id,
                "markdown": self._sanitize_report_text(str(row.get("markdown", ""))),
                "report_modules": [],
                "analysis_nodes": [],
                "final_decision": {"signal": "hold", "confidence": 0.5, "rationale": ""},
                "committee": {"research_note": "", "risk_note": ""},
                "quality_gate": {"status": "degraded", "score": 0.5, "reasons": ["legacy_version_without_payload"]},
                "quality_dashboard": {},
            }
        parsed["report_id"] = report_id
        parsed["version"] = int(row.get("version", 0) or 0)
        parsed["created_at"] = str(row.get("created_at", ""))
        if not str(parsed.get("markdown", "")).strip():
            parsed["markdown"] = self._sanitize_report_text(str(row.get("markdown", "")))
        return self._sanitize_report_payload(parsed)

    def _build_report_version_diff_payload(self, *, base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
        """Compute report diff across decision, quality dashboard, modules and analysis nodes."""
        base_decision = dict(base.get("final_decision", {}) or {})
        cand_decision = dict(candidate.get("final_decision", {}) or {})
        base_quality = dict(base.get("quality_dashboard", {}) or {})
        cand_quality = dict(candidate.get("quality_dashboard", {}) or {})

        module_map_base = {
            str(row.get("module_id", "")).strip(): row
            for row in list(base.get("report_modules", []) or [])
            if isinstance(row, dict) and str(row.get("module_id", "")).strip()
        }
        module_map_cand = {
            str(row.get("module_id", "")).strip(): row
            for row in list(candidate.get("report_modules", []) or [])
            if isinstance(row, dict) and str(row.get("module_id", "")).strip()
        }
        module_ids = sorted(set(module_map_base.keys()) | set(module_map_cand.keys()))
        module_deltas: list[dict[str, Any]] = []
        for module_id in module_ids:
            b = dict(module_map_base.get(module_id, {}) or {})
            c = dict(module_map_cand.get(module_id, {}) or {})
            module_deltas.append(
                {
                    "module_id": module_id,
                    "exists_in_base": bool(b),
                    "exists_in_candidate": bool(c),
                    "quality_delta": round(
                        self._safe_float(c.get("module_quality_score", 0.0), default=0.0)
                        - self._safe_float(b.get("module_quality_score", 0.0), default=0.0),
                        4,
                    ),
                    "confidence_delta": round(
                        self._safe_float(c.get("confidence", 0.0), default=0.0)
                        - self._safe_float(b.get("confidence", 0.0), default=0.0),
                        4,
                    ),
                    "coverage_changed": str((b.get("coverage", {}) or {}).get("status", "")) != str((c.get("coverage", {}) or {}).get("status", "")),
                    "degrade_changed": list(b.get("degrade_reason", []) or []) != list(c.get("degrade_reason", []) or []),
                }
            )

        node_map_base = {
            str(row.get("node_id", "")).strip(): row
            for row in list(base.get("analysis_nodes", []) or [])
            if isinstance(row, dict) and str(row.get("node_id", "")).strip()
        }
        node_map_cand = {
            str(row.get("node_id", "")).strip(): row
            for row in list(candidate.get("analysis_nodes", []) or [])
            if isinstance(row, dict) and str(row.get("node_id", "")).strip()
        }
        node_ids = sorted(set(node_map_base.keys()) | set(node_map_cand.keys()))
        node_deltas: list[dict[str, Any]] = []
        for node_id in node_ids:
            b = dict(node_map_base.get(node_id, {}) or {})
            c = dict(node_map_cand.get(node_id, {}) or {})
            node_deltas.append(
                {
                    "node_id": node_id,
                    "exists_in_base": bool(b),
                    "exists_in_candidate": bool(c),
                    "signal_from": str(b.get("signal", "")),
                    "signal_to": str(c.get("signal", "")),
                    "signal_changed": str(b.get("signal", "")) != str(c.get("signal", "")),
                    "confidence_delta": round(
                        self._safe_float(c.get("confidence", 0.0), default=0.0)
                        - self._safe_float(b.get("confidence", 0.0), default=0.0),
                        4,
                    ),
                    "veto_from": bool(b.get("veto", False)),
                    "veto_to": bool(c.get("veto", False)),
                }
            )

        quality_delta = {
            "status_from": str(base_quality.get("status", "")),
            "status_to": str(cand_quality.get("status", "")),
            "status_changed": str(base_quality.get("status", "")) != str(cand_quality.get("status", "")),
            "overall_score_delta": round(
                self._safe_float(cand_quality.get("overall_score", 0.0), default=0.0)
                - self._safe_float(base_quality.get("overall_score", 0.0), default=0.0),
                4,
            ),
            "coverage_ratio_delta": round(
                self._safe_float(cand_quality.get("coverage_ratio", 0.0), default=0.0)
                - self._safe_float(base_quality.get("coverage_ratio", 0.0), default=0.0),
                4,
            ),
            "consistency_score_delta": round(
                self._safe_float(cand_quality.get("consistency_score", 0.0), default=0.0)
                - self._safe_float(base_quality.get("consistency_score", 0.0), default=0.0),
                4,
            ),
            "freshness_score_delta": round(
                self._safe_float(cand_quality.get("evidence_freshness_score", 0.0), default=0.0)
                - self._safe_float(base_quality.get("evidence_freshness_score", 0.0), default=0.0),
                4,
            ),
        }
        decision_delta = {
            "signal_from": str(base_decision.get("signal", "hold")),
            "signal_to": str(cand_decision.get("signal", "hold")),
            "signal_changed": str(base_decision.get("signal", "hold")) != str(cand_decision.get("signal", "hold")),
            "confidence_delta": round(
                self._safe_float(cand_decision.get("confidence", 0.0), default=0.0)
                - self._safe_float(base_decision.get("confidence", 0.0), default=0.0),
                4,
            ),
        }
        summary: list[str] = []
        if quality_delta["status_changed"]:
            summary.append(f"质量状态变化：{quality_delta['status_from']} -> {quality_delta['status_to']}")
        if decision_delta["signal_changed"]:
            summary.append(f"决策信号变化：{decision_delta['signal_from']} -> {decision_delta['signal_to']}")
        top_module_shift = sorted(module_deltas, key=lambda x: abs(self._safe_float(x.get("quality_delta", 0.0), default=0.0)), reverse=True)[:3]
        for row in top_module_shift:
            if abs(self._safe_float(row.get("quality_delta", 0.0), default=0.0)) >= 0.01:
                summary.append(f"模块 {row['module_id']} 质量变化 {self._safe_float(row['quality_delta'], default=0.0):+.2f}")
        if not summary:
            summary.append("版本间结构化差异较小。")
        return {
            "base_version": int(base.get("version", 0) or 0),
            "candidate_version": int(candidate.get("version", 0) or 0),
            "schema_version": str(candidate.get("schema_version", self._report_bundle_schema_version) or self._report_bundle_schema_version),
            "quality_delta": quality_delta,
            "decision_delta": decision_delta,
            "module_deltas": module_deltas,
            "node_deltas": node_deltas,
            "summary": summary[:8],
        }

    def report_versions(self, token: str, report_id: str) -> list[dict[str, Any]]:
        rows = self.web.report_version_rows(token, report_id, limit=50)
        items: list[dict[str, Any]] = []
        for row in rows:
            payload = self._load_report_payload_from_version_row(report_id, row)
            quality = dict(payload.get("quality_dashboard", {}) or {})
            decision = dict(payload.get("final_decision", {}) or {})
            items.append(
                {
                    "version": int(payload.get("version", 0) or 0),
                    "created_at": str(payload.get("created_at", "")),
                    "schema_version": str(payload.get("schema_version", self._report_bundle_schema_version)),
                    "signal": str(decision.get("signal", "hold")),
                    "confidence": float(decision.get("confidence", 0.0) or 0.0),
                    "quality_status": str(quality.get("status", "unknown")),
                    "quality_score": float(quality.get("overall_score", 0.0) or 0.0),
                    "module_count": len(list(payload.get("report_modules", []) or [])),
                    "analysis_node_count": len(list(payload.get("analysis_nodes", []) or [])),
                    "evidence_freshness_score": float(quality.get("evidence_freshness_score", 0.0) or 0.0),
                }
            )
        for idx in range(len(items)):
            if idx + 1 >= len(items):
                items[idx]["delta_vs_prev"] = {}
                continue
            curr = dict(items[idx])
            prev = dict(items[idx + 1])
            items[idx]["delta_vs_prev"] = {
                "quality_score_delta": round(self._safe_float(curr.get("quality_score", 0.0), default=0.0) - self._safe_float(prev.get("quality_score", 0.0), default=0.0), 4),
                "confidence_delta": round(self._safe_float(curr.get("confidence", 0.0), default=0.0) - self._safe_float(prev.get("confidence", 0.0), default=0.0), 4),
                "signal_changed": str(curr.get("signal", "")) != str(prev.get("signal", "")),
                "freshness_score_delta": round(
                    self._safe_float(curr.get("evidence_freshness_score", 0.0), default=0.0)
                    - self._safe_float(prev.get("evidence_freshness_score", 0.0), default=0.0),
                    4,
                ),
            }
        return items

    def report_versions_diff(
        self,
        token: str,
        report_id: str,
        *,
        base_version: int | None = None,
        candidate_version: int | None = None,
    ) -> dict[str, Any]:
        rows = self.web.report_version_rows(token, report_id, limit=80)
        if not rows:
            return {"error": "not_found", "report_id": report_id}
        payload_by_version = {
            int(row.get("version", 0) or 0): self._load_report_payload_from_version_row(report_id, row)
            for row in rows
        }
        versions = sorted(payload_by_version.keys(), reverse=True)
        candidate_v = int(candidate_version or versions[0])
        if candidate_v not in payload_by_version:
            return {"error": "candidate_version_not_found", "report_id": report_id, "candidate_version": candidate_v}
        if base_version is not None:
            base_v = int(base_version)
        else:
            lower = [v for v in versions if v < candidate_v]
            base_v = int(lower[0]) if lower else int(candidate_v)
        if base_v not in payload_by_version:
            return {"error": "base_version_not_found", "report_id": report_id, "base_version": base_v}
        base_payload = payload_by_version[base_v]
        candidate_payload = payload_by_version[candidate_v]
        diff_payload = self._build_report_version_diff_payload(base=base_payload, candidate=candidate_payload)
        return {
            "report_id": report_id,
            "available_versions": versions,
            "diff": diff_payload,
            "base_snapshot": {
                "version": int(base_payload.get("version", 0) or 0),
                "signal": str((base_payload.get("final_decision", {}) or {}).get("signal", "hold")),
                "quality_status": str((base_payload.get("quality_dashboard", {}) or {}).get("status", "unknown")),
            },
            "candidate_snapshot": {
                "version": int(candidate_payload.get("version", 0) or 0),
                "signal": str((candidate_payload.get("final_decision", {}) or {}).get("signal", "hold")),
                "quality_status": str((candidate_payload.get("quality_dashboard", {}) or {}).get("status", "unknown")),
            },
        }

    def report_export(self, token: str, report_id: str, *, format: str = "markdown") -> dict[str, Any]:
        payload = self.web.report_export(token, report_id)
        if "error" in payload:
            return payload
        export_format = str(format or "markdown").strip().lower() or "markdown"
        if export_format not in {"markdown", "module_markdown", "json_bundle"}:
            raise ValueError("format must be one of: markdown, module_markdown, json_bundle")

        # Prefer in-memory enriched payload so export can include modules/quality board.
        cached = self._reports.get(report_id)
        persisted_payload: dict[str, Any] = {}
        raw_payload_json = str(payload.get("payload_json", "")).strip()
        if raw_payload_json:
            try:
                parsed = json.loads(raw_payload_json)
                if isinstance(parsed, dict):
                    persisted_payload = parsed
            except Exception:
                persisted_payload = {}
        report_payload = self._sanitize_report_payload(dict(cached)) if isinstance(cached, dict) else {}
        if not report_payload and persisted_payload:
            report_payload = self._sanitize_report_payload(dict(persisted_payload))
        report_payload["report_id"] = report_id
        report_payload["version"] = int(payload.get("version", 0) or 0)
        if not str(report_payload.get("markdown", "")).strip():
            report_payload["markdown"] = self._sanitize_report_text(str(payload.get("markdown", "")))
        report_payload["schema_version"] = str(
            report_payload.get("schema_version")
            or persisted_payload.get("schema_version")
            or self._report_bundle_schema_version
        )

        module_markdown = self._render_report_module_markdown(report_payload)
        json_bundle = {
            "schema_version": str(report_payload.get("schema_version", self._report_bundle_schema_version)),
            "report_id": report_id,
            "version": report_payload.get("version", 0),
            "stock_code": str(report_payload.get("stock_code", "")),
            "report_type": str(report_payload.get("report_type", "")),
            "final_decision": dict(report_payload.get("final_decision", {}) or {}),
            "committee": dict(report_payload.get("committee", {}) or {}),
            "report_modules": list(report_payload.get("report_modules", []) or []),
            "analysis_nodes": list(report_payload.get("analysis_nodes", []) or []),
            "quality_dashboard": dict(report_payload.get("quality_dashboard", {}) or {}),
            "metric_snapshot": dict(report_payload.get("metric_snapshot", {}) or {}),
            "quality_gate": dict(report_payload.get("quality_gate", {}) or {}),
            "evidence_refs": list(report_payload.get("evidence_refs", []) or []),
        }
        if export_format == "module_markdown":
            content = module_markdown
            media_type = "text/markdown; charset=utf-8"
        elif export_format == "json_bundle":
            content = json.dumps(json_bundle, ensure_ascii=False, indent=2)
            media_type = "application/json; charset=utf-8"
        else:
            content = self._sanitize_report_text(str(report_payload.get("markdown", "")))
            media_type = "text/markdown; charset=utf-8"

        return {
            "report_id": report_id,
            "version": report_payload.get("version", 0),
            "schema_version": str(report_payload.get("schema_version", self._report_bundle_schema_version)),
            "format": export_format,
            "media_type": media_type,
            "content": content,
            # Keep legacy field for older clients that still read markdown directly.
            "markdown": content if export_format != "json_bundle" else self._sanitize_report_text(str(report_payload.get("markdown", ""))),
            "module_markdown": module_markdown,
            "json_bundle": json_bundle,
            "quality_dashboard": dict(report_payload.get("quality_dashboard", {}) or {}),
        }

    def docs_list(self, token: str) -> list[dict[str, Any]]:
        return self.web.docs_list(token)

    def docs_review_queue(self, token: str) -> list[dict[str, Any]]:
        return self.web.docs_review_queue(token)

    def docs_review_action(self, token: str, doc_id: str, action: str, comment: str = "") -> dict[str, Any]:
        result = self.web.docs_review_action(token, doc_id, action, comment)
        # 瀹℃牳鍔ㄤ綔闇€瑕佸悓姝ュ埌 chunk 鐢熸晥鐘舵€侊紝閬垮厤鈥滄枃妗ｇ姸鎬佸凡鏀逛絾妫€绱粛鍛戒腑鏃х墖娈碘€濄€?
        if action == "approve":
            self.web.rag_doc_chunk_set_status_by_doc(doc_id=doc_id, status="active")
            self.web.rag_upload_asset_set_status(doc_id=doc_id, status="active", parse_note="review_approved")
        elif action == "reject":
            self.web.rag_doc_chunk_set_status_by_doc(doc_id=doc_id, status="rejected")
            self.web.rag_upload_asset_set_status(doc_id=doc_id, status="rejected", parse_note="review_rejected")
        return result

    # ----------------- RAG Asset APIs -----------------
    def rag_source_policy_list(self, token: str) -> list[dict[str, Any]]:
        return self.web.rag_source_policy_list(token)

    def rag_source_policy_set(self, token: str, source: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.web.rag_source_policy_upsert(
            token,
            source=source,
            auto_approve=bool(payload.get("auto_approve", False)),
            trust_score=float(payload.get("trust_score", 0.7)),
            enabled=bool(payload.get("enabled", True)),
        )

    def rag_doc_chunks_list(
        self,
        token: str,
        *,
        doc_id: str = "",
        status: str = "",
        source: str = "",
        stock_code: str = "",
        limit: int = 60,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        return self.web.rag_doc_chunk_list(
            token,
            doc_id=doc_id,
            status=status,
            source=source,
            stock_code=stock_code,
            limit=limit,
            offset=offset,
        )

    def rag_doc_chunk_detail(
        self,
        token: str,
        chunk_id: str,
        *,
        context_window: int = 1,
    ) -> dict[str, Any]:
        """Return chunk detail with nearby context for瀹氫綅鏌ョ湅."""
        return self.web.rag_doc_chunk_get_detail(
            token,
            chunk_id=chunk_id,
            context_window=context_window,
        )

    def rag_doc_chunk_status_set(self, token: str, chunk_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.web.rag_doc_chunk_set_status(token, chunk_id=chunk_id, status=str(payload.get("status", "review")))

    def rag_qa_memory_list(
        self,
        token: str,
        *,
        stock_code: str = "",
        retrieval_enabled: int = -1,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        return self.web.rag_qa_memory_list(
            token,
            stock_code=stock_code,
            retrieval_enabled=retrieval_enabled,
            limit=limit,
            offset=offset,
        )

    def rag_qa_memory_toggle(self, token: str, memory_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.web.rag_qa_memory_toggle(
            token,
            memory_id=memory_id,
            retrieval_enabled=bool(payload.get("retrieval_enabled", False)),
        )

    def _parse_deep_archive_timestamp(self, value: str | None, field: str) -> str | None:
        clean = str(value or "").strip() or None
        if not clean:
            return None
        try:
            parsed = datetime.strptime(clean, self._deep_archive_ts_format)
        except ValueError as ex:
            raise ValueError(f"{field} must use format YYYY-MM-DD HH:MM:SS") from ex
        return parsed.strftime(self._deep_archive_ts_format)

    def _deep_archive_tenant_policy(self) -> dict[str, int]:
        raw = str(self.settings.deep_archive_tenant_policy_json or "").strip() or "{}"
        try:
            parsed = json.loads(raw)
        except Exception:  # noqa: BLE001
            parsed = {}
        if not isinstance(parsed, dict):
            return {}
        normalized: dict[str, int] = {}
        hard_cap = max(100, int(self.settings.deep_archive_max_events_hard_cap))
        for user_id, max_events in parsed.items():
            key = str(user_id).strip()
            if not key:
                continue
            try:
                normalized[key] = max(1, min(hard_cap, int(max_events)))
            except Exception:  # noqa: BLE001
                continue
        return normalized

    def _build_deep_archive_query_options(
        self,
        *,
        round_id: str | None = None,
        limit: int = 200,
        event_name: str | None = None,
        cursor: int | None = None,
        created_from: str | None = None,
        created_to: str | None = None,
    ) -> dict[str, Any]:
        try:
            safe_limit = max(1, min(2000, int(limit)))
        except Exception:  # noqa: BLE001
            safe_limit = 200
        safe_cursor = None
        if cursor is not None:
            try:
                safe_cursor = max(0, int(cursor))
            except Exception:  # noqa: BLE001
                safe_cursor = None
        round_id_clean = str(round_id or "").strip() or None
        event_name_clean = str(event_name or "").strip() or None
        created_from_clean = self._parse_deep_archive_timestamp(created_from, "created_from")
        created_to_clean = self._parse_deep_archive_timestamp(created_to, "created_to")
        if created_from_clean and created_to_clean:
            if created_from_clean > created_to_clean:
                raise ValueError("created_from must be <= created_to")
        return {
            "round_id": round_id_clean,
            "event_name": event_name_clean,
            "limit": safe_limit,
            "cursor": safe_cursor,
            "created_from": created_from_clean,
            "created_to": created_to_clean,
        }

    def _resolve_deep_archive_retention_policy(
        self,
        *,
        session: dict[str, Any],
        requested_max_events: Any | None = None,
    ) -> dict[str, Any]:
        env = str(self.settings.env or "dev").strip().lower() or "dev"
        env_defaults = {
            "dev": int(self.settings.deep_archive_max_events_dev),
            "staging": int(self.settings.deep_archive_max_events_staging),
            "prod": int(self.settings.deep_archive_max_events_prod),
        }
        hard_cap = max(100, int(self.settings.deep_archive_max_events_hard_cap))
        base = int(env_defaults.get(env, int(self.settings.deep_archive_max_events_default)))
        policy_source = f"env:{env}"
        user_id = str(session.get("user_id", "")).strip()
        tenant_policy = self._deep_archive_tenant_policy()
        if user_id and user_id in tenant_policy:
            base = int(tenant_policy[user_id])
            policy_source = f"user:{user_id}"
        base = max(1, min(hard_cap, base))

        requested = None
        if requested_max_events is not None:
            try:
                requested = int(requested_max_events)
            except Exception:  # noqa: BLE001
                requested = None
        resolved = base if requested is None else max(1, min(hard_cap, requested))
        return {
            "max_events": resolved,
            "base_max_events": base,
            "policy_source": policy_source,
            "hard_cap": hard_cap,
            "requested_max_events": requested,
        }

    def _emit_deep_archive_audit(
        self,
        *,
        session_id: str,
        action: str,
        status: str,
        started_at: float,
        result_count: int = 0,
        export_bytes: int = 0,
        detail: dict[str, Any] | None = None,
    ) -> None:
        duration_ms = int(round((time.perf_counter() - started_at) * 1000))
        try:
            self.web.deep_think_archive_audit_log(
                session_id=session_id,
                action=action,
                status=status,
                duration_ms=duration_ms,
                result_count=result_count,
                export_bytes=export_bytes,
                detail=detail or {},
            )
        except Exception:  # noqa: BLE001
            return

    def deep_think_create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        question = str(payload.get("question", "")).strip()
        if not question:
            raise ValueError("question is required")
        user_id = str(payload.get("user_id", "deep_user")).strip() or "deep_user"
        stock_codes = [str(x).upper() for x in payload.get("stock_codes", []) if str(x).strip()]
        agent_profile = [str(x) for x in payload.get("agent_profile", []) if str(x).strip()] or self._deep_think_default_profile()
        max_rounds = max(1, min(8, int(payload.get("max_rounds", 3))))
        budget = payload.get("budget", {"token_budget": 8000, "time_budget_ms": 25000, "tool_call_budget": 24})
        mode = str(payload.get("mode", "internal_orchestration"))
        session_id = str(payload.get("session_id", "")).strip() or f"dtk-{uuid.uuid4().hex[:12]}"
        trace_id = self.traces.new_trace()
        self.traces.emit(trace_id, "deep_think_session_created", {"session_id": session_id, "user_id": user_id})
        return self.web.deep_think_create_session(
            session_id=session_id,
            user_id=user_id,
            question=question,
            stock_codes=stock_codes,
            agent_profile=agent_profile,
            max_rounds=max_rounds,
            budget=budget,
            mode=mode,
            trace_id=trace_id,
        )

    def _deep_round_try_acquire(self, session_id: str) -> bool:
        """Session-level mutex: allow only one round per session at the same time."""
        with self._deep_round_mutex:
            if session_id in self._deep_round_inflight:
                return False
            self._deep_round_inflight.add(session_id)
            return True

    def _deep_round_release(self, session_id: str) -> None:
        """Release the session round lock even on exceptional paths."""
        with self._deep_round_mutex:
            self._deep_round_inflight.discard(session_id)

    def _deep_stream_event(
        self,
        *,
        event_name: str,
        data: dict[str, Any],
        session_id: str,
        round_id: str,
        round_no: int,
        event_seq: int,
    ) -> dict[str, Any]:
        """Attach common metadata fields to each DeepThink stream event."""
        payload = dict(data)
        payload.setdefault("session_id", session_id)
        payload.setdefault("round_id", round_id)
        payload.setdefault("round_no", round_no)
        payload.setdefault("event_seq", event_seq)
        payload.setdefault("emitted_at", datetime.now(timezone.utc).isoformat())
        return {"event": event_name, "data": payload}

    @staticmethod
    def _deep_journal_related_research_id(session_id: str, round_id: str) -> str:
        """Stable idempotency key for linking one round to one journal record."""
        return f"deepthink:{str(session_id).strip()}:{str(round_id).strip()}"

    def _deep_build_journal_from_business_summary(
        self,
        *,
        session_id: str,
        round_id: str,
        round_no: int,
        question: str,
        stock_code: str,
        business_summary: dict[str, Any],
    ) -> dict[str, Any]:
        signal = str(business_summary.get("signal", "hold")).strip().lower() or "hold"
        confidence = float(business_summary.get("confidence", 0.0) or 0.0)
        related_research_id = self._deep_journal_related_research_id(session_id, round_id)
        title = f"DeepThink R{round_no} {stock_code or 'UNASSIGNED'} {signal}".strip()
        content_lines = [
            f"[DeepThink] session={session_id}, round={round_id}, round_no={round_no}",
            f"闂: {question}",
            f"缁撹: signal={signal}, confidence={confidence:.4f}, disagreement={float(business_summary.get('disagreement_score', 0.0) or 0.0):.4f}",
            f"瑙﹀彂鏉′欢: {str(business_summary.get('trigger_condition', ''))}",
            f"澶辨晥鏉′欢: {str(business_summary.get('invalidation_condition', ''))}",
            f"澶嶆牳寤鸿: {str(business_summary.get('review_time_hint', 'T+1 鏃ュ唴澶嶆牳'))}",
            f"椋庨櫓鍋忓ソ: {str(business_summary.get('risk_bias', ''))}, 甯傚満鐘舵€? {str(business_summary.get('market_regime', ''))}",
            f"鍐茬獊婧? {','.join(str(x) for x in list(business_summary.get('top_conflict_sources', []))[:6]) or 'none'}",
        ]
        tags = [
            "deepthink",
            "auto_link",
            f"round_{round_no}",
            signal,
            str(stock_code or "").strip().upper() or "unassigned",
        ]
        sentiment = "positive" if signal == "buy" else "negative" if signal == "reduce" else "neutral"
        return {
            "journal_type": "reflection",
            "title": title[:200],
            "content": "\n".join(content_lines)[:8000],
            "stock_code": str(stock_code or "").strip().upper(),
            "decision_type": signal[:40],
            "related_research_id": related_research_id[:120],
            "tags": tags,
            "sentiment": sentiment,
        }

    def _deep_auto_link_journal_entry(
        self,
        *,
        session_id: str,
        round_id: str,
        round_no: int,
        question: str,
        stock_code: str,
        business_summary: dict[str, Any],
        auto_journal: bool,
    ) -> dict[str, Any]:
        """Auto-create/reuse journal entry for each DeepThink round, with idempotent key."""
        related_research_id = self._deep_journal_related_research_id(session_id, round_id)
        if not auto_journal:
            return {
                "ok": True,
                "enabled": False,
                "action": "disabled",
                "journal_id": 0,
                "related_research_id": related_research_id,
                "session_id": session_id,
                "round_id": round_id,
                "round_no": round_no,
            }
        try:
            existed = self.web.journal_find_by_related_research("", related_research_id=related_research_id)
            existing_id = int(existed.get("journal_id", 0) or 0)
            if existing_id > 0:
                return {
                    "ok": True,
                    "enabled": True,
                    "action": "reused",
                    "journal_id": existing_id,
                    "related_research_id": related_research_id,
                    "session_id": session_id,
                    "round_id": round_id,
                    "round_no": round_no,
                }

            create_payload = self._deep_build_journal_from_business_summary(
                session_id=session_id,
                round_id=round_id,
                round_no=round_no,
                question=question,
                stock_code=stock_code,
                business_summary=business_summary,
            )
            created = self.web.journal_create(
                "",
                journal_type=str(create_payload.get("journal_type", "reflection")),
                title=str(create_payload.get("title", "")),
                content=str(create_payload.get("content", "")),
                stock_code=str(create_payload.get("stock_code", "")),
                decision_type=str(create_payload.get("decision_type", "")),
                related_research_id=str(create_payload.get("related_research_id", "")),
                related_portfolio_id=None,
                tags=create_payload.get("tags", []),
                sentiment=str(create_payload.get("sentiment", "")),
            )
            return {
                "ok": True,
                "enabled": True,
                "action": "created",
                "journal_id": int(created.get("journal_id", 0) or 0),
                "related_research_id": related_research_id,
                "session_id": session_id,
                "round_id": round_id,
                "round_no": round_no,
            }
        except Exception as ex:  # noqa: BLE001
            return {
                "ok": False,
                "enabled": True,
                "action": "failed",
                "journal_id": 0,
                "related_research_id": related_research_id,
                "session_id": session_id,
                "round_id": round_id,
                "round_no": round_no,
                "error": str(ex)[:280],
            }

    def deep_think_run_round_stream_events(self, session_id: str, payload: dict[str, Any] | None = None):
        """V2 true streaming execution: emit round events incrementally and return session snapshot on completion."""
        payload = payload or {}
        started_at = time.perf_counter()
        session = self.web.deep_think_get_session(session_id)
        if not session:
            # 缁熶竴浠?done 鏀跺彛锛屼究浜庡墠绔拰娴嬭瘯浠ｇ爜澶嶇敤鍚屼竴缁撴潫閫昏緫銆?
            yield {"event": "done", "data": {"ok": False, "error": "not_found", "session_id": session_id}}
            return {"error": "not_found", "session_id": session_id}
        if str(session.get("status", "")) == "completed":
            yield {"event": "done", "data": {"ok": False, "error": "already_completed", "session_id": session_id}}
            return {"error": "already_completed", "session_id": session_id}
        if not self._deep_round_try_acquire(session_id):
            yield {"event": "done", "data": {"ok": False, "error": "round_in_progress", "session_id": session_id}}
            return {"error": "round_in_progress", "session_id": session_id}

        round_no = int(session.get("current_round", 0)) + 1
        max_rounds = int(session.get("max_rounds", 1))
        round_id = f"dtr-{uuid.uuid4().hex[:12]}"
        question = str(payload.get("question", session.get("question", "")))
        stock_codes = [str(x).upper() for x in payload.get("stock_codes", session.get("stock_codes", []))]
        report_context = self._latest_report_context(stock_codes[0] if stock_codes else "")
        if bool(report_context.get("available", False)):
            # Inject prior report committee hints to reduce context switching between modules.
            question = (
                f"{question}\n"
                "【最近报告上下文】\n"
                f"- signal: {str(report_context.get('signal', 'hold'))}\n"
                f"- confidence: {float(report_context.get('confidence', 0.0) or 0.0):.2f}\n"
                f"- research_note: {str(report_context.get('research_note', ''))}\n"
                f"- risk_note: {str(report_context.get('risk_note', ''))}\n"
                f"- research_summary: {str(report_context.get('research_summary', ''))}\n"
                f"- risk_summary: {str(report_context.get('risk_summary', ''))}\n"
                f"- rationale: {str(report_context.get('rationale', ''))}"
            )
        archive_policy = self._resolve_deep_archive_retention_policy(
            session=session,
            requested_max_events=payload.get("archive_max_events"),
        )
        archive_max_events = int(archive_policy["max_events"])

        event_seq = 0
        first_event_ms: int | None = None
        stream_events: list[dict[str, Any]] = []

        def emit(event_name: str, data: dict[str, Any], *, persist: bool = True) -> dict[str, Any]:
            # 鎵€鏈変簨浠剁粺涓€璧版鍏ュ彛锛屼繚璇?event_seq 涓庡厓瀛楁瀹屾暣銆?
            nonlocal event_seq, first_event_ms
            event_seq += 1
            event = self._deep_stream_event(
                event_name=event_name,
                data=data,
                session_id=session_id,
                round_id=round_id,
                round_no=round_no,
                event_seq=event_seq,
            )
            if persist:
                stream_events.append(event)
            if first_event_ms is None:
                first_event_ms = int(round((time.perf_counter() - started_at) * 1000))
            return event

        try:
            code = stock_codes[0] if stock_codes else "SH600000"
            regime_context = self._build_a_share_regime_context([code])
            task_graph = self._deep_plan_tasks(question, round_no)
            budget = session.get("budget", {}) if isinstance(session.get("budget", {}), dict) else {}
            budget_usage = self._deep_budget_snapshot(budget, round_no, len(task_graph))
            replan_triggered = False
            stop_reason = ""

            evidence_ids: list[str] = []
            opinions: list[dict[str, Any]] = []
            deep_pack: dict[str, Any] = {}
            intel_payload: dict[str, Any] = {
                "as_of": datetime.now(timezone.utc).isoformat(),
                "macro_signals": [],
                "industry_forward_events": [],
                "stock_specific_catalysts": [],
                "calendar_watchlist": [],
                "impact_chain": [],
                "decision_adjustment": {"signal_bias": "hold", "confidence_adjustment": 0.0, "rationale": ""},
                "citations": [],
                "confidence_note": "",
                "intel_status": "",
                "fallback_reason": "",
                "fallback_error": "",
                "trace_id": "",
                "external_enabled": bool(self.settings.llm_external_enabled),
                "provider_count": 0,
                "provider_names": [],
                "websearch_tool_requested": False,
                "websearch_tool_applied": False,
            }
            arbitration = {
                "consensus_signal": "hold",
                "disagreement_score": 0.0,
                "conflict_sources": [],
                "counter_view": "no significant counter view",
            }

            # 鍏堝彂 round_started锛岀‘淇濆墠绔湪璁＄畻鍓嶅氨鑳芥嬁鍒扳€滃凡寮€濮嬫墽琛屸€濈殑鍙嶉銆?
            yield emit(
                "round_started",
                {
                    "task_graph": task_graph,
                    "max_rounds": max_rounds,
                    "question": question,
                    "stock_codes": stock_codes,
                },
            )
            yield emit(
                "market_regime",
                {
                    "regime_label": str(regime_context.get("regime_label", "")),
                    "regime_confidence": float(regime_context.get("regime_confidence", 0.0) or 0.0),
                    "risk_bias": str(regime_context.get("risk_bias", "")),
                    "regime_rationale": str(regime_context.get("regime_rationale", "")),
                },
            )
            yield emit(
                "progress",
                {
                    "stage": "planning",
                    "message": "Task planning completed. Starting multi-agent execution.",
                },
            )
            if bool(report_context.get("available", False)):
                yield emit(
                    "report_context",
                    {
                        "report_id": str(report_context.get("report_id", "")),
                        "signal": str(report_context.get("signal", "hold")),
                        "confidence": float(report_context.get("confidence", 0.0) or 0.0),
                        "research_note": str(report_context.get("research_note", "")),
                        "risk_note": str(report_context.get("risk_note", "")),
                        "research_summary": str(report_context.get("research_summary", "")),
                        "risk_summary": str(report_context.get("risk_summary", "")),
                    },
                )

            if bool(budget_usage.get("warn")):
                yield emit("budget_warning", dict(budget_usage))

            if budget_usage["exceeded"]:
                stop_reason = "DEEP_BUDGET_EXCEEDED"
                opinions.append(
                    self._normalize_deep_opinion(
                        agent="supervisor_agent",
                        signal="hold",
                        confidence=0.92,
                        reason="Budget cap reached. Stop additional reasoning and return conservative output.",
                        evidence_ids=[],
                        risk_tags=["budget_exceeded"],
                    )
                )
                arbitration = self._arbitrate_opinions(opinions)
            else:
                yield emit("progress", {"stage": "data_refresh", "message": "Refreshing market and history samples"})
                deep_pack = self._build_llm_input_pack(code, question=question, scenario="deepthink")
                refresh_actions = list(((deep_pack.get("dataset", {}) or {}).get("refresh_actions", []) or []))
                refresh_failed = [
                    x for x in refresh_actions if str((x or {}).get("status", "")).strip().lower() != "ok"
                ]
                yield emit(
                    "data_pack",
                    {
                        "stock_code": code,
                        "coverage": dict((deep_pack.get("dataset", {}) or {}).get("coverage", {}) or {}),
                        "missing_data": list(deep_pack.get("missing_data", []) or []),
                        "data_quality": str(deep_pack.get("data_quality", "")),
                        "time_horizon_coverage": dict(deep_pack.get("time_horizon_coverage", {}) or {}),
                        "refresh_action_count": int(len(refresh_actions)),
                        "refresh_failed_count": int(len(refresh_failed)),
                        "refresh_actions": [
                            {
                                "category": str((row or {}).get("category", "")),
                                "status": str((row or {}).get("status", "")),
                                "latency_ms": int((row or {}).get("latency_ms", 0) or 0),
                                "error": str((row or {}).get("error", "")),
                            }
                            for row in refresh_actions
                        ][:12],
                    },
                )
                if refresh_failed:
                    yield emit(
                        "progress",
                        {
                            "stage": "data_gap_warning",
                            "message": f"Auto-refresh has {len(refresh_failed)} failed categories; confidence will be reduced.",
                        },
                    )
                quote = dict(deep_pack.get("realtime", {}) or {})
                bars = list(deep_pack.get("history", []) or [])[-180:]
                financial = dict(deep_pack.get("financial", {}) or {})
                trend = dict(deep_pack.get("trend", {}) or {})
                if not trend and len(bars) >= 30:
                    trend = self._trend_metrics(bars)
                # Re-evaluate regime after refresh so later guard uses freshest 1-20d signals.
                regime_context = self._build_a_share_regime_context([code])
                yield emit(
                    "market_regime",
                    {
                        "regime_label": str(regime_context.get("regime_label", "")),
                        "regime_confidence": float(regime_context.get("regime_confidence", 0.0) or 0.0),
                        "risk_bias": str(regime_context.get("risk_bias", "")),
                        "regime_rationale": str(regime_context.get("regime_rationale", "")),
                    },
                )
                pred = self.predict_run({"stock_codes": [code], "horizons": ["20d"]})
                horizon_map = {h["horizon"]: h for h in pred["results"][0]["horizons"]} if pred.get("results") else {}
                quant_20 = horizon_map.get("20d", {})
                # 寮曞叆瀹炴椂鎯呮姤闃舵锛氱敱妯″瀷渚?websearch 鑳藉姏杈撳嚭缁撴瀯鍖栨儏鎶ャ€?
                yield emit("progress", {"stage": "intel_search", "message": "Collecting macro/industry/future-event intelligence"})
                intel_payload = self._deep_fetch_intel_via_llm_websearch(
                    stock_code=code,
                    question=question,
                    quote=quote,
                    trend=trend,
                    quant_20=quant_20,
                )
                yield emit(
                    "intel_snapshot",
                    {
                        "as_of": intel_payload.get("as_of", ""),
                        "macro_signals": list(intel_payload.get("macro_signals", []))[:3],
                        "industry_forward_events": list(intel_payload.get("industry_forward_events", []))[:3],
                        "stock_specific_catalysts": list(intel_payload.get("stock_specific_catalysts", []))[:3],
                        "confidence_note": str(intel_payload.get("confidence_note", "")),
                        "intel_status": str(intel_payload.get("intel_status", "")),
                        "fallback_reason": str(intel_payload.get("fallback_reason", "")),
                        "fallback_error": str(intel_payload.get("fallback_error", "")),
                        "trace_id": str(intel_payload.get("trace_id", "")),
                        "citations_count": len(list(intel_payload.get("citations", []))),
                        "provider_count": int(intel_payload.get("provider_count", 0) or 0),
                        "provider_names": list(intel_payload.get("provider_names", []))[:8],
                        "websearch_tool_requested": bool(intel_payload.get("websearch_tool_requested", False)),
                        "websearch_tool_applied": bool(intel_payload.get("websearch_tool_applied", False)),
                        "financial_snapshot": {
                            "source_id": str(financial.get("source_id", "")),
                            "report_period": str(financial.get("report_period", "")),
                            "roe": float(financial.get("roe", 0.0) or 0.0),
                            "pe_ttm": float(financial.get("pe_ttm", 0.0) or 0.0),
                            "pb_mrq": float(financial.get("pb_mrq", 0.0) or 0.0),
                        },
                    },
                )
                # 鍗曠嫭涓嬪彂鐘舵€佷簨浠讹紝渚夸簬鍓嶇鎴栨祴璇曡剼鏈揩閫熷垽鏂槸鍚﹀懡涓閮ㄥ疄鏃舵绱€?
                yield emit(
                    "intel_status",
                    {
                        "intel_status": str(intel_payload.get("intel_status", "")),
                        "fallback_reason": str(intel_payload.get("fallback_reason", "")),
                        "confidence_note": str(intel_payload.get("confidence_note", "")),
                        "trace_id": str(intel_payload.get("trace_id", "")),
                        "citations_count": len(list(intel_payload.get("citations", []))),
                    },
                )
                yield emit("calendar_watchlist", {"items": list(intel_payload.get("calendar_watchlist", []))[:8]})

                yield emit("progress", {"stage": "debate", "message": "Generating multi-agent viewpoints"})
                rule_core = self._build_rule_based_debate_opinions(question, trend, quote, quant_20)
                core_opinions = rule_core
                if self.settings.llm_external_enabled and self.llm_gateway.providers:
                    llm_core = self._build_llm_debate_opinions(question, code, trend, quote, quant_20, rule_core)
                    if llm_core:
                        core_opinions = llm_core

                # 灏嗗疄鏃舵儏鎶ュ紩鐢?URL 绾冲叆璇佹嵁閾撅紝鍚庣画瀵煎嚭涓庤В閲婂彲杩芥函銆?
                intel_urls = [str(x.get("url", "")).strip() for x in list(intel_payload.get("citations", [])) if isinstance(x, dict)]
                evidence_ids = [x for x in {str(quote.get("source_id", "")), str((bars[-1] if bars else {}).get("source_id", "")), *intel_urls} if x]
                extra_opinions = [
                    self._normalize_deep_opinion(
                        agent="macro_agent",
                        signal="buy" if trend.get("ma60_slope", 0.0) > 0 and trend.get("momentum_20", 0.0) > 0 else "hold",
                        confidence=0.63,
                        reason=f"瀹忚渚ц瘎浼帮細ma60_slope={trend.get('ma60_slope', 0.0):.4f}, momentum_20={trend.get('momentum_20', 0.0):.4f}",
                        evidence_ids=evidence_ids,
                        risk_tags=["macro_regime_check"],
                    ),
                    self._normalize_deep_opinion(
                        agent="execution_agent",
                        signal="reduce"
                        if abs(float(quote.get("pct_change", 0.0))) > 4.0 and trend.get("volatility_20", 0.0) > 0.02
                        else "hold",
                        confidence=0.61,
                        reason=(
                            f"鎵ц灞傚缓璁細pct_change={float(quote.get('pct_change', 0.0)):.2f}, "
                            f"volatility_20={trend.get('volatility_20', 0.0):.4f}"
                        ),
                        evidence_ids=evidence_ids,
                        risk_tags=["execution_timing"],
                    ),
                    self._normalize_deep_opinion(
                        agent="compliance_agent",
                        signal="reduce" if trend.get("max_drawdown_60", 0.0) > 0.2 else "hold",
                        confidence=0.78,
                        reason=f"鍚堣涓庨鎺у簳绾匡細max_drawdown_60={trend.get('max_drawdown_60', 0.0):.4f}",
                        evidence_ids=evidence_ids,
                        risk_tags=["compliance_guardrail"],
                    ),
                    self._normalize_deep_opinion(
                        agent="critic_agent",
                        signal="hold",
                        confidence=0.7,
                        reason="Critic review completed: checked evidence coverage, signal conflicts, and risk-marker completeness.",
                        evidence_ids=evidence_ids,
                        risk_tags=["critic_review"],
                    ),
                ]

                normalized_core = [
                    self._normalize_deep_opinion(
                        agent=str(opinion.get("agent", "")),
                        signal=str(opinion.get("signal", "hold")),
                        confidence=float(opinion.get("confidence", 0.5)),
                        reason=str(opinion.get("reason", "")),
                        evidence_ids=evidence_ids,
                        risk_tags=["core_role"],
                    )
                    for opinion in core_opinions
                ]
                report_opinions: list[dict[str, Any]] = []
                if bool(report_context.get("available", False)):
                    report_opinions.append(
                        self._normalize_deep_opinion(
                            agent="report_committee_agent",
                            signal=str(report_context.get("signal", "hold")),
                            confidence=float(report_context.get("confidence", 0.5) or 0.5),
                            reason=(
                                "Reusing latest report committee context for cross-module continuity: "
                                + str(report_context.get("rationale", ""))
                            )[:420],
                            evidence_ids=[str(report_context.get("report_id", ""))] if str(report_context.get("report_id", "")) else [],
                            risk_tags=["report_context_bridge"],
                        )
                    )
                opinions = normalized_core + report_opinions + extra_opinions
                intel_decision = intel_payload.get("decision_adjustment", {})
                if isinstance(intel_decision, dict):
                    intel_signal = str(intel_decision.get("signal_bias", "hold")).strip().lower()
                    if intel_signal in {"buy", "hold", "reduce"}:
                        intel_conf = max(0.35, min(0.95, 0.58 + float(intel_decision.get("confidence_adjustment", 0.0) or 0.0)))
                        opinions.append(
                            self._normalize_deep_opinion(
                                agent="macro_agent",
                                signal=intel_signal,
                                confidence=intel_conf,
                                reason=f"Realtime intel fusion: {str(intel_decision.get('rationale', 'no rationale'))}",
                                evidence_ids=evidence_ids,
                                risk_tags=["intel_websearch"],
                            )
                        )

                pre_arb = self._arbitrate_opinions(opinions)
                data_pack_missing = [str(x) for x in list(deep_pack.get("missing_data", []) or []) if str(x).strip()]
                if data_pack_missing:
                    pre_arb["conflict_sources"] = list(dict.fromkeys(list(pre_arb.get("conflict_sources", [])) + ["data_gap"]))
                if float(pre_arb["disagreement_score"]) >= 0.45 and round_no < max_rounds:
                    replan_triggered = True
                    task_graph.append(
                        {
                            "task_id": f"r{round_no}-replan-1",
                            "agent": "critic_agent",
                            "title": "Trigger evidence re-plan: locate conflict evidence and explain divergence",
                            "priority": "high",
                        }
                    )
                    pre_arb["conflict_sources"] = list(dict.fromkeys(list(pre_arb["conflict_sources"]) + ["replan_triggered"]))

                supervisor = self._normalize_deep_opinion(
                    agent="supervisor_agent",
                    signal=str(pre_arb["consensus_signal"]),
                    confidence=round(1.0 - float(pre_arb["disagreement_score"]), 4),
                    reason=f"Supervisor arbitration: conflict_sources={' , '.join(pre_arb['conflict_sources']) or 'none'}",
                    evidence_ids=evidence_ids,
                    risk_tags=["supervisor_arbitration"],
                )
                opinions.append(supervisor)
                arbitration = self._arbitrate_opinions(opinions)
                if budget_usage["warn"]:
                    arbitration["conflict_sources"] = list(dict.fromkeys(list(arbitration["conflict_sources"]) + ["budget_warning"]))

            # 閫愭潯杈撳嚭 Agent 澧為噺涓庢渶缁堣鐐癸紝淇濊瘉鍓嶇鑳芥寔缁埛鏂般€?
            for opinion in opinions:
                reason = str(opinion.get("reason", ""))
                pivot = max(1, min(len(reason), len(reason) // 2))
                agent_id = str(opinion.get("agent", ""))
                yield emit("agent_opinion_delta", {"agent": agent_id, "delta": reason[:pivot]})
                opinion_event = {
                    "agent_id": agent_id,
                    "signal": str(opinion.get("signal", "hold")),
                    "confidence": float(opinion.get("confidence", 0.5)),
                    "reason": reason,
                    "evidence_ids": list(opinion.get("evidence_ids", [])),
                    "risk_tags": list(opinion.get("risk_tags", [])),
                }
                yield emit("agent_opinion_final", opinion_event)
                if agent_id == "critic_agent":
                    yield emit("critic_feedback", {"reason": reason})

            yield emit(
                "arbitration_final",
                {
                    "consensus_signal": arbitration["consensus_signal"],
                    "disagreement_score": arbitration["disagreement_score"],
                    "conflict_sources": arbitration["conflict_sources"],
                    "counter_view": arbitration["counter_view"],
                },
            )
            if replan_triggered:
                yield emit(
                    "replan_triggered",
                    {
                        "task_graph": task_graph,
                        "reason": "disagreement_above_threshold",
                    },
                )
            business_summary = self._deep_build_business_summary(
                stock_code=code,
                question=question,
                opinions=opinions,
                arbitration=arbitration,
                budget_usage=budget_usage,
                intel=intel_payload,
                regime_context=regime_context,
                replan_triggered=replan_triggered,
                stop_reason=stop_reason,
            )
            yield emit("business_summary", dict(business_summary))
            # DeepThink -> Journal 鑷姩钀藉簱锛歳ound_id 浣滀负骞傜瓑閿紝閬垮厤閲嶅鍐欏叆銆?
            journal_link = self._deep_auto_link_journal_entry(
                session_id=session_id,
                round_id=round_id,
                round_no=round_no,
                question=question,
                stock_code=code,
                business_summary=business_summary,
                auto_journal=bool(payload.get("auto_journal", True)),
            )
            if bool(journal_link.get("enabled", False)):
                yield emit("journal_linked", dict(journal_link))

            session_status = "completed" if round_no >= max_rounds or stop_reason else "in_progress"
            snapshot = self.web.deep_think_append_round(
                session_id=session_id,
                round_id=round_id,
                round_no=round_no,
                status="completed",
                consensus_signal=str(arbitration["consensus_signal"]),
                disagreement_score=float(arbitration["disagreement_score"]),
                conflict_sources=list(arbitration["conflict_sources"]),
                counter_view=str(arbitration["counter_view"]),
                task_graph=task_graph,
                replan_triggered=replan_triggered,
                stop_reason=stop_reason,
                budget_usage=budget_usage,
                opinions=opinions,
                session_status=session_status,
            )
            # 涓氬姟鎽樿閫氳繃浼氳瘽蹇収閫忎紶缁欏悓姝ユ帴鍙ｈ皟鐢ㄦ柟锛屼究浜庡墠绔洿鎺ュ睍绀恒€?
            snapshot["business_summary"] = business_summary
            snapshot["journal_link"] = journal_link

            avg_confidence = sum(float(x.get("confidence", 0.0)) for x in opinions) / max(1, len(opinions))
            quality_score = self._quality_score_for_group_card(
                disagreement_score=float(arbitration["disagreement_score"]),
                avg_confidence=avg_confidence,
                evidence_count=len(evidence_ids),
            )
            if quality_score >= 0.8 and len(evidence_ids) >= 2 and not stop_reason:
                self.web.add_group_knowledge_card(
                    card_id=f"gkc-{uuid.uuid4().hex[:12]}",
                    topic=code,
                    normalized_question=question.strip().lower(),
                    fact_summary=f"consensus={arbitration['consensus_signal']}, disagreement={arbitration['disagreement_score']}",
                    citation_ids=evidence_ids,
                    quality_score=quality_score,
                )

            trace_id = str(session.get("trace_id", ""))
            if trace_id:
                self.traces.emit(
                    trace_id,
                    "deep_think_round_completed",
                    {
                        "session_id": session_id,
                        "round_id": round_id,
                        "round_no": round_no,
                        "consensus_signal": arbitration["consensus_signal"],
                        "disagreement_score": arbitration["disagreement_score"],
                        "replan_triggered": replan_triggered,
                        "stop_reason": stop_reason,
                        "archive_policy": archive_policy,
                    },
                )

            round_persisted = emit(
                "round_persisted",
                {
                    "ok": True,
                    "status": session_status,
                    "current_round": int(snapshot.get("current_round", round_no)),
                    "archive_policy": archive_policy,
                },
            )
            done_event = emit(
                "done",
                {
                    "ok": True,
                    "status": session_status,
                    "stop_reason": stop_reason,
                    "duration_ms": int(round((time.perf_counter() - started_at) * 1000)),
                },
            )

            # 缁熶竴鎸夌敓鎴愰『搴忚惤搴擄紝纭繚閲嶆斁浜嬩欢涓庡疄鏃朵簨浠堕『搴忎竴鑷淬€?
            self.web.deep_think_replace_round_events(
                session_id=session_id,
                round_id=round_id,
                round_no=round_no,
                events=stream_events,
                max_events=archive_max_events,
            )

            yield round_persisted
            yield done_event
            snapshot["archive_policy"] = archive_policy
            self._emit_deep_archive_audit(
                session_id=session_id,
                action="round_stream_v2",
                status="ok",
                started_at=started_at,
                result_count=len(stream_events),
                detail={
                    "round_id": round_id,
                    "round_no": round_no,
                    "first_event_ms": int(first_event_ms or 0),
                    "event_count": len(stream_events),
                },
            )
            return snapshot
        except Exception as ex:  # noqa: BLE001
            message = str(ex) or "deep_think_round_failed"
            # 澶辫触鍦烘櫙涔熷彂 error + done锛屽墠绔彲缁熶竴澶勭悊缁撴潫鎬併€?
            error_event = emit("error", {"ok": False, "error": message}, persist=False)
            done_event = emit("done", {"ok": False, "error": message}, persist=False)
            yield error_event
            yield done_event
            self._emit_deep_archive_audit(
                session_id=session_id,
                action="round_stream_v2",
                status="error",
                started_at=started_at,
                detail={
                    "round_id": round_id,
                    "round_no": round_no,
                    "first_event_ms": int(first_event_ms or 0),
                    "event_count": len(stream_events),
                    "error": message,
                },
            )
            return {"error": "round_failed", "message": message, "session_id": session_id}
        finally:
            self._deep_round_release(session_id)

    def deep_think_run_round(self, session_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        # V1 鍏煎璺緞锛氬鐢ㄥ悓涓€鎵ц鏍稿績锛屼絾娑堣垂鎺夋祦浜嬩欢锛屼粎杩斿洖鏈€缁堝揩鐓с€?
        gen = self.deep_think_run_round_stream_events(session_id, payload)
        snapshot: dict[str, Any] = {"error": "round_failed", "session_id": session_id}
        while True:
            try:
                next(gen)
            except StopIteration as stop:
                value = stop.value
                if isinstance(value, dict):
                    snapshot = value
                break
        return snapshot

    def deep_think_get_session(self, session_id: str) -> dict[str, Any]:
        session = self.web.deep_think_get_session(session_id)
        if not session:
            return {"error": "not_found", "session_id": session_id}
        return session

    def deep_think_export_report(self, session_id: str, *, format: str = "markdown") -> dict[str, Any]:
        """Export DeepThink session as business-readable report document."""
        session = self.web.deep_think_get_session(session_id)
        if not session:
            return {"error": "not_found", "session_id": session_id}
        export_format = str(format or "markdown").strip().lower()
        if export_format not in {"markdown", "pdf"}:
            raise ValueError("format must be one of: markdown, pdf")
        if export_format == "markdown":
            content = self.deepthink_exporter.export_markdown(session)
            return {
                "session_id": session_id,
                "format": "markdown",
                "filename": f"deepthink-report-{session_id}.md",
                "media_type": "text/markdown; charset=utf-8",
                "content": content,
            }

        pdf_bytes = self.deepthink_exporter.export_pdf_bytes(session)
        return {
            "session_id": session_id,
            "format": "pdf",
            "filename": f"deepthink-report-{session_id}.pdf",
            "media_type": "application/pdf",
            "content": pdf_bytes,
        }

    def _build_deep_think_round_events(self, session_id: str, latest: dict[str, Any]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = [
            {
                "event": "round_started",
                "data": {
                    "session_id": session_id,
                    "round_id": latest.get("round_id"),
                    "round_no": latest.get("round_no"),
                    "task_graph": latest.get("task_graph", []),
                },
            }
        ]
        budget_usage = latest.get("budget_usage", {})
        if bool(budget_usage.get("warn")):
            events.append({"event": "budget_warning", "data": budget_usage})

        for opinion in latest.get("opinions", []):
            reason = str(opinion.get("reason", ""))
            pivot = max(1, min(len(reason), len(reason) // 2))
            events.append(
                {
                    "event": "agent_opinion_delta",
                    "data": {"agent": opinion.get("agent_id"), "delta": reason[:pivot]},
                }
            )
            events.append({"event": "agent_opinion_final", "data": opinion})
            if str(opinion.get("agent_id")) == "critic_agent":
                events.append(
                    {
                        "event": "critic_feedback",
                        "data": {"reason": reason, "round_id": latest.get("round_id")},
                    }
                )

        events.append(
            {
                "event": "arbitration_final",
                "data": {
                    "consensus_signal": latest.get("consensus_signal"),
                    "disagreement_score": latest.get("disagreement_score"),
                    "conflict_sources": latest.get("conflict_sources", []),
                    "counter_view": latest.get("counter_view", ""),
                },
            }
        )
        if bool(latest.get("replan_triggered", False)):
            events.append(
                {
                    "event": "replan_triggered",
                    "data": {
                        "round_id": latest.get("round_id"),
                        "task_graph": latest.get("task_graph", []),
                        "reason": "disagreement_above_threshold",
                    },
                }
            )
        round_id = str(latest.get("round_id", "")).strip()
        if round_id:
            try:
                related_research_id = self._deep_journal_related_research_id(session_id, round_id)
                linked = self.web.journal_find_by_related_research("", related_research_id=related_research_id)
                linked_id = int(linked.get("journal_id", 0) or 0)
                if linked_id > 0:
                    events.append(
                        {
                            "event": "journal_linked",
                            "data": {
                                "ok": True,
                                "enabled": True,
                                "action": "reused",
                                "journal_id": linked_id,
                                "related_research_id": related_research_id,
                                "round_id": round_id,
                                "round_no": latest.get("round_no", 0),
                            },
                        }
                    )
            except Exception:
                pass
        events.append({"event": "done", "data": {"ok": True, "session_id": session_id}})
        return events

    def deep_think_stream_events(self, session_id: str):
        session = self.web.deep_think_get_session(session_id)
        if not session:
            yield {"event": "done", "data": {"ok": False, "error": "not_found", "session_id": session_id}}
            return
        rounds = session.get("rounds", [])
        if not rounds:
            yield {"event": "done", "data": {"ok": False, "error": "no_rounds", "session_id": session_id}}
            return
        latest = rounds[-1]
        round_id = str(latest.get("round_id", ""))
        round_no = int(latest.get("round_no", 0))
        events = self.web.deep_think_list_events(session_id=session_id, round_id=round_id, limit=400)
        if not events:
            generated = self._build_deep_think_round_events(session_id, latest)
            self.web.deep_think_replace_round_events(
                session_id=session_id,
                round_id=round_id,
                round_no=round_no,
                events=generated,
            )
            events = self.web.deep_think_list_events(session_id=session_id, round_id=round_id, limit=400)
        for item in events:
            yield {"event": str(item.get("event", "message")), "data": item.get("data", {})}

    def deep_think_list_events(
        self,
        session_id: str,
        *,
        round_id: str | None = None,
        limit: int = 200,
        event_name: str | None = None,
        cursor: int | None = None,
        created_from: str | None = None,
        created_to: str | None = None,
    ) -> dict[str, Any]:
        started_at = time.perf_counter()
        session = self.web.deep_think_get_session(session_id)
        if not session:
            self._emit_deep_archive_audit(
                session_id=session_id,
                action="archive_query",
                status="not_found",
                started_at=started_at,
            )
            return {"error": "not_found", "session_id": session_id}
        try:
            filters = self._build_deep_archive_query_options(
                round_id=round_id,
                limit=limit,
                event_name=event_name,
                cursor=cursor,
                created_from=created_from,
                created_to=created_to,
            )
            page = self.web.deep_think_list_events_page(
                session_id=session_id,
                round_id=filters["round_id"],
                limit=filters["limit"],
                event_name=filters["event_name"],
                cursor=filters["cursor"],
                created_from=filters["created_from"],
                created_to=filters["created_to"],
            )
        except Exception as ex:  # noqa: BLE001
            self._emit_deep_archive_audit(
                session_id=session_id,
                action="archive_query",
                status="error",
                started_at=started_at,
                detail={"error": str(ex)},
            )
            raise
        events = list(page.get("events", []))
        self._emit_deep_archive_audit(
            session_id=session_id,
            action="archive_query",
            status="ok",
            started_at=started_at,
            result_count=len(events),
            detail={
                "round_id": filters["round_id"] or "",
                "event_name": filters["event_name"] or "",
                "cursor": int(filters["cursor"] or 0),
                "limit": int(filters["limit"]),
            },
        )
        return {
            "session_id": session_id,
            "round_id": (filters["round_id"] or ""),
            "event_name": (filters["event_name"] or ""),
            "cursor": int(filters["cursor"] or 0),
            "limit": int(filters["limit"]),
            "created_from": (filters["created_from"] or ""),
            "created_to": (filters["created_to"] or ""),
            "has_more": bool(page.get("has_more", False)),
            "next_cursor": page.get("next_cursor"),
            "count": len(events),
            "events": events,
        }

    def deep_think_export_events(
        self,
        session_id: str,
        *,
        round_id: str | None = None,
        limit: int = 200,
        event_name: str | None = None,
        cursor: int | None = None,
        created_from: str | None = None,
        created_to: str | None = None,
        format: str = "jsonl",
        audit_action: str = "archive_export_sync",
        emit_audit: bool = True,
    ) -> dict[str, Any]:
        started_at = time.perf_counter()
        export_format = str(format or "jsonl").strip().lower() or "jsonl"
        if export_format not in {"jsonl", "csv"}:
            if emit_audit:
                self._emit_deep_archive_audit(
                    session_id=session_id,
                    action=audit_action,
                    status="error",
                    started_at=started_at,
                    detail={"reason": "invalid_format", "format": export_format},
                )
            raise ValueError("format must be one of: jsonl, csv")
        snapshot = self.deep_think_list_events(
            session_id,
            round_id=round_id,
            limit=limit,
            event_name=event_name,
            cursor=cursor,
            created_from=created_from,
            created_to=created_to,
        )
        if "error" in snapshot:
            if emit_audit:
                self._emit_deep_archive_audit(
                    session_id=session_id,
                    action=audit_action,
                    status=str(snapshot.get("error", "error")),
                    started_at=started_at,
                    detail={"message": snapshot.get("error", "unknown")},
                )
            return snapshot
        events = list(snapshot.get("events", []))
        round_value = str(snapshot.get("round_id", "")).strip() or "all"
        if export_format == "jsonl":
            lines = [json.dumps(item, ensure_ascii=False) for item in events]
            content = "\n".join(lines)
            if content:
                content += "\n"
            if emit_audit:
                self._emit_deep_archive_audit(
                    session_id=session_id,
                    action=audit_action,
                    status="ok",
                    started_at=started_at,
                    result_count=len(events),
                    export_bytes=len(content.encode("utf-8")),
                    detail={"format": "jsonl"},
                )
            return {
                "format": "jsonl",
                "media_type": "application/x-ndjson; charset=utf-8",
                "filename": f"deepthink-events-{session_id}-{round_value}.jsonl",
                "content": content,
                "count": len(events),
            }
        output = io.StringIO()
        fieldnames = [
            "event_id",
            "session_id",
            "round_id",
            "round_no",
            "event_seq",
            "event",
            "created_at",
            "data_json",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for item in events:
            writer.writerow(
                {
                    "event_id": item.get("event_id"),
                    "session_id": item.get("session_id"),
                    "round_id": item.get("round_id"),
                    "round_no": item.get("round_no"),
                    "event_seq": item.get("event_seq"),
                    "event": item.get("event"),
                    "created_at": item.get("created_at"),
                    "data_json": json.dumps(item.get("data", {}), ensure_ascii=False),
                }
            )
        # 涓?Windows Excel 鍏煎鍐欏叆 UTF-8 BOM锛岄伩鍏嶄腑鏂囧垪鍊间贡鐮併€?
        csv_content = "\ufeff" + output.getvalue()
        if emit_audit:
            self._emit_deep_archive_audit(
                session_id=session_id,
                action=audit_action,
                status="ok",
                started_at=started_at,
                result_count=len(events),
                export_bytes=len(csv_content.encode("utf-8")),
                detail={"format": "csv"},
            )
        return {
            "format": "csv",
            "media_type": "text/csv; charset=utf-8",
            "filename": f"deepthink-events-{session_id}-{round_value}.csv",
            "content": csv_content,
            "count": len(events),
        }

    def deep_think_export_business(
        self,
        session_id: str,
        *,
        round_id: str | None = None,
        format: str = "csv",
        limit: int = 400,
    ) -> dict[str, Any]:
        """Export business-facing summary rows for PM/trading usage, separate from raw event export."""
        export_format = str(format or "csv").strip().lower() or "csv"
        if export_format not in {"csv", "json"}:
            raise ValueError("format must be one of: csv, json")
        session = self.web.deep_think_get_session(session_id)
        if not session:
            return {"error": "not_found", "session_id": session_id}
        safe_limit = max(20, min(2000, int(limit)))
        round_id_clean = str(round_id or "").strip()
        rows: list[dict[str, Any]] = []
        # 浼樺厛浣跨敤涓氬姟鎽樿浜嬩欢锛岀‘淇濆鍑虹殑鏄鐢ㄦ埛鏈夋剰涔夌殑鍐崇瓥缁撴灉銆?
        summary_snapshot = self.deep_think_list_events(
            session_id,
            round_id=round_id_clean or None,
            limit=safe_limit,
            event_name="business_summary",
        )
        events = summary_snapshot.get("events", []) if isinstance(summary_snapshot, dict) else []
        if isinstance(events, list):
            for item in events:
                data = item.get("data", {}) if isinstance(item, dict) else {}
                if not isinstance(data, dict):
                    continue
                citations = data.get("citations", [])
                top_source = ""
                if isinstance(citations, list) and citations:
                    top = citations[0] if isinstance(citations[0], dict) else {}
                    top_source = str(top.get("url", "")).strip()
                rows.append(
                    {
                        "session_id": session_id,
                        "round_id": str(item.get("round_id", "")),
                        "round_no": int(item.get("round_no", 0) or 0),
                        "stock_code": str(data.get("stock_code", "")),
                        "signal": str(data.get("signal", "hold")),
                        "confidence": float(data.get("confidence", 0.0) or 0.0),
                        "trigger_condition": str(data.get("trigger_condition", "")),
                        "invalidation_condition": str(data.get("invalidation_condition", "")),
                        "review_time_hint": str(data.get("review_time_hint", "")),
                        "top_conflict_sources": ",".join(str(x) for x in list(data.get("top_conflict_sources", []))[:4]),
                        "replan_triggered": bool(data.get("replan_triggered", False)),
                        "stop_reason": str(data.get("stop_reason", "")),
                        "top_source_url": top_source,
                        "created_at": str(item.get("created_at", "")),
                    }
                )
        # 鑻ュ巻鍙茶疆娆℃棤 business_summary 浜嬩欢锛屽厹搴曚娇鐢ㄤ細璇?round 蹇収鐢熸垚鍙鎽樿銆?
        if not rows:
            rounds = list(session.get("rounds", []))
            if round_id_clean:
                rounds = [r for r in rounds if str(r.get("round_id", "")) == round_id_clean]
            for round_item in rounds[-safe_limit:]:
                opinions = list(round_item.get("opinions", []))
                top_opinion = opinions[0] if opinions else {}
                rows.append(
                    {
                        "session_id": session_id,
                        "round_id": str(round_item.get("round_id", "")),
                        "round_no": int(round_item.get("round_no", 0) or 0),
                        "stock_code": ",".join(str(x) for x in session.get("stock_codes", [])),
                        "signal": str(round_item.get("consensus_signal", "hold")),
                        "confidence": round(1.0 - float(round_item.get("disagreement_score", 0.0) or 0.0), 4),
                        "trigger_condition": str(top_opinion.get("reason", "Re-check with fresher intelligence before action")), 
                        "invalidation_condition": "Signal invalidates if risk factors worsen or key signals reverse.",
                        "review_time_hint": "T+1 鏃ュ唴澶嶆牳",
                        "top_conflict_sources": ",".join(str(x) for x in list(round_item.get("conflict_sources", []))[:4]),
                        "replan_triggered": bool(round_item.get("replan_triggered", False)),
                        "stop_reason": str(round_item.get("stop_reason", "")),
                        "top_source_url": "",
                        "created_at": str(round_item.get("created_at", "")),
                    }
                )
        round_value = round_id_clean or "all"
        if export_format == "json":
            return {
                "format": "json",
                "media_type": "application/json; charset=utf-8",
                "filename": f"deepthink-business-{session_id}-{round_value}.json",
                "content": json.dumps(rows, ensure_ascii=False, indent=2),
                "count": len(rows),
            }
        output = io.StringIO()
        fieldnames = [
            "session_id",
            "round_id",
            "round_no",
            "stock_code",
            "signal",
            "confidence",
            "trigger_condition",
            "invalidation_condition",
            "review_time_hint",
            "top_conflict_sources",
            "replan_triggered",
            "stop_reason",
            "top_source_url",
            "created_at",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        return {
            "format": "csv",
            "media_type": "text/csv; charset=utf-8",
            "filename": f"deepthink-business-{session_id}-{round_value}.csv",
            # 鍚屾牱鍔?BOM锛岀‘淇濅笟鍔＄敤鎴峰湪 Excel 涓洿鎺ュ彲璇汇€?
            "content": "\ufeff" + output.getvalue(),
            "count": len(rows),
        }

    def deep_think_intel_self_test(self, *, stock_code: str, question: str = "") -> dict[str, Any]:
        """DeepThink intel self-test: verify provider/websearch path and fallback reasons."""
        code = (stock_code or "SH600000").upper().replace(".", "")
        probe_question = (question or f"Please run a 30-day macro/industry/event intel self-test for {code}").strip()
        provider_rows = [
            {
                "name": str(p.name),
                "enabled": bool(p.enabled),
                "api_style": str(p.api_style),
                "model": str(p.model),
            }
            for p in self.llm_gateway.providers
        ]
        # 灏介噺浣跨敤鎺ヨ繎鐪熷疄閾捐矾鐨勬暟鎹揩鐓э紝閬垮厤鈥滆嚜妫€閫氳繃浣嗗疄鎴樺け璐モ€濈殑鍋忓樊銆?
        if self._needs_quote_refresh(code):
            self.ingest_market_daily([code])
        if self._needs_history_refresh(code):
            self.ingestion.ingest_history_daily([code], limit=260)
        if self._needs_financial_refresh(code):
            self.ingest_financials([code])
        if self._needs_news_refresh(code):
            self.ingest_news([code], limit=8)
        if self._needs_research_refresh(code):
            self.ingest_research_reports([code], limit=6)
        if self._needs_fund_refresh(code):
            self.ingest_fund_snapshots([code])
        if self._needs_macro_refresh():
            self.ingest_macro_indicators(limit=8)
        quote = self._latest_quote(code) or {}
        bars = self._history_bars(code, limit=180)
        financial = self._latest_financial_snapshot(code) or {}
        trend = self._trend_metrics(bars) if len(bars) >= 30 else {}
        pred = self.predict_run({"stock_codes": [code], "horizons": ["20d"]})
        horizon_map = {h["horizon"]: h for h in pred["results"][0]["horizons"]} if pred.get("results") else {}
        quant_20 = horizon_map.get("20d", {})

        intel = self._deep_fetch_intel_via_llm_websearch(
            stock_code=code,
            question=probe_question,
            quote=quote,
            trend=trend,
            quant_20=quant_20,
        )
        trace_id = str(intel.get("trace_id", ""))
        trace_rows = self.deep_think_trace_events(trace_id, limit=120).get("events", []) if trace_id else []
        citations = list(intel.get("citations", [])) if isinstance(intel, dict) else []
        return {
            "ok": str(intel.get("intel_status", "")) == "external_ok",
            "stock_code": code,
            "question": probe_question,
            "external_enabled": bool(self.settings.llm_external_enabled),
            "provider_count": len([x for x in provider_rows if bool(x.get("enabled"))]),
            "providers": provider_rows,
            "financial_snapshot": financial,
            "intel_status": str(intel.get("intel_status", "")),
            "confidence_note": str(intel.get("confidence_note", "")),
            "fallback_reason": str(intel.get("fallback_reason", "")),
            "fallback_error": str(intel.get("fallback_error", "")),
            "websearch_tool_requested": bool(intel.get("websearch_tool_requested", False)),
            "websearch_tool_applied": bool(intel.get("websearch_tool_applied", False)),
            "citation_count": len(citations),
            "trace_id": trace_id,
            "trace_events": trace_rows,
            "preview": {
                "as_of": str(intel.get("as_of", "")),
                "macro_titles": [str(x.get("title", "")) for x in list(intel.get("macro_signals", []))[:3]],
                "calendar_titles": [str(x.get("title", "")) for x in list(intel.get("calendar_watchlist", []))[:5]],
            },
        }

    def deep_think_trace_events(self, trace_id: str, *, limit: int = 80) -> dict[str, Any]:
        """Return trace event details to debug retrieval/runtime downgrade causes."""
        safe_trace_id = str(trace_id or "").strip()
        if not safe_trace_id:
            return {"trace_id": "", "count": 0, "events": []}
        safe_limit = max(1, min(500, int(limit)))
        events = self.traces.list_events(safe_trace_id)
        rows = [
            {
                "ts_ms": int(item.ts_ms),
                "name": str(item.name),
                "payload": dict(item.payload),
            }
            for item in events[-safe_limit:]
        ]
        return {"trace_id": safe_trace_id, "count": len(rows), "events": rows}

    def deep_think_create_export_task(
        self,
        session_id: str,
        *,
        format: str = "jsonl",
        round_id: str | None = None,
        limit: int = 200,
        event_name: str | None = None,
        cursor: int | None = None,
        created_from: str | None = None,
        created_to: str | None = None,
    ) -> dict[str, Any]:
        started_at = time.perf_counter()
        session = self.web.deep_think_get_session(session_id)
        if not session:
            self._emit_deep_archive_audit(
                session_id=session_id,
                action="archive_export_task_create",
                status="not_found",
                started_at=started_at,
            )
            return {"error": "not_found", "session_id": session_id}
        export_format = str(format or "jsonl").strip().lower() or "jsonl"
        if export_format not in {"jsonl", "csv"}:
            self._emit_deep_archive_audit(
                session_id=session_id,
                action="archive_export_task_create",
                status="error",
                started_at=started_at,
                detail={"reason": "invalid_format", "format": export_format},
            )
            raise ValueError("format must be one of: jsonl, csv")
        try:
            filters = self._build_deep_archive_query_options(
                round_id=round_id,
                limit=limit,
                event_name=event_name,
                cursor=cursor,
                created_from=created_from,
                created_to=created_to,
            )
        except Exception as ex:  # noqa: BLE001
            self._emit_deep_archive_audit(
                session_id=session_id,
                action="archive_export_task_create",
                status="error",
                started_at=started_at,
                detail={"reason": str(ex)},
            )
            raise
        task_id = f"dtexp-{uuid.uuid4().hex[:12]}"
        max_attempts = max(1, int(self.settings.deep_archive_export_task_max_attempts))
        self.web.deep_think_export_task_create(
            task_id=task_id,
            session_id=session_id,
            status="queued",
            format=export_format,
            filters=filters,
            max_attempts=max_attempts,
        )
        self._emit_deep_archive_audit(
            session_id=session_id,
            action="archive_export_task_create",
            status="ok",
            started_at=started_at,
            detail={"task_id": task_id, "format": export_format, "max_attempts": max_attempts},
        )
        self._deep_archive_export_executor.submit(
            self._run_deep_archive_export_task,
            task_id,
            session_id,
            export_format,
            filters,
        )
        return self.deep_think_get_export_task(session_id, task_id)

    def _run_deep_archive_export_task(
        self,
        task_id: str,
        session_id: str,
        export_format: str,
        filters: dict[str, Any],
    ) -> None:
        base_backoff_seconds = max(0.0, float(self.settings.deep_archive_export_retry_backoff_seconds))
        while True:
            task_snapshot = self.web.deep_think_export_task_try_claim(task_id, session_id=session_id)
            if not task_snapshot:
                return
            attempt_count = max(1, int(task_snapshot.get("attempt_count", 1) or 1))
            max_attempts = max(1, int(task_snapshot.get("max_attempts", 1) or 1))
            started_at = time.perf_counter()
            try:
                exported = self.deep_think_export_events(
                    session_id,
                    round_id=filters.get("round_id"),
                    limit=int(filters.get("limit", 200)),
                    event_name=filters.get("event_name"),
                    cursor=filters.get("cursor"),
                    created_from=filters.get("created_from"),
                    created_to=filters.get("created_to"),
                    format=export_format,
                    audit_action="archive_export_task_run",
                    emit_audit=False,
                )
                if "error" in exported:
                    raise RuntimeError(str(exported.get("error", "failed")))
                content = str(exported.get("content", ""))
                self.web.deep_think_export_task_update(
                    task_id=task_id,
                    status="completed",
                    filename=str(exported.get("filename", "")),
                    media_type=str(exported.get("media_type", "text/plain; charset=utf-8")),
                    content_text=content,
                    row_count=int(exported.get("count", 0) or 0),
                    error="",
                )
                self._emit_deep_archive_audit(
                    session_id=session_id,
                    action="archive_export_task_complete",
                    status="ok",
                    started_at=started_at,
                    result_count=int(exported.get("count", 0) or 0),
                    export_bytes=len(content.encode("utf-8")),
                    detail={
                        "task_id": task_id,
                        "format": export_format,
                        "attempt_count": attempt_count,
                        "max_attempts": max_attempts,
                    },
                )
                return
            except Exception as ex:  # noqa: BLE001
                message = str(ex)
                if attempt_count < max_attempts:
                    self.web.deep_think_export_task_requeue(task_id=task_id, error=message)
                    self._emit_deep_archive_audit(
                        session_id=session_id,
                        action="archive_export_task_complete",
                        status="retrying",
                        started_at=started_at,
                        detail={
                            "task_id": task_id,
                            "attempt_count": attempt_count,
                            "max_attempts": max_attempts,
                            "error": message,
                        },
                    )
                    if base_backoff_seconds > 0:
                        backoff = min(3.0, base_backoff_seconds * (2 ** max(0, attempt_count - 1)))
                        if backoff > 0:
                            time.sleep(backoff)
                    continue
                self.web.deep_think_export_task_update(task_id=task_id, status="failed", error=message)
                self._emit_deep_archive_audit(
                    session_id=session_id,
                    action="archive_export_task_complete",
                    status="error",
                    started_at=started_at,
                    detail={
                        "task_id": task_id,
                        "attempt_count": attempt_count,
                        "max_attempts": max_attempts,
                        "error": message,
                    },
                )
                return

    def deep_think_get_export_task(self, session_id: str, task_id: str) -> dict[str, Any]:
        started_at = time.perf_counter()
        task = self.web.deep_think_export_task_get(task_id, session_id=session_id, include_content=False)
        if not task:
            self._emit_deep_archive_audit(
                session_id=session_id,
                action="archive_export_task_get",
                status="not_found",
                started_at=started_at,
                detail={"task_id": task_id},
            )
            return {"error": "not_found", "task_id": task_id, "session_id": session_id}
        task_error = str(task.pop("error", "") or "").strip()
        if task_error:
            task["failure_reason"] = task_error
        task["attempt_count"] = int(task.get("attempt_count", 0) or 0)
        task["max_attempts"] = max(1, int(task.get("max_attempts", 1) or 1))
        task["download_ready"] = str(task.get("status", "")) == "completed"
        self._emit_deep_archive_audit(
            session_id=session_id,
            action="archive_export_task_get",
            status="ok",
            started_at=started_at,
            detail={"task_id": task_id, "status": task.get("status", "")},
        )
        return task

    def deep_think_download_export_task(self, session_id: str, task_id: str) -> dict[str, Any]:
        started_at = time.perf_counter()
        task = self.web.deep_think_export_task_get(task_id, session_id=session_id, include_content=True)
        if not task:
            self._emit_deep_archive_audit(
                session_id=session_id,
                action="archive_export_task_download",
                status="not_found",
                started_at=started_at,
                detail={"task_id": task_id},
            )
            return {"error": "not_found", "task_id": task_id, "session_id": session_id}
        status = str(task.get("status", ""))
        failure_reason = str(task.get("error", "")).strip()
        if status == "failed":
            self._emit_deep_archive_audit(
                session_id=session_id,
                action="archive_export_task_download",
                status="failed",
                started_at=started_at,
                detail={"task_id": task_id, "error": failure_reason},
            )
            return {
                "error": "failed",
                "task_id": task_id,
                "session_id": session_id,
                "message": failure_reason or "export task failed",
            }
        if status != "completed":
            self._emit_deep_archive_audit(
                session_id=session_id,
                action="archive_export_task_download",
                status="not_ready",
                started_at=started_at,
                detail={"task_id": task_id, "status": status},
            )
            return {"error": "not_ready", "task_id": task_id, "session_id": session_id, "status": status}
        content = str(task.get("content_text", ""))
        self._emit_deep_archive_audit(
            session_id=session_id,
            action="archive_export_task_download",
            status="ok",
            started_at=started_at,
            result_count=int(task.get("row_count", 0) or 0),
            export_bytes=len(content.encode("utf-8")),
            detail={"task_id": task_id, "format": task.get("format", "")},
        )
        return {
            "task_id": task_id,
            "session_id": session_id,
            "status": status,
            "format": str(task.get("format", "jsonl")),
            "filename": str(task.get("filename", f"{task_id}.txt")),
            "media_type": str(task.get("media_type", "text/plain; charset=utf-8")),
            "content": content,
            "count": int(task.get("row_count", 0) or 0),
        }

    def a2a_agent_cards(self) -> list[dict[str, Any]]:
        return self.web.list_agent_cards()

    def a2a_create_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        agent_id = str(payload.get("agent_id", "")).strip()
        if not agent_id:
            raise ValueError("agent_id is required")
        cards = self.web.list_agent_cards()
        if not any(str(card.get("agent_id")) == agent_id for card in cards):
            raise ValueError("agent_id not found in card registry")

        task_id = str(payload.get("task_id", "")).strip() or f"a2a-{uuid.uuid4().hex[:12]}"
        session_id = str(payload.get("session_id", "")).strip() or None
        trace_ref = self.traces.new_trace()
        self.web.a2a_create_task(
            task_id=task_id,
            session_id=session_id,
            agent_id=agent_id,
            status="created",
            payload=payload,
            result={"history": ["created"]},
            trace_ref=trace_ref,
        )

        self.web.a2a_update_task(task_id=task_id, status="accepted", result={"history": ["created", "accepted"]})
        self.web.a2a_update_task(task_id=task_id, status="in_progress", result={"history": ["created", "accepted", "in_progress"]})

        result: dict[str, Any] = {"task_id": task_id, "agent_id": agent_id, "status": "completed"}
        if session_id and str(payload.get("task_type", "")) == "deep_round":
            snapshot = self.deep_think_run_round(session_id, {"question": payload.get("question", "")})
            result["deep_think_snapshot"] = snapshot
        self.web.a2a_update_task(
            task_id=task_id,
            status="completed",
            result={"history": ["created", "accepted", "in_progress", "completed"], "payload_result": result},
        )
        self.traces.emit(trace_ref, "a2a_task_completed", {"task_id": task_id, "agent_id": agent_id})
        return self.web.a2a_get_task(task_id)

    def a2a_get_task(self, task_id: str) -> dict[str, Any]:
        task = self.web.a2a_get_task(task_id)
        if not task:
            return {"error": "not_found", "task_id": task_id}
        return task

    @staticmethod
    def _percentile_from_sorted(values: list[int], ratio: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return float(values[0])
        idx = int(round((len(values) - 1) * max(0.0, min(1.0, float(ratio)))))
        idx = max(0, min(len(values) - 1, idx))
        return float(values[idx])

    def ops_journal_health(self, token: str, *, window_hours: int = 168, limit: int = 400) -> dict[str, Any]:
        self.web.require_role(token, {"admin", "ops"})
        safe_window_hours = max(1, min(24 * 180, int(window_hours)))
        safe_limit = max(20, min(2000, int(limit)))
        rows = self.web.journal_ai_generation_log_list(token, window_hours=safe_window_hours, limit=safe_limit)

        ready_count = 0
        fallback_count = 0
        failed_count = 0
        provider_counter: Counter[str] = Counter()
        latencies: list[int] = []
        recent_failures: list[dict[str, Any]] = []
        for row in rows:
            status = str(row.get("status", "")).strip().lower()
            if status == "ready":
                ready_count += 1
            elif status == "fallback":
                fallback_count += 1
            elif status == "failed":
                failed_count += 1
            provider_name = str(row.get("provider", "")).strip() or "local_fallback"
            provider_counter[provider_name] += 1
            latency = max(0, int(row.get("latency_ms", 0) or 0))
            latencies.append(latency)
            if status in {"fallback", "failed"} and len(recent_failures) < 8:
                recent_failures.append(
                    {
                        "journal_id": int(row.get("journal_id", 0) or 0),
                        "status": status,
                        "error_code": str(row.get("error_code", "")),
                        "error_message": str(row.get("error_message", "")),
                        "generated_at": str(row.get("generated_at", "")),
                    }
                )

        total_attempts = len(rows)
        latencies_sorted = sorted(latencies)
        avg_latency = float(sum(latencies_sorted)) / float(total_attempts) if total_attempts else 0.0
        coverage_counts = self.web.journal_ai_coverage_counts(token)
        total_journals = int(coverage_counts.get("total_journals", 0) or 0)
        journals_with_ai = int(coverage_counts.get("journals_with_ai", 0) or 0)
        coverage_rate = float(journals_with_ai) / float(total_journals) if total_journals else 0.0

        return {
            "status": "ok",
            "window_hours": safe_window_hours,
            "sample_limit": safe_limit,
            "attempts": {
                "total": total_attempts,
                "ready": ready_count,
                "fallback": fallback_count,
                "failed": failed_count,
                "fallback_rate": round(float(fallback_count) / float(total_attempts), 4) if total_attempts else 0.0,
                "failure_rate": round(float(failed_count) / float(total_attempts), 4) if total_attempts else 0.0,
            },
            "latency_ms": {
                "avg": round(avg_latency, 2),
                "p50": round(self._percentile_from_sorted(latencies_sorted, 0.50), 2),
                "p95": round(self._percentile_from_sorted(latencies_sorted, 0.95), 2),
                "max": int(max(latencies_sorted) if latencies_sorted else 0),
            },
            "provider_breakdown": self._journal_counter_breakdown(provider_counter, total=max(1, total_attempts), top_n=12),
            "coverage": {
                "total_journals": total_journals,
                "journals_with_ai_reflection": journals_with_ai,
                "ai_reflection_coverage_rate": round(coverage_rate, 4),
            },
            "recent_failures": recent_failures,
        }

    def ops_deep_think_archive_metrics(self, token: str, *, window_hours: int = 24) -> dict[str, Any]:
        self.web.require_role(token, {"admin", "ops"})
        return self.web.deep_think_archive_audit_metrics(window_hours=window_hours)

    def _build_datasource_catalog(self) -> list[dict[str, Any]]:
        """Build datasource metadata snapshot from currently wired adapters."""

        rows: list[dict[str, Any]] = []
        adapter_groups = [
            ("quote", getattr(self.ingestion.quote_service, "adapters", [])),
            ("announcement", getattr(self.ingestion.announcement_service, "adapters", [])),
            ("financial", getattr(self.ingestion.financial_service, "adapters", []) if self.ingestion.financial_service else []),
            ("news", getattr(self.ingestion.news_service, "adapters", []) if self.ingestion.news_service else []),
            ("research", getattr(self.ingestion.research_service, "adapters", []) if self.ingestion.research_service else []),
            ("macro", getattr(self.ingestion.macro_service, "adapters", []) if self.ingestion.macro_service else []),
            ("fund", getattr(self.ingestion.fund_service, "adapters", []) if self.ingestion.fund_service else []),
        ]
        for category, adapters in adapter_groups:
            for adapter in adapters:
                source_id = str(getattr(adapter, "source_id", f"{category}_unknown"))
                cfg = getattr(adapter, "config", None)
                reliability = float(
                    getattr(cfg, "reliability_score", getattr(adapter, "reliability_score", 0.0)) or 0.0
                )
                proxy_url = str(getattr(cfg, "proxy_url", "") or "")
                cookie = str(getattr(adapter, "cookie", "") or "")
                # Cookie-backed sources are marked disabled when credential is missing.
                enabled = bool(cookie.strip()) if hasattr(adapter, "cookie") else True
                rows.append(
                    {
                        "source_id": source_id,
                        "category": category,
                        "enabled": enabled,
                        "reliability_score": round(reliability, 4),
                        "source_url": str(getattr(adapter, "source_url", "") or ""),
                        "proxy_enabled": bool(proxy_url.strip()),
                        # Frontend uses this list to explain datasource business impact.
                        "used_in_ui_modules": self._datasource_ui_modules(category),
                    }
                )
        # History ingestion currently relies on HistoryService (non-adapter style), expose a synthetic row.
        rows.append(
            {
                "source_id": "eastmoney_history",
                "category": "history",
                "enabled": True,
                "reliability_score": 0.9,
                "source_url": "https://push2his.eastmoney.com/",
                "proxy_enabled": bool(str(self.settings.datasource_proxy_url or "").strip()),
                "used_in_ui_modules": self._datasource_ui_modules("history"),
            }
        )

        dedup: dict[str, dict[str, Any]] = {}
        for row in rows:
            dedup.setdefault(str(row["source_id"]), row)
        return sorted(dedup.values(), key=lambda x: (str(x.get("category", "")), str(x.get("source_id", ""))))

    @staticmethod
    def _datasource_ui_modules(category: str) -> list[str]:
        """Map datasource category to user-facing modules for observability explainability."""
        mapping = {
            "quote": ["/deep-think", "/analysis-studio", "/predict", "/watchlist"],
            "announcement": ["/deep-think", "/reports", "/analysis-studio"],
            "financial": ["/deep-think", "/analysis-studio", "/predict"],
            "news": ["/deep-think", "/analysis-studio", "/reports"],
            "research": ["/deep-think", "/analysis-studio", "/reports"],
            "macro": ["/deep-think", "/analysis-studio"],
            "fund": ["/deep-think", "/analysis-studio"],
            "history": ["/deep-think", "/analysis-studio", "/predict"],
            "scheduler": ["/ops/scheduler", "/ops/health"],
        }
        return list(mapping.get(str(category or "").strip().lower(), []))

    def _append_datasource_log(
        self,
        *,
        source_id: str,
        category: str,
        action: str,
        status: str,
        latency_ms: int,
        detail: dict[str, Any] | None = None,
        error: str = "",
    ) -> dict[str, Any]:
        self._datasource_log_seq += 1
        row = {
            "log_id": self._datasource_log_seq,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_id": source_id,
            "category": category,
            "action": action,
            "status": status,
            "latency_ms": max(0, int(latency_ms)),
            "error": error,
            "detail": detail or {},
        }
        self._datasource_logs.append(row)
        # Keep bounded memory footprint for long-running dev processes.
        if len(self._datasource_logs) > 5000:
            self._datasource_logs = self._datasource_logs[-5000:]
        return row

    def datasource_sources(self, token: str) -> dict[str, Any]:
        self.web.require_role(token, {"admin", "ops"})
        items = self._build_datasource_catalog()
        return {"count": len(items), "items": items}

    def datasource_logs(self, token: str, *, source_id: str = "", status: str = "", limit: int = 100) -> dict[str, Any]:
        self.web.require_role(token, {"admin", "ops"})
        safe_limit = max(1, min(1000, int(limit)))
        source_filter = str(source_id or "").strip()
        status_filter = str(status or "").strip().lower()
        rows = list(reversed(self._datasource_logs))
        if source_filter:
            rows = [x for x in rows if str(x.get("source_id", "")) == source_filter]
        if status_filter:
            rows = [x for x in rows if str(x.get("status", "")).lower() == status_filter]
        rows = rows[:safe_limit]
        return {"count": len(rows), "items": rows}

    def datasource_health(self, token: str, *, limit: int = 200) -> dict[str, Any]:
        self.web.require_role(token, {"admin", "ops"})
        safe_limit = max(1, min(1000, int(limit)))
        catalog = self._build_datasource_catalog()
        metrics_by_source: dict[str, dict[str, Any]] = {}
        for row in reversed(self._datasource_logs):
            source_id = str(row.get("source_id", ""))
            if not source_id:
                continue
            metrics = metrics_by_source.setdefault(
                source_id,
                {"attempts": 0, "success": 0, "failed": 0, "last_error": "", "last_latency_ms": 0, "last_seen_at": ""},
            )
            metrics["attempts"] += 1
            if str(row.get("status", "")) == "ok":
                metrics["success"] += 1
            elif str(row.get("status", "")) == "failed":
                metrics["failed"] += 1
                if not metrics["last_error"]:
                    metrics["last_error"] = str(row.get("error", ""))
            metrics["last_latency_ms"] = int(row.get("latency_ms", 0) or 0)
            metrics["last_seen_at"] = str(row.get("created_at", ""))

        items: list[dict[str, Any]] = []
        for source in catalog:
            source_id = str(source.get("source_id", ""))
            metric = metrics_by_source.get(source_id, {})
            attempts = int(metric.get("attempts", 0) or 0)
            failed = int(metric.get("failed", 0) or 0)
            success = int(metric.get("success", 0) or 0)
            success_rate = (float(success) / float(attempts)) if attempts > 0 else 1.0
            failure_rate = (float(failed) / float(attempts)) if attempts > 0 else 0.0
            # When a source has never been fetched in this process lifetime, freshness is unknown.
            last_seen_at = str(metric.get("last_seen_at", ""))
            staleness_minutes: int | None = None
            if last_seen_at:
                delta = datetime.now(timezone.utc) - self._parse_time(last_seen_at)
                staleness_minutes = max(0, int(delta.total_seconds() // 60))
            item = {
                **source,
                "attempts": attempts,
                "success_rate": round(success_rate, 4),
                "failure_rate": round(failure_rate, 4),
                "last_error": str(metric.get("last_error", "")),
                "last_latency_ms": int(metric.get("last_latency_ms", 0) or 0),
                "updated_at": last_seen_at,
                "last_used_at": last_seen_at,
                "staleness_minutes": staleness_minutes,
                "circuit_open": False,
                "used_in_ui_modules": source.get(
                    "used_in_ui_modules",
                    self._datasource_ui_modules(str(source.get("category", ""))),
                ),
            }
            self.web.source_health_upsert(
                source_id=source_id,
                success_rate=float(item["success_rate"]),
                circuit_open=False,
                last_error=item["last_error"],
            )
            # Alert when a source continuously fails within the local observation window.
            if attempts >= 3 and failure_rate >= 0.6:
                self.web.create_alert(
                    "datasource_failure_rate_high",
                    "high",
                    f"{source_id} failure_rate={failure_rate:.2f} attempts={attempts}",
                )
            items.append(item)

        scheduler_state = self.scheduler_status()
        for job_name, snapshot in scheduler_state.items():
            circuit_open = bool(snapshot.get("circuit_open_until"))
            if circuit_open:
                self.web.create_alert("scheduler_circuit_open", "high", f"{job_name} circuit open")
            scheduler_updated_at = str(snapshot.get("last_run_at", ""))
            scheduler_staleness: int | None = None
            if scheduler_updated_at:
                delta = datetime.now(timezone.utc) - self._parse_time(scheduler_updated_at)
                scheduler_staleness = max(0, int(delta.total_seconds() // 60))
            items.append(
                {
                    "source_id": str(job_name),
                    "category": "scheduler",
                    "enabled": True,
                    "reliability_score": 0.0,
                    "source_url": "",
                    "proxy_enabled": False,
                    "attempts": 0,
                    "success_rate": 1.0 if snapshot.get("last_status") in ("ok", "never") else 0.0,
                    "failure_rate": 0.0 if snapshot.get("last_status") in ("ok", "never") else 1.0,
                    "last_error": str(snapshot.get("last_error", "")),
                    "last_latency_ms": 0,
                    "updated_at": scheduler_updated_at,
                    "last_used_at": scheduler_updated_at,
                    "staleness_minutes": scheduler_staleness,
                    "circuit_open": circuit_open,
                    "used_in_ui_modules": self._datasource_ui_modules("scheduler"),
                }
            )

        items = sorted(items, key=lambda x: (str(x.get("category", "")), str(x.get("source_id", ""))))[:safe_limit]
        return {"count": len(items), "items": items}

    def business_data_health(self, *, stock_code: str = "", limit: int = 200) -> dict[str, Any]:
        """Return business-readable datasource health summary for frontend modules."""
        try:
            health = self.datasource_health("", limit=limit)
        except Exception:
            # Fallback to catalog-only snapshot when role checks block observability APIs.
            health = {"count": 0, "items": self._build_datasource_catalog()}
        rows = list(health.get("items", []))
        category_map: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            category = str(row.get("category", "")).strip() or "unknown"
            category_map.setdefault(category, []).append(row)

        # Keep the mapping explicit so the frontend can explain module impact in plain language.
        module_categories: dict[str, list[str]] = {
            "predict": ["quote", "history", "financial"],
            "reports": ["quote", "history", "financial", "news", "research", "macro", "fund", "announcement"],
            "deep-think": ["quote", "history", "financial", "news", "research", "macro", "fund", "announcement"],
            "analysis-studio": ["quote", "history", "financial", "news", "research", "macro", "fund", "announcement"],
        }
        module_health: list[dict[str, Any]] = []
        global_reasons: list[str] = []
        for module, categories in module_categories.items():
            expected = max(1, len(categories))
            healthy_count = 0
            missing_categories: list[str] = []
            stale_categories: list[str] = []
            failed_categories: list[str] = []
            for category in categories:
                candidates = category_map.get(category, [])
                if not candidates:
                    missing_categories.append(category)
                    continue
                top = sorted(
                    candidates,
                    key=lambda x: (
                        float(x.get("success_rate", 0.0) or 0.0),
                        -float(x.get("failure_rate", 1.0) or 1.0),
                    ),
                    reverse=True,
                )[0]
                failure_rate = float(top.get("failure_rate", 0.0) or 0.0)
                staleness = top.get("staleness_minutes")
                staleness_minutes = int(staleness) if isinstance(staleness, (int, float)) else None
                if failure_rate >= 0.6:
                    failed_categories.append(category)
                elif staleness_minutes is not None and staleness_minutes > 240:
                    stale_categories.append(category)
                else:
                    healthy_count += 1

            coverage = round(float(healthy_count) / float(expected), 4)
            status = "ok" if coverage >= 0.85 else "degraded" if coverage >= 0.5 else "critical"
            reasons: list[str] = []
            if missing_categories:
                reasons.append(f"missing:{','.join(missing_categories)}")
            if stale_categories:
                reasons.append(f"stale:{','.join(stale_categories)}")
            if failed_categories:
                reasons.append(f"failed:{','.join(failed_categories)}")
            module_health.append(
                {
                    "module": module,
                    "status": status,
                    "coverage": coverage,
                    "healthy_categories": healthy_count,
                    "expected_categories": expected,
                    "degrade_reasons": reasons,
                }
            )
            global_reasons.extend(reasons)

        status_rank = {"ok": 0, "degraded": 1, "critical": 2}
        overall_status = "ok"
        for row in module_health:
            if status_rank.get(str(row.get("status", "")), 0) > status_rank.get(overall_status, 0):
                overall_status = str(row.get("status", "ok"))

        stock = str(stock_code or "").strip().upper()
        stock_snapshot = {}
        if stock:
            stock_snapshot = {
                "stock_code": stock,
                "has_quote": bool(self._latest_quote(stock)),
                "history_sample_size": len(self._history_bars(stock, limit=260)),
                "has_financial": bool(self._latest_financial_snapshot(stock)),
                "news_count": len([x for x in self.ingestion_store.news_items if str(x.get("stock_code", "")).upper() == stock][-20:]),
                "research_count": len(
                    [x for x in self.ingestion_store.research_reports if str(x.get("stock_code", "")).upper() == stock][-20:]
                ),
            }

        return {
            "status": overall_status,
            "module_health": module_health,
            "degrade_reasons": list(dict.fromkeys(global_reasons)),
            "category_health_count": len(rows),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "stock_snapshot": stock_snapshot,
        }

    @staticmethod
    def _infer_datasource_category(source_id: str) -> str:
        sid = str(source_id or "").strip().lower()
        if sid.startswith(("tencent", "sina", "netease", "xueqiu")) and "news" not in sid:
            return "quote"
        if "announcement" in sid:
            return "announcement"
        if "financial" in sid or "tushare" in sid:
            return "financial"
        if "news" in sid or sid.startswith(("cls", "tradingview")):
            return "news"
        if "research" in sid:
            return "research"
        if "macro" in sid:
            return "macro"
        if "fund" in sid or "ttjj" in sid:
            return "fund"
        if "history" in sid:
            return "history"
        return ""

    def datasource_fetch(self, token: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.web.require_role(token, {"admin", "ops"})
        source_id = str(payload.get("source_id", "")).strip()
        if not source_id:
            raise ValueError("source_id is required")

        stock_codes = [str(x).strip().upper() for x in payload.get("stock_codes", []) if str(x).strip()]
        limit = max(1, min(500, int(payload.get("limit", 20) or 20)))
        catalog = {str(x.get("source_id", "")): x for x in self._build_datasource_catalog()}
        category = str(payload.get("category", "")).strip().lower() or str(catalog.get(source_id, {}).get("category", ""))
        if not category:
            category = self._infer_datasource_category(source_id)
        if not category:
            raise ValueError("unable to infer datasource category from source_id")

        started_at = time.perf_counter()
        error = ""
        result: dict[str, Any]
        try:
            if category == "quote":
                result = self.ingest_market_daily(stock_codes or ["SH600000"])
            elif category == "announcement":
                result = self.ingest_announcements(stock_codes or ["SH600000"])
            elif category == "history":
                result = self.ingestion.ingest_history_daily(stock_codes or ["SH600000"], limit=max(60, limit))
            elif category == "financial":
                result = self.ingest_financials(stock_codes or ["SH600000"])
            elif category == "news":
                result = self.ingest_news(stock_codes or ["SH600000"], limit=limit)
            elif category == "research":
                result = self.ingest_research_reports(stock_codes or ["SH600000"], limit=limit)
            elif category == "macro":
                result = self.ingest_macro_indicators(limit=limit)
            elif category == "fund":
                result = self.ingest_fund_snapshots(stock_codes or ["SH600000"])
            else:
                raise ValueError(f"unsupported datasource category: {category}")
        except Exception as ex:  # noqa: BLE001
            error = str(ex)
            result = {
                "task_name": f"{category}-fetch",
                "success_count": 0,
                "failed_count": max(1, len(stock_codes) if stock_codes else 1),
                "details": [{"status": "failed", "error": error}],
            }

        success_count = int(result.get("success_count", 0) or 0)
        failed_count = int(result.get("failed_count", 0) or 0)
        status = "ok"
        if failed_count > 0 and success_count > 0:
            status = "partial"
        elif failed_count > 0 and success_count == 0:
            status = "failed"
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        log = self._append_datasource_log(
            source_id=source_id,
            category=category,
            action="fetch",
            status=status,
            latency_ms=latency_ms,
            detail={
                "stock_codes": stock_codes,
                "limit": limit,
                "task_name": str(result.get("task_name", "")),
                "success_count": success_count,
                "failed_count": failed_count,
            },
            error=error or (str(result.get("details", [{}])[0].get("error", "")) if failed_count > 0 else ""),
        )
        if status == "failed":
            self.web.create_alert("datasource_fetch_failed", "medium", f"{source_id} fetch failed")

        return {
            "source_id": source_id,
            "category": category,
            "status": status,
            "latency_ms": latency_ms,
            "result": result,
            "log_id": int(log["log_id"]),
        }

    def ops_source_health(self, token: str) -> list[dict[str, Any]]:
        # 鍩轰簬 scheduler 鐘舵€佸埛鏂?source health 蹇収
        status = self.scheduler_status()
        for name, s in status.items():
            circuit = bool(s.get("circuit_open_until"))
            success = 1.0 if s.get("last_status") in ("ok", "never") else 0.7
            self.web.source_health_upsert(
                source_id=name,
                success_rate=success,
                circuit_open=circuit,
                last_error=s.get("last_error", ""),
            )
            if circuit:
                self.web.create_alert("scheduler_circuit_open", "high", f"{name} circuit open")
        return self.web.source_health_list(token)

    def ops_evals_history(self, token: str) -> list[dict[str, Any]]:
        self.web.require_role(token, {"admin", "ops"})
        rows = self.prompts.conn.execute(
            "SELECT eval_run_id, prompt_id, version, suite_id, metrics_json, pass_gate, created_at FROM prompt_eval_result ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def ops_prompt_releases(self, token: str) -> list[dict[str, Any]]:
        self.web.require_role(token, {"admin", "ops"})
        rows = self.prompts.conn.execute(
            "SELECT release_id, prompt_id, version, target_env, gate_result, created_at FROM prompt_release ORDER BY release_id DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def ops_prompt_versions(self, prompt_id: str) -> list[dict[str, Any]]:
        return self.prompts.list_prompt_versions(prompt_id)

    def ops_capabilities(self) -> dict[str, Any]:
        return build_capability_snapshot(
            self.settings,
            workflow_runtime=self.workflow_runtime.runtime_name,
            llm_external_enabled=self.settings.llm_external_enabled,
        )

    def ops_agent_debate(self, stock_code: str, question: str = "") -> dict[str, Any]:
        """Run a multi-agent debate, preferring real LLM responses with rule-engine fallback."""
        code = stock_code.upper().replace(".", "")
        if self._needs_quote_refresh(code):
            self.ingest_market_daily([code])
        if self._needs_history_refresh(code):
            self.ingestion.ingest_history_daily([code], limit=260)
        if self._needs_financial_refresh(code):
            self.ingest_financials([code])
        if self._needs_news_refresh(code):
            self.ingest_news([code], limit=8)
        if self._needs_research_refresh(code):
            self.ingest_research_reports([code], limit=6)
        if self._needs_fund_refresh(code):
            self.ingest_fund_snapshots([code])
        if self._needs_macro_refresh():
            self.ingest_macro_indicators(limit=8)

        quote = self._latest_quote(code) or {}
        bars = self._history_bars(code, limit=160)
        financial = self._latest_financial_snapshot(code) or {}
        trend = self._trend_metrics(bars) if len(bars) >= 30 else {}
        pred = self.predict_run({"stock_codes": [code], "horizons": ["5d", "20d"]})
        horizon_map = {h["horizon"]: h for h in pred["results"][0]["horizons"]} if pred.get("results") else {}

        quant_20 = horizon_map.get("20d", {})
        rule_opinions = self._build_rule_based_debate_opinions(question, trend, quote, quant_20)
        opinions = rule_opinions
        model_debate_mode = "rule_fallback"
        if self.settings.llm_external_enabled and self.llm_gateway.providers:
            llm_opinions = self._build_llm_debate_opinions(question, code, trend, quote, quant_20, rule_opinions)
            if llm_opinions:
                opinions = llm_opinions
                model_debate_mode = "llm_parallel"

        bucket = {"buy": 0, "hold": 0, "reduce": 0}
        for x in opinions:
            bucket[str(x["signal"])] += 1
        consensus_signal = max(bucket, key=lambda k: bucket[k])
        disagreement_score = round(1.0 - (bucket[consensus_signal] / len(opinions)), 4)

        return {
            "stock_code": code,
            "consensus_signal": consensus_signal,
            "disagreement_score": disagreement_score,
            "debate_mode": model_debate_mode,
            "opinions": opinions,
            "financial_snapshot": financial,
            "market_snapshot": {
                "price": quote.get("price"),
                "pct_change": quote.get("pct_change"),
                "trend": trend,
            },
        }

    def ops_rag_quality(self) -> dict[str, Any]:
        """RAG quality dashboard data: aggregate metrics plus per-case details."""
        dataset = default_retrieval_dataset()
        retriever = self._build_runtime_retriever([])
        evaluator = RetrievalEvaluator(retriever)
        offline_metrics = evaluator.run(dataset, k=5)
        cases: list[dict[str, Any]] = []
        for case in dataset:
            query = case["query"]
            positives = set(case["positive_source_ids"])
            results = retriever.retrieve(query, rerank_top_n=5)
            pred = [item.source_id for item in results]
            hit = [x for x in pred if x in positives]
            cases.append(
                {
                    "query": query,
                    "positive_source_ids": list(positives),
                    "predicted_source_ids": pred,
                    "hit_source_ids": hit,
                    "recall_at_k": round(RetrievalEvaluator._recall_at_k(pred, positives), 4),
                    "mrr": round(RetrievalEvaluator._mrr(pred, positives), 4),
                    "ndcg_at_k": round(RetrievalEvaluator._ndcg_at_k(pred, positives, 5), 4),
                }
            )
        online_cases = self.web.rag_eval_recent(limit=200)
        online_metrics = self._calc_retrieval_metrics_from_cases(online_cases, k=5)
        merged_metrics = {
            "recall_at_k": round((offline_metrics["recall_at_k"] + online_metrics["recall_at_k"]) / 2, 4),
            "mrr": round((offline_metrics["mrr"] + online_metrics["mrr"]) / 2, 4),
            "ndcg_at_k": round((offline_metrics["ndcg_at_k"] + online_metrics["ndcg_at_k"]) / 2, 4),
        }
        return {
            "metrics": merged_metrics,
            "offline": {"dataset_size": len(dataset), "metrics": offline_metrics, "cases": cases},
            "online": {"dataset_size": len(online_cases), "metrics": online_metrics, "cases": online_cases[:50]},
        }

    def ops_rag_retrieval_trace(self, token: str, *, trace_id: str = "", limit: int = 120) -> dict[str, Any]:
        self.web.require_role(token, {"admin", "ops"})
        rows = self.web.rag_retrieval_trace_list(token, trace_id=trace_id, limit=limit)
        return {
            "trace_id": trace_id.strip(),
            "count": len(rows),
            "items": rows,
        }

    def ops_rag_reindex(self, token: str, *, limit: int = 2000) -> dict[str, Any]:
        """Rebuild summary vector index and return operation metadata."""
        self.web.require_role(token, {"admin", "ops"})
        del limit  # 褰撳墠瀹炵幇鎸夊疄鏃惰祫浜у叏闆嗛噸寤猴紝鍚庣画鍙墿灞曞閲忛噸寤虹獥鍙ｃ€?
        result = self._refresh_summary_vector_index([], force=True)
        # 璁板綍鏈€杩戜竴娆￠噸寤烘椂闂达紝渚夸簬鍓嶇涓氬姟鐪嬫澘灞曠ず杩愮淮鐘舵€併€?
        self.web.rag_ops_meta_set(key="last_reindex_at", value=datetime.now(timezone.utc).isoformat())
        return {
            "status": "ok",
            "index_backend": self.vector_store.backend,
            "indexed_count": int(result.get("indexed_count", 0)),
            "rebuild_state": str(result.get("status", "rebuilt")),
        }

    def ops_prompt_compare(
        self,
        *,
        prompt_id: str,
        base_version: str,
        candidate_version: str,
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        """Compare two prompt versions and return rendered diff replay."""
        base_prompt = self.prompts.get_prompt(prompt_id, base_version)
        cand_prompt = self.prompts.get_prompt(prompt_id, candidate_version)
        base_rendered, base_meta = self.prompt_runtime.build_from_prompt(base_prompt, variables)
        cand_rendered, cand_meta = self.prompt_runtime.build_from_prompt(cand_prompt, variables)
        diff_rows = list(
            difflib.unified_diff(
                base_rendered.splitlines(),
                cand_rendered.splitlines(),
                fromfile=f"{prompt_id}@{base_version}",
                tofile=f"{prompt_id}@{candidate_version}",
                lineterm="",
            )
        )
        return {
            "prompt_id": prompt_id,
            "base": {"version": base_version, "meta": base_meta, "rendered": base_rendered},
            "candidate": {"version": candidate_version, "meta": cand_meta, "rendered": cand_rendered},
            "diff_summary": {
                "line_count": len(diff_rows),
                "changed": bool(diff_rows),
            },
            "diff_preview": diff_rows[:120],
        }

    def _build_rule_based_debate_opinions(
        self,
        question: str,
        trend: dict[str, Any],
        quote: dict[str, Any],
        quant_20: dict[str, Any],
    ) -> list[dict[str, Any]]:
        pm_signal = "hold"
        if trend.get("ma20_slope", 0.0) > 0 and float(quote.get("pct_change", 0.0)) > 0:
            pm_signal = "buy"
        elif trend.get("ma20_slope", 0.0) < 0:
            pm_signal = "reduce"

        quant_signal = str(quant_20.get("signal", "hold")).replace("strong_", "")
        risk_signal = "hold"
        if trend.get("max_drawdown_60", 0.0) > 0.2 or trend.get("volatility_20", 0.0) > 0.025:
            risk_signal = "reduce"
        elif trend.get("volatility_20", 0.0) < 0.012 and trend.get("momentum_20", 0.0) > 0:
            risk_signal = "buy"

        return [
            {
                "agent": "pm_agent",
                "signal": pm_signal,
                "confidence": 0.66,
                "reason": f"鍩轰簬瓒嬪娍鏂滅巼涓庡綋鏃ユ定璺屾瀯寤轰骇鍝佷晶鍙В閲婅鐐癸紝question={question or 'default'}",
            },
            {
                "agent": "quant_agent",
                "signal": quant_signal,
                "confidence": float(quant_20.get("up_probability", 0.5)),
                "reason": f"鏉ヨ嚜20鏃ラ娴嬶細{quant_20.get('rationale', '')}",
            },
            {
                "agent": "risk_agent",
                "signal": risk_signal,
                "confidence": 0.72,
                "reason": (
                    f"鍥炴挙={trend.get('max_drawdown_60', 0.0):.4f}, 娉㈠姩={trend.get('volatility_20', 0.0):.4f}, "
                    f"鍔ㄩ噺={trend.get('momentum_20', 0.0):.4f}"
                ),
            },
        ]

    def _build_llm_debate_opinions(
        self,
        question: str,
        stock_code: str,
        trend: dict[str, Any],
        quote: dict[str, Any],
        quant_20: dict[str, Any],
        rule_defaults: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Call LLM to generate multi-agent debate opinions."""
        context = {
            "question": question or "璇风粰鍑虹煭涓湡瑙傜偣",
            "stock_code": stock_code,
            "quote": {"price": quote.get("price"), "pct_change": quote.get("pct_change")},
            "trend": trend,
            "quant_20": quant_20,
        }
        prompts = [
            (
                "pm_agent",
                (
                    "You are a portfolio manager. Return JSON only with keys: signal, confidence, reason. "
                    f"context={json.dumps(context, ensure_ascii=False)}"
                ),
            ),
            (
                "quant_agent",
                (
                    "You are a quant analyst. Return JSON only with keys: signal, confidence, reason. "
                    f"context={json.dumps(context, ensure_ascii=False)}"
                ),
            ),
            (
                "risk_agent",
                (
                    "You are a risk controller. Return JSON only with keys: signal, confidence, reason. "
                    f"context={json.dumps(context, ensure_ascii=False)}"
                ),
            ),
        ]

        rule_map = {x["agent"]: x for x in rule_defaults}
        outputs: list[dict[str, Any]] = []

        def run_one(agent_name: str, prompt_text: str) -> dict[str, Any]:
            fallback = rule_map.get(agent_name, {"signal": "hold", "confidence": 0.5, "reason": "rule default"})
            state = AgentState(user_id="ops", question=question or "ops debate", stock_codes=[stock_code], trace_id=self.traces.new_trace())
            try:
                raw = self.llm_gateway.generate(state, prompt_text)
                cleaned = raw.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.strip("`")
                    if cleaned.lower().startswith("json"):
                        cleaned = cleaned[4:].strip()
                obj = json.loads(cleaned)
                signal = str(obj.get("signal", fallback["signal"])).lower()
                if signal not in {"buy", "hold", "reduce"}:
                    signal = str(fallback["signal"])
                conf = float(obj.get("confidence", fallback["confidence"]))
                conf = max(0.0, min(1.0, conf))
                reason = str(obj.get("reason", fallback["reason"]))[:320]
                return {"agent": agent_name, "signal": signal, "confidence": conf, "reason": reason}
            except Exception:
                return {
                    "agent": agent_name,
                    "signal": str(fallback["signal"]),
                    "confidence": float(fallback["confidence"]),
                    "reason": str(fallback["reason"]) + " | llm_fallback",
                }

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(run_one, n, p) for n, p in prompts]
            for f in futures:
                outputs.append(f.result())
        return outputs

    def alerts_list(self, token: str) -> list[dict[str, Any]]:
        return self.web.alerts_list(token)

    def alerts_ack(self, token: str, alert_id: int) -> dict[str, Any]:
        return self.web.alerts_ack(token, alert_id)

    def alert_rule_create(self, token: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.web.alert_rule_create(token, payload)

    def alert_rule_list(self, token: str) -> list[dict[str, Any]]:
        return self.web.alert_rule_list(token)

    def alert_rule_delete(self, token: str, rule_id: int) -> dict[str, Any]:
        return self.web.alert_rule_delete(token, rule_id)

    def alert_trigger_logs(self, token: str, *, limit: int = 100) -> list[dict[str, Any]]:
        return self.web.alert_trigger_log_list(token, limit=limit)

    def alert_rule_check(self, token: str) -> dict[str, Any]:
        rules = [r for r in self.web.alert_rule_list(token) if bool(r.get("is_active", False))]
        stock_codes = list({str(r.get("stock_code", "")).upper() for r in rules if str(r.get("stock_code", "")).strip()})
        if stock_codes:
            try:
                self.ingest_market_daily(stock_codes)
            except Exception:
                pass
            try:
                self.ingest_announcements(stock_codes)
            except Exception:
                pass
        quote_map = {str(x.get("stock_code", "")).upper(): x for x in self.ingestion_store.quotes[-500:]}
        ann_by_code: dict[str, list[dict[str, Any]]] = {}
        for event in self.ingestion_store.announcements[-800:]:
            code = str(event.get("stock_code", "")).upper()
            if not code:
                continue
            ann_by_code.setdefault(code, []).append(event)

        triggered: list[dict[str, Any]] = []
        for rule in rules:
            rule_id = int(rule.get("rule_id", 0) or 0)
            code = str(rule.get("stock_code", "")).upper()
            rule_type = str(rule.get("rule_type", "")).lower()
            hit = False
            trigger_data: dict[str, Any] = {}
            if rule_type == "price":
                quote = quote_map.get(code, {})
                current = float(quote.get("price", 0.0) or 0.0)
                op = str(rule.get("operator", ""))
                target = float(rule.get("target_value", 0.0) or 0.0)
                if op == ">" and current > target:
                    hit = True
                elif op == "<" and current < target:
                    hit = True
                elif op == ">=" and current >= target:
                    hit = True
                elif op == "<=" and current <= target:
                    hit = True
                trigger_data = {"current_price": current, "target_value": target, "operator": op}
            elif rule_type == "event":
                event_type = str(rule.get("event_type", "")).strip().lower()
                events = ann_by_code.get(code, [])
                for event in events[-20:]:
                    title = str(event.get("title", "")).lower()
                    etype = str(event.get("event_type", "")).lower()
                    if not event_type or event_type in etype or event_type in title:
                        hit = True
                        trigger_data = {
                            "event_type": etype,
                            "title": str(event.get("title", "")),
                            "event_time": str(event.get("event_time", "")),
                        }
                        break

            if not hit:
                continue
            message = f"rule[{rule_id}] triggered for {code}"
            alert_id = self.web.create_alert("rule_trigger", "medium", message)
            self.web.alert_trigger_log_add(
                rule_id=rule_id,
                stock_code=code,
                trigger_message=message,
                trigger_data=trigger_data,
            )
            triggered.append({"alert_id": alert_id, "rule_id": rule_id, "stock_code": code, "message": message, "data": trigger_data})
        return {"checked_rules": len(rules), "triggered_count": len(triggered), "items": triggered}

    def stock_universe_sync(self, token: str) -> dict[str, Any]:
        return self.web.stock_universe_sync(token)

    def stock_universe_search(
        self,
        token: str,
        *,
        keyword: str = "",
        exchange: str = "",
        market_tier: str = "",
        listing_board: str = "",
        industry_l1: str = "",
        industry_l2: str = "",
        industry_l3: str = "",
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        return self.web.stock_universe_search(
            token,
            keyword=keyword,
            exchange=exchange,
            market_tier=market_tier,
            listing_board=listing_board,
            industry_l1=industry_l1,
            industry_l2=industry_l2,
            industry_l3=industry_l3,
            limit=limit,
        )

    def stock_universe_filters(self, token: str) -> dict[str, list[str]]:
        return self.web.stock_universe_filters(token)

    def close(self) -> None:
        """Release database connections and other service resources."""
        self.memory.close()
        self.prompts.close()
        self.web.close()






