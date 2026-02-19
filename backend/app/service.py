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
from concurrent.futures import ThreadPoolExecutor
import time
import uuid
from datetime import datetime, timezone
from statistics import mean
from typing import Any

from backend.app.capabilities import build_capability_snapshot
from backend.app.agents.langgraph_runtime import build_workflow_runtime
from backend.app.agents.workflow import AgentWorkflow
from backend.app.config import Settings
from backend.app.data.ingestion import IngestionService, IngestionStore
from backend.app.data.scheduler import JobConfig, LocalJobScheduler
from backend.app.data.sources import AnnouncementService, QuoteService
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


class AShareAgentService:
    """应用服务层。

    负责聚合 API 所需能力，可类比 Java 的 Facade + ApplicationService。
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """初始化各子系统依赖。"""
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
        # Phase-B 向量检索组件：支持可配置 embedding provider + 本地向量索引。
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
            # DATA-001: 真实源优先，失败后自动进入 mock 兜底回退。
            quote_service=QuoteService.build_default(),
            announcement_service=AnnouncementService(),
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

        # 报告存储：MVP 先使用内存字典
        self._reports: dict[str, dict[str, Any]] = {}
        self._backtest_runs: dict[str, dict[str, Any]] = {}
        self._deep_archive_ts_format = "%Y-%m-%d %H:%M:%S"
        self._deep_archive_export_executor = ThreadPoolExecutor(max_workers=2)
        self._deep_round_mutex = threading.Lock()
        self._deep_round_inflight: set[str] = set()
        self._register_default_agent_cards()

    def _select_runtime(self, preference: str | None = None):
        """按请求参数选择运行时：langgraph/direct/auto。"""
        pref = (preference or "").strip().lower()
        if pref in ("", "auto", "default"):
            return self.workflow_runtime
        if pref == "direct":
            return build_workflow_runtime(self.workflow, prefer_langgraph=False)
        if pref == "langgraph":
            return build_workflow_runtime(self.workflow, prefer_langgraph=True)
        return self.workflow_runtime

    def _register_default_jobs(self) -> None:
        """注册默认调度任务（对应 DATA-004）。"""

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
        """注册内置多 Agent 配置，供内部 A2A 适配层发现能力。"""
        cards = [
            ("supervisor_agent", "Supervisor", "负责任务拆解、轮次推进与仲裁", ["plan", "route", "arbitrate"]),
            ("pm_agent", "PM Agent", "关注主题、叙事与产品侧解释", ["theme_analysis", "narrative"]),
            ("quant_agent", "Quant Agent", "关注因子、概率与收益风险比", ["factor_analysis", "probability"]),
            ("risk_agent", "Risk Agent", "关注波动、回撤与下行风险", ["risk_scoring", "drawdown_check"]),
            ("critic_agent", "Critic Agent", "质检证据完整性与逻辑一致性", ["consistency_check", "counter_view"]),
            ("macro_agent", "Macro Agent", "关注政策与宏观环境冲击", ["macro_event", "policy_watch"]),
            ("execution_agent", "Execution Agent", "关注仓位节奏与执行约束", ["execution_plan", "position_sizing"]),
            ("compliance_agent", "Compliance Agent", "关注合规边界与敏感表述", ["compliance_check", "policy_block"]),
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
        counter_view = str(counter_candidates[0]["reason"]) if counter_candidates else "无显著反方观点"

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
        q = question.lower()
        tasks: list[dict[str, Any]] = [
            {"task_id": f"r{round_no}-t1", "agent": "quant_agent", "title": "估值与收益风险比评估", "priority": "high"},
            {"task_id": f"r{round_no}-t2", "agent": "risk_agent", "title": "回撤与波动风险评估", "priority": "high"},
            {"task_id": f"r{round_no}-t3", "agent": "pm_agent", "title": "主题与叙事一致性评估", "priority": "medium"},
            {"task_id": f"r{round_no}-t4", "agent": "compliance_agent", "title": "合规边界审查", "priority": "high"},
        ]
        if any(k in q for k in ("宏观", "政策", "利率", "财政")):
            tasks.append({"task_id": f"r{round_no}-t5", "agent": "macro_agent", "title": "宏观与政策冲击评估", "priority": "high"})
        if any(k in q for k in ("执行", "仓位", "节奏", "交易")):
            tasks.append({"task_id": f"r{round_no}-t6", "agent": "execution_agent", "title": "执行节奏与仓位约束评估", "priority": "medium"})
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
        """从模型文本中提取 JSON 对象，兼容 fenced code block 与前后杂文本。"""
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
        # 宽松兜底：截取第一个 JSON 对象片段再解析。
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
        """校验并标准化 LLM WebSearch 情报输出，防止前端消费不稳定结构。"""
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
            # 诊断字段：统一返回，便于前端明确“是否命中外部实时情报”与失败原因。
            "intel_status": str(payload.get("intel_status", "external_ok")).strip() or "external_ok",
            "fallback_reason": str(payload.get("fallback_reason", "")).strip()[:120],
            "fallback_error": str(payload.get("fallback_error", "")).strip()[:320],
            "trace_id": str(payload.get("trace_id", "")).strip()[:80],
            "external_enabled": bool(payload.get("external_enabled", True)),
            # provider_count 为诊断字段，解析失败时回退 0，避免影响主流程。
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
            # 兼容模型输出文本强弱（例如 "down"/"up"），避免 float 强转异常导致整段降级。
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
                    # 若文本里含数字（如 "-0.1"、"0.08"），尝试提取首个浮点值。
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
        """根据异常文本映射稳定原因码，避免前端只能看到含糊报错。"""
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
        """返回当前已启用 provider 名单，用于诊断输出。"""
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
        """当外部 WebSearch 不可用时，用本地信号生成可解释的降级情报。"""
        providers = list(provider_names or self._deep_enabled_provider_names())
        ann = [x for x in self.ingestion_store.announcements if str(x.get("stock_code", "")).upper() == stock_code][-3:]
        calendar_items = [
            {
                "title": "央行议息窗口（关注流动性预期）",
                "summary": "建议在会议前后观察利率预期变化对估值与风险偏好的影响。",
                "impact_direction": "uncertain",
                "impact_horizon": "1w",
                "why_relevant_to_stock": f"{stock_code} 对市场风险偏好与资金成本变化较敏感。",
                "url": "",
                "published_at": datetime.now(timezone.utc).isoformat(),
                "source_type": "other",
            }
        ]
        for item in ann:
            calendar_items.append(
                {
                    "title": str(item.get("title", "公司公告")).strip()[:220],
                    "summary": str(item.get("content", "建议核对公告正文")).strip()[:320],
                    "impact_direction": "mixed",
                    "impact_horizon": "1w",
                    "why_relevant_to_stock": f"{stock_code} 直接公告事项，需核实披露细节。",
                    "url": str(item.get("source_url", "")).strip()[:420],
                    "published_at": str(item.get("event_time", "")).strip()[:64],
                    "source_type": "official",
                }
            )
        bias = "hold"
        if trend.get("momentum_20", 0.0) > 0 and float(quote.get("pct_change", 0.0)) > 0:
            bias = "buy"
        elif trend.get("max_drawdown_60", 0.0) > 0.2:
            bias = "reduce"
        return {
            "as_of": datetime.now(timezone.utc).isoformat(),
            "macro_signals": [
                {
                    "title": "本地降级情报模式",
                    "summary": "未获取外部 WebSearch 实时情报，当前结论仅基于本地行情、趋势与公告快照。",
                    "impact_direction": "uncertain",
                    "impact_horizon": "1w",
                    "why_relevant_to_stock": f"问题：{question[:120]}",
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
                    "event": "本地降级模式",
                    "transmission_path": "数据可得性下降 -> 结论置信度下降",
                    "industry_impact": "行业级外部冲击信息缺失",
                    "stock_impact": f"{stock_code} 的事件前瞻判断能力下降",
                    "price_factor": "短期波动解释能力下降",
                }
            ],
            "decision_adjustment": {
                "signal_bias": bias,
                "confidence_adjustment": -0.12,
                "rationale": "外部实时情报不可用，按降级策略下调置信度。",
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
        """构建 WebSearch 情报提示词，要求模型输出严格 JSON。"""
        context = {
            "stock_code": stock_code,
            "question": question,
            "quote": {"price": quote.get("price"), "pct_change": quote.get("pct_change")},
            "trend": trend,
            "quant_20": quant_20,
            "scope": "中国为主 + 全球关键事件",
            "lookback_hours": 72,
            "lookahead_days": 30,
        }
        return (
            "你是A股资深投研情报官。请使用模型的 web search 能力，"
            "检索并总结会影响该股票的宏观、行业、国际与未来会议事件。\n"
            "必须返回严格 JSON，不要返回 markdown，不要返回多余文本。\n"
            "要求：\n"
            "1) 每条情报要有 title/summary/impact_direction/impact_horizon/why_relevant_to_stock/url/published_at/source_type。\n"
            "2) 覆盖字段：macro_signals, industry_forward_events, stock_specific_catalysts, calendar_watchlist。\n"
            "3) 给出 impact_chain（事件到股价因子的传导链）。\n"
            "4) 给出 decision_adjustment: {signal_bias, confidence_adjustment, rationale}。\n"
            "5) 给出 citations，至少 3 条，优先官方与主流媒体。\n"
            "JSON schema:\n"
            "{"
            "\"as_of\":\"...\","
            "\"macro_signals\":[],"
            "\"industry_forward_events\":[],"
            "\"stock_specific_catalysts\":[],"
            "\"calendar_watchlist\":[],"
            "\"impact_chain\":[],"
            "\"decision_adjustment\":{},"
            "\"citations\":[],"
            "\"confidence_note\":\"...\""
            "}\n"
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
        """通过 LLM WebSearch 拉取实时情报；失败时回落到本地降级情报。"""
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
        # 显式要求 Responses API 挂载 web-search tool，避免“仅靠提示词触发搜索”的不确定性。
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
                # 若 provider 不支持 tools 字段，降级为“prompt-only”尝试，避免完全不可用。
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
            # 保障至少有一条可追溯引用，避免“无来源高确信”。
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
        """将仲裁结果与情报层融合成业务可执行摘要。"""
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
            "trigger_condition": str(decision.get("rationale", "关注情报催化与趋势共振。"))[:280],
            "invalidation_condition": (
                "若关键风险事件落地偏负面、分歧持续扩大或预算风控触发，则信号失效。"
            ),
            "review_time_hint": str(next_event.get("published_at", "")) or "T+1 日内复核",
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

    @staticmethod
    def _deep_build_analysis_dimensions(
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
                "summary": str(pm_op.get("reason", "行业景气与叙事需结合基本面验证。"))[:180],
                "signals": [str(x.get("title", ""))[:60] for x in industry_events[:3] if str(x.get("title", "")).strip()],
            },
            {
                "dimension": "competition",
                "score": round(float(critic_op.get("confidence", 0.5) or 0.5), 4),
                "summary": "竞争格局需结合份额变化与盈利能力对比。",
                "signals": [str(x.get("summary", ""))[:60] for x in catalysts[:2] if str(x.get("summary", "")).strip()],
            },
            {
                "dimension": "supply_chain",
                "score": round(float(macro_op.get("confidence", 0.5) or 0.5), 4),
                "summary": "关注上下游传导链路与成本压力释放。",
                "signals": [str(x.get("to", ""))[:60] for x in impact_chain[:3] if str(x.get("to", "")).strip()],
            },
            {
                "dimension": "risk",
                "score": round(float(risk_op.get("confidence", 0.5) or 0.5), 4),
                "summary": str(risk_op.get("reason", "优先关注回撤、波动与下行尾部风险。"))[:180],
                "signals": [str(regime.get("risk_bias", ""))[:40]],
            },
            {
                "dimension": "macro",
                "score": round(float(regime.get("regime_confidence", 0.0) or 0.0), 4),
                "summary": str(regime.get("regime_rationale", "宏观与政策节奏将影响风险偏好。"))[:180],
                "signals": [str(x.get("title", ""))[:60] for x in list(intel.get("macro_signals", []))[:3] if str(x.get("title", "")).strip()],
            },
            {
                "dimension": "execution",
                "score": round(float(exec_op.get("confidence", 0.5) or 0.5), 4),
                "summary": str(exec_op.get("reason", "执行层需控制仓位节奏与流动性冲击。"))[:180],
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
            # Real data first: refresh quote/announcement/history before retrieval.
            if req.stock_codes:
                try:
                    quote_refresh = [c for c in req.stock_codes if self._needs_quote_refresh(c)]
                    ann_refresh = [c for c in req.stock_codes if self._needs_announcement_refresh(c)]
                    hist_refresh = [c for c in req.stock_codes if self._needs_history_refresh(c)]
                    if quote_refresh:
                        self.ingest_market_daily(quote_refresh)
                    if ann_refresh:
                        self.ingest_announcements(ann_refresh)
                    if hist_refresh:
                        self.ingestion.ingest_history_daily(hist_refresh, limit=260)
                except Exception:
                    # Keep query available even when data refresh partially fails.
                    pass

            self.workflow.retriever = self._build_runtime_retriever(req.stock_codes)
            enriched_question = self._augment_question_with_history_context(req.question, req.stock_codes)
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
            analysis_brief = self._build_analysis_brief(req.stock_codes, merged_citations, regime_context=regime_context)
            state.report = answer
            state.citations = merged_citations
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
            "本次查询触发系统降级，已返回最小可用结果。"
            "请稍后重试，或缩小问题范围（减少股票数量/问题长度）。"
            f" 错误码：{error_code}。"
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
        """以事件流形式输出问答结果，便于前端逐段渲染。"""
        _ = chunk_size
        started_at = time.perf_counter()
        req = QueryRequest(**payload)
        selected_runtime = self._select_runtime(str(payload.get("workflow_runtime", "")))
        # 先发送 start，确保前端在任何耗时步骤前就能感知“任务已启动”。
        yield {"event": "start", "data": {"status": "started", "phase": "init"}}
        # 与同步 query 保持一致：先做数据刷新与动态语料装载。
        if req.stock_codes:
            yield {"event": "progress", "data": {"phase": "data_refresh", "message": "正在刷新行情/公告/历史数据"}}
            try:
                quote_refresh = [c for c in req.stock_codes if self._needs_quote_refresh(c)]
                ann_refresh = [c for c in req.stock_codes if self._needs_announcement_refresh(c)]
                hist_refresh = [c for c in req.stock_codes if self._needs_history_refresh(c)]
                if quote_refresh:
                    self.ingest_market_daily(quote_refresh)
                if ann_refresh:
                    self.ingest_announcements(ann_refresh)
                if hist_refresh:
                    self.ingestion.ingest_history_daily(hist_refresh, limit=260)
            except Exception:
                # 刷新失败不阻断主流程，只向前端透出 warning 便于定位时延与降级行为。
                yield {"event": "progress", "data": {"phase": "data_refresh", "status": "degraded"}}
                pass
            yield {"event": "progress", "data": {"phase": "data_refresh", "status": "done"}}
        yield {"event": "progress", "data": {"phase": "retriever", "message": "正在准备检索语料"}}
        self.workflow.retriever = self._build_runtime_retriever(req.stock_codes)
        enriched_question = self._augment_question_with_history_context(req.question, req.stock_codes)
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
        yield {"event": "progress", "data": {"phase": "model", "message": "开始模型流式输出"}}
        # 为了避免“模型首 token 迟迟不来时前端无反馈”，
        # 在独立线程消费 runtime 事件，并在主协程周期性发送 model_wait 心跳。
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
                        "message": "模型推理中，等待首个增量输出",
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
        citations = [c for c in state.citations if isinstance(c, dict)]
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
        # 结果沉淀事件：便于前端/运维确认本轮问答已进入共享语料池。
        yield {"event": "knowledge_persisted", "data": {"trace_id": trace_id}}
        yield {
            "event": "analysis_brief",
            "data": self._build_analysis_brief(req.stock_codes, citations, regime_context=regime_context),
        }

    def _build_evidence_rich_answer(
        self,
        question: str,
        stock_codes: list[str],
        base_answer: str,
        citations: list[dict[str, Any]],
        regime_context: dict[str, Any] | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """生成带数据支撑的增强回答（实时+历史+双Agent讨论）。"""
        lines: list[str] = []
        merged = list(citations)

        lines.append("## Conclusion Summary")
        lines.append("## 结论摘要")
        lines.append(base_answer)
        lines.append("")
        regime = regime_context if isinstance(regime_context, dict) else {}
        if regime:
            lines.append(
                "## A-share Regime Snapshot"
            )
            lines.append(
                "- "
                f"label=`{regime.get('regime_label', 'unknown')}` | "
                f"confidence=`{float(regime.get('regime_confidence', 0.0) or 0.0):.2f}` | "
                f"risk_bias=`{regime.get('risk_bias', 'neutral')}`"
            )
            lines.append(f"- rationale: {regime.get('regime_rationale', '')}")
        lines.append("## Data Snapshot and History Trend")
        lines.append("## 数据证据分析")
        for code in stock_codes:
            realtime = self._latest_quote(code)
            # 问答主链路尽量使用更长窗口，减少“样本稀疏”判断误差。
            bars = self._history_bars(code, limit=260)
            summary_3m = self._history_3m_summary(code)
            if not realtime or len(bars) < 30:
                lines.append(f"- {code}: 数据不足（实时或历史样本不足），仅给出保守判断。 [insufficient_data]")
                continue
            trend = self._trend_metrics(bars)
            lines.append(
                f"- {code}: 最新价 `{realtime['price']:.3f}`，当日涨跌 `{realtime['pct_change']:.2f}%`，"
                f"20日趋势 `{trend['ma20_slope']:.4f}`，60日趋势 `{trend['ma60_slope']:.4f}`，"
                f"20日波动 `{trend['volatility_20']:.4f}`。"
            )
            lines.append(
                f"  解释: MA20>{'MA60' if trend['ma20'] >= trend['ma60'] else 'MA60以下'}，"
                f"近20日动量 `{trend['momentum_20']:.4f}`，最大回撤 `{trend['max_drawdown_60']:.4f}`。"
            )
            if int(summary_3m.get("sample_count", 0)) >= 60:
                lines.append(
                    "  三个月窗口: "
                    f"`{summary_3m.get('start_date','')}` -> `{summary_3m.get('end_date','')}`，"
                    f"连续样本 `{int(summary_3m.get('sample_count', 0))}` 条，"
                    f"收盘 `{float(summary_3m.get('start_close', 0.0)):.3f}` -> "
                    f"`{float(summary_3m.get('end_close', 0.0)):.3f}`，"
                    f"区间 `{float(summary_3m.get('pct_change', 0.0)) * 100:.2f}%`。"
                )
            merged.append(
                {
                    "source_id": realtime.get("source_id", "unknown"),
                    "source_url": realtime.get("source_url", ""),
                    "event_time": realtime.get("ts"),
                    "reliability_score": realtime.get("reliability_score", 0.7),
                    "excerpt": f"{code} 实时: price={realtime['price']}, pct={realtime['pct_change']}",
                }
            )
            if bars:
                merged.append(
                    {
                        "source_id": bars[-1].get("source_id", "eastmoney_history"),
                        "source_url": bars[-1].get("source_url", ""),
                        "event_time": bars[-1].get("trade_date"),
                        "reliability_score": bars[-1].get("reliability_score", 0.9),
                        "excerpt": f"{code} 历史K线样本 {len(bars)} 条，最近交易日 {bars[-1].get('trade_date','')}",
                    }
                )
            # 额外补充三个月窗口首末点引用，避免模型仅看到“离散两点”却不知完整窗口样本量。
            if int(summary_3m.get("sample_count", 0)) >= 2:
                merged.append(
                    {
                        "source_id": "eastmoney_history_3m_window",
                        "source_url": bars[-1].get("source_url", "") if bars else "",
                        "event_time": summary_3m.get("end_date", ""),
                        "reliability_score": 0.92,
                        "excerpt": (
                            f"{code} 近3个月样本 {int(summary_3m.get('sample_count', 0))} 条，"
                            f"区间 {summary_3m.get('start_date','')}->{summary_3m.get('end_date','')}，"
                            f"收盘 {float(summary_3m.get('start_close', 0.0)):.3f}->{float(summary_3m.get('end_close', 0.0)):.3f}"
                        ),
                    }
                )

        pm_section = self._pm_agent_view(question, stock_codes)
        dev_section = self._dev_manager_view(stock_codes)
        lines.append("")
        lines.append("## PM Agent View")
        lines.append("## PM Agent 观点（产品侧）")
        lines.extend(pm_section)
        lines.append("")
        lines.append("## Dev Manager Agent View")
        lines.append("## 开发经理 Agent 观点（工程侧）")
        lines.extend(dev_section)

        # 业务可读性补充：告诉用户“本轮结论是否命中了共享知识资产”。
        shared_hits = [
            c
            for c in merged
            if str(c.get("source_id", "")).startswith("doc::")
            or str(c.get("source_id", "")) == "qa_memory_summary"
        ]
        lines.append("")
        lines.append("## Shared Knowledge Hits")
        lines.append("## 共享知识命中")
        if shared_hits:
            lines.append(f"- 命中条数: `{len(shared_hits)}`")
            for idx, item in enumerate(shared_hits[:4], start=1):
                lines.append(
                    f"- [{idx}] source=`{item.get('source_id','unknown')}` | "
                    f"{item.get('excerpt', '')[:140]}"
                )
        else:
            lines.append("- 本轮未命中共享知识资产，结论主要来自实时行情/公告/历史序列。")
        lines.append("")
        lines.append("## Evidence References")
        lines.append("## 引用清单")
        for idx, c in enumerate(merged[:10], start=1):
            lines.append(
                f"- [{idx}] `{c.get('source_id','unknown')}` | {c.get('source_url','')} | "
                f"score={float(c.get('reliability_score',0.0)):.2f} | {c.get('excerpt','')}"
            )
        return "\n".join(lines), merged[:10]

    def _build_analysis_brief(
        self,
        stock_codes: list[str],
        citations: list[dict[str, Any]],
        regime_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """构造结构化证据摘要，供前端可视化展示。"""
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
        """记录线上查询样本，供 RAG 持续评测使用。"""
        predicted = [str(x.get("source_id", "")) for x in evidence_pack[:5] if x.get("source_id")]
        positive = [str(x.get("source_id", "")) for x in citations if x.get("source_id")]
        # 保序去重
        pred_u = list(dict.fromkeys(predicted))
        pos_u = list(dict.fromkeys(positive))
        if not query_text.strip() or not pred_u:
            return
        self.web.rag_eval_add(query_text=query_text, positive_source_ids=pos_u, predicted_source_ids=pred_u)

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
        """记录召回/选用轨迹，便于线上排查“召回命中但答案未使用”问题。"""
        retrieved_ids = [
            str(x.get("source_id", "")).strip()
            for x in evidence_pack[:12]
            if str(x.get("source_id", "")).strip()
        ]
        selected_ids = [
            str(x.get("source_id", "")).strip()
            for x in citations[:12]
            if str(x.get("source_id", "")).strip()
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
        """把问答结果写入共享语料池（raw + redacted/summary 双轨）。"""
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
        """轻量质量评分：用于共享语料的上线门禁。"""
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
        """构造可检索摘要：优先结构化前缀 + 截断正文。"""
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
        """统一构建查询期 retriever：可按配置启用向量语义增强。"""
        corpus = self._build_runtime_corpus(stock_codes)
        if not self.settings.rag_vector_enabled:
            return HybridRetriever(corpus=corpus)
        self._refresh_summary_vector_index(stock_codes)

        def _semantic_search(query: str, top_k: int) -> list[RetrievalItem]:
            return self._semantic_summary_origin_hits(query, top_k=max(1, top_k))

        return HybridRetrieverV2(corpus=corpus, semantic_search_fn=_semantic_search)

    def _build_summary_vector_records(self, stock_codes: list[str]) -> list[VectorSummaryRecord]:
        """把持久化资产映射为“摘要索引记录”，用于 summary-first 检索。"""
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
        # 减少重复 rebuild：签名未变化时复用现有索引。
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
        """摘要先召回 -> 原文回补：返回 summary hit 与 origin backfill hit。"""
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
            # 回补原文：让最终证据更可读且可用于生成更完整答案。
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
        """把摄取层数据转换为检索语料。"""
        symbols = set(stock_codes)
        corpus = HybridRetriever()._default_corpus()

        # 行情快照 -> 文本事实
        for q in self.ingestion_store.quotes[-80:]:
            code = str(q.get("stock_code", ""))
            if symbols and code not in symbols:
                continue
            ts = self._parse_time(str(q.get("ts", "")))
            text = (
                f"{code} 行情快照：现价 {q.get('price')}，涨跌幅 {q.get('pct_change')}%，"
                f"成交量 {q.get('volume')}，成交额 {q.get('turnover')}。"
            )
            corpus.append(
                RetrievalItem(
                    text=text,
                    source_id=str(q.get("source_id", "unknown")),
                    source_url=str(q.get("source_url", "")),
                    score=0.0,
                    event_time=ts,
                    reliability_score=float(q.get("reliability_score", 0.6)),
                )
            )

        # 公告 -> 文本事实
        for a in self.ingestion_store.announcements[-120:]:
            code = str(a.get("stock_code", ""))
            if symbols and code not in symbols:
                continue
            event_time = self._parse_time(str(a.get("event_time", "")))
            text = f"{code} 公告：{a.get('title', '')}。{a.get('content', '')}"
            corpus.append(
                RetrievalItem(
                    text=text,
                    source_id=str(a.get("source_id", "announcement")),
                    source_url=str(a.get("source_url", "")),
                    score=0.0,
                    event_time=event_time,
                    reliability_score=float(a.get("reliability_score", 0.9)),
                )
            )

        # 历史K线 -> 文本事实（趋势查询时关键证据）
        history_by_code: dict[str, list[dict[str, Any]]] = {}
        for b in self.ingestion_store.history_bars[-2000:]:
            code = str(b.get("stock_code", ""))
            if symbols and code not in symbols:
                continue
            history_by_code.setdefault(code, []).append(b)
            text = (
                f"{code} 历史K线 {b.get('trade_date','')} 开{b.get('open')} 高{b.get('high')} "
                f"低{b.get('low')} 收{b.get('close')} 量{b.get('volume')}"
            )
            corpus.append(
                RetrievalItem(
                    text=text,
                    source_id=str(b.get("source_id", "eastmoney_history")),
                    source_url=str(b.get("source_url", "")),
                    score=0.0,
                    event_time=self._parse_time(str(b.get("trade_date", ""))),
                    reliability_score=float(b.get("reliability_score", 0.9)),
                )
            )
        # 为每个标的增加“最近三个月连续窗口摘要”，避免模型只看到离散点后误判样本稀疏。
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
                f"{code} 最近三个月连续日线样本={len(window)}，区间={start.get('trade_date')}到{end.get('trade_date')}，"
                f"收盘从{start_close:.3f}到{end_close:.3f}，区间涨跌={pct_change * 100:.2f}%。"
            )
            corpus.append(
                RetrievalItem(
                    text=text,
                    source_id="eastmoney_history_3m_window",
                    source_url=str(end.get("source_url", "")),
                    score=0.0,
                    event_time=self._parse_time(str(end.get("trade_date", ""))),
                    reliability_score=0.92,
                )
            )

        # 已索引文档分块 -> 文本事实
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
                    )
                )

        # 持久化文档 chunk（白名单/审核后生效）-> 在线检索语料。
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

        # 共享 QA 摘要语料（全站共享）-> 在线检索语料。
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
                        # 为 Phase-B“摘要召回 -> 原文回补”预留 parent 文本。
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
            # 支持 ISO 时间与 YYYY-MM-DD 日期两种格式。
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
        # 若本地历史样本不足三个月（约90个交易日窗口），也强制刷新一次。
        sample_count = sum(1 for b in self.ingestion_store.history_bars if str(b.get("stock_code", "")).upper() == code)
        if sample_count < max(30, int(min_samples)):
            return True
        for b in reversed(self.ingestion_store.history_bars):
            if str(b.get("stock_code", "")).upper() != code:
                continue
            ts = self._parse_time(str(b.get("trade_date", "")))
            return (now - ts).total_seconds() > max_age_seconds
        return True

    def _latest_quote(self, stock_code: str) -> dict[str, Any] | None:
        code = stock_code.upper().replace(".", "")
        for q in reversed(self.ingestion_store.quotes):
            if str(q.get("stock_code", "")).upper() == code:
                return q
        return None

    def _history_bars(self, stock_code: str, limit: int = 120) -> list[dict[str, Any]]:
        code = stock_code.upper().replace(".", "")
        rows = [x for x in self.ingestion_store.history_bars if str(x.get("stock_code", "")).upper() == code]
        rows.sort(key=lambda x: str(x.get("trade_date", "")))
        return rows[-limit:]

    def _history_3m_summary(self, stock_code: str) -> dict[str, Any]:
        """提炼最近三个月（约90日）历史窗口，供模型与前端输出更稳健的连续样本描述。"""
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

    def _augment_question_with_history_context(self, question: str, stock_codes: list[str]) -> str:
        """在模型输入前注入三个月连续样本摘要，减少“离散样本误判”。"""
        extras: list[str] = []
        for code in stock_codes:
            summary = self._history_3m_summary(code)
            sample_count = int(summary.get("sample_count", 0) or 0)
            if sample_count < 30:
                continue
            extras.append(
                f"{code}: 最近三个月连续日线样本 {sample_count} 条，"
                f"区间 {summary.get('start_date','')} -> {summary.get('end_date','')}，"
                f"收盘 {float(summary.get('start_close', 0.0)):.3f} -> {float(summary.get('end_close', 0.0)):.3f}，"
                f"区间涨跌 {float(summary.get('pct_change', 0.0)) * 100:.2f}%。"
            )
        if not extras:
            return question
        return (
            f"{question}\n"
            "【系统补充：连续样本上下文】\n"
            "以下为历史连续样本摘要，请优先基于该窗口判断，避免把结论建立在离散点上：\n"
            + "\n".join(f"- {line}" for line in extras)
        )

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
        targets = ",".join(stock_codes) if stock_codes else "未指定"
        return [
            f"- 用户目标: 围绕 `{targets}` 回答 `{question}`，输出可验证的数据结论。",
            "- 产品判断: 需要同时展示实时快照与历史趋势，避免仅凭单点行情下结论。",
            "- 信息缺口: 若关键财报指标缺失，需明确提示并建议补充文档/财报源。",
        ]

    def _dev_manager_view(self, stock_codes: list[str]) -> list[str]:
        targets = ",".join(stock_codes) if stock_codes else "未指定"
        return [
            f"- 本轮实现: 已接入历史K线免费源并用于 `{targets}` 趋势计算。",
            "- 工程计划: 下一步补充财报结构化字段（营收/利润/现金流）与异常检测。",
            "- 质量门禁: 每次发布前验证数据新鲜度、引用覆盖率、趋势指标完整性。",
        ]

    def report_generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        """基于问答结果生成报告并缓存。"""
        req = ReportRequest(**payload)
        run_id = str(payload.get("run_id", "")).strip()
        pool_snapshot_id = str(payload.get("pool_snapshot_id", "")).strip()
        template_id = str(payload.get("template_id", "default")).strip() or "default"
        query_result = self.query(
            {
                "user_id": req.user_id,
                "question": f"请生成{req.stock_code} {req.period} 的{req.report_type}报告",
                "stock_codes": [req.stock_code],
            }
        )
        report_id = str(uuid.uuid4())
        evidence_refs = [
            {
                "source_id": str(c.get("source_id", "")),
                "source_url": str(c.get("source_url", "")),
                "reliability_score": float(c.get("reliability_score", 0.0)),
                "excerpt": str(c.get("excerpt", ""))[:240],
            }
            for c in query_result["citations"]
        ]
        report_sections = [
            {"section_id": "summary", "title": "结论摘要", "content": str(query_result["answer"])[:800]},
            {
                "section_id": "evidence",
                "title": "证据清单",
                "content": "\n".join(f"- {x['source_id']}: {x['excerpt']}" for x in evidence_refs[:8]),
            },
            {"section_id": "risk", "title": "风险与反证", "content": "需结合估值、流动性、政策扰动进行反证校验。"},
            {"section_id": "action", "title": "操作建议", "content": "建议分批验证信号稳定性，避免一次性重仓。"},
        ]
        markdown = (
            f"# {req.stock_code} 分析报告\n\n"
            f"## 结论\n{query_result['answer']}\n\n"
            f"## 证据\n"
            + "\n".join(f"- {c['source_id']}: {c['excerpt']}" for c in query_result["citations"])
            + "\n\n## 风险与反证\n需结合估值、流动性、政策扰动进行反证校验。"
            + "\n\n## 操作建议\n建议分批验证信号稳定性，避免一次性重仓。"
        )
        self._reports[report_id] = {
            "report_id": report_id,
            "trace_id": query_result["trace_id"],
            "markdown": markdown,
            "citations": query_result["citations"],
            "run_id": run_id,
            "pool_snapshot_id": pool_snapshot_id,
            "template_id": template_id,
            "evidence_refs": evidence_refs,
            "report_sections": report_sections,
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
        return resp

    def report_get(self, report_id: str) -> dict[str, Any]:
        """按 ID 查询报告。"""
        report = self._reports.get(report_id)
        if not report:
            return {"error": "not_found", "report_id": report_id}
        return report

    def ingest_market_daily(self, stock_codes: list[str]) -> dict[str, Any]:
        """触发行情日级数据摄取。"""
        return self.ingestion.ingest_market_daily(stock_codes)

    def ingest_announcements(self, stock_codes: list[str]) -> dict[str, Any]:
        """触发公告数据摄取。"""
        return self.ingestion.ingest_announcements(stock_codes)

    def docs_upload(self, doc_id: str, filename: str, content: str, source: str) -> dict[str, Any]:
        """上传文档原文。"""
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
        """执行文档分块索引。"""
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
            # 文档索引完成后，把 chunk 持久化到 RAG 资产库，供后续检索与治理复用。
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
        """统一处理 RAG 附件上传：解码 -> 去重 -> docs_upload -> (可选)index -> 资产入库。"""
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
            # 用 hash 前缀生成稳定 doc_id，便于后续跨入口追踪同一文件。
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

    def rag_workflow_upload_and_index(self, token: str, payload: dict[str, Any]) -> dict[str, Any]:
        """业务入口：上传并立即索引，返回阶段时间线，便于前端给出可解释反馈。"""
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
        return {"status": "ok", "result": result, "timeline": timeline}

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
        """从附件字节提取文本内容；无第三方解析器时执行可回退的轻量策略。"""
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
            # 无 pdf 解析库时的兜底：尝试抽取可见 ASCII 串，至少保留部分可检索内容。
            ascii_chunks = re.findall(rb"[A-Za-z0-9][A-Za-z0-9 ,.;:%()\-_/]{16,}", raw_bytes)
            decoded = " ".join(x.decode("latin1", errors="ignore") for x in ascii_chunks)
            note_parts.append("pdf_ascii_fallback")
            return decoded, ",".join(note_parts)

        note_parts.append(f"generic_decode:{content_type or 'unknown'}")
        return self._decode_text_bytes(raw_bytes), ",".join(note_parts)

    def _persist_doc_chunks_to_rag(self, doc_id: str, doc: dict[str, Any]) -> None:
        """把内存态文档 chunk 同步到 Web 持久层。"""
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
                    # 双轨中的“摘要/脱敏轨”：在线检索默认使用 redacted 文本。
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
        """从文本中提取 SH/SZ 股票代码，用于检索时的标的过滤。"""
        items = re.findall(r"\b(?:SH|SZ)\d{6}\b", str(text or "").upper())
        return list(dict.fromkeys(items))

    @staticmethod
    def _redact_text(text: str) -> str:
        """轻量脱敏：去除邮箱/手机号/连续证件号，降低共享语料风险。"""
        value = str(text or "")
        value = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]", value)
        value = re.sub(r"\b1\d{10}\b", "[REDACTED_PHONE]", value)
        value = re.sub(r"\b\d{15,18}[0-9Xx]?\b", "[REDACTED_ID]", value)
        return value

    def evals_run(self, samples: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """运行评测并返回门禁结果。"""
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
        """查询评测任务状态。"""
        return {"eval_run_id": eval_run_id, "status": "not_persisted_in_mvp"}

    def scheduler_run(self, job_name: str) -> dict[str, Any]:
        """手动触发一次调度任务运行。"""
        return self.scheduler.run_once(job_name)

    def scheduler_status(self) -> dict[str, Any]:
        """查询所有调度任务状态。"""
        return self.scheduler.list_status()

    # ---------- Prediction domain methods ----------
    def predict_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """执行量化预测任务。"""
        stock_codes = payload.get("stock_codes", [])
        pool_id = str(payload.get("pool_id", "")).strip()
        token = str(payload.get("token", "")).strip()
        if pool_id:
            stock_codes = self.web.watchlist_pool_codes(token, pool_id)
        horizons = payload.get("horizons") or ["5d", "20d"]
        as_of_date = payload.get("as_of_date")
        result = self.prediction.run_prediction(stock_codes=stock_codes, horizons=horizons, as_of_date=as_of_date)
        if pool_id:
            result["pool_id"] = pool_id
        result["segment_metrics"] = self._predict_segment_metrics(result.get("results", []))
        return result

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
        """查询预测任务详情。"""
        return self.prediction.get_prediction(run_id)

    def factor_snapshot(self, stock_code: str) -> dict[str, Any]:
        """查询单股票因子快照。"""
        return self.prediction.get_factor_snapshot(stock_code)

    def predict_eval_latest(self) -> dict[str, Any]:
        """查询最近一次预测评测摘要。"""
        return self.prediction.eval_latest()

    def market_overview(self, stock_code: str) -> dict[str, Any]:
        """返回单标的结构化行情总览（实时+历史+公告+趋势）。"""
        code = stock_code.upper().replace(".", "")
        if self._needs_quote_refresh(code):
            self.ingest_market_daily([code])
        if self._needs_announcement_refresh(code):
            self.ingest_announcements([code])
        if self._needs_history_refresh(code):
            self.ingestion.ingest_history_daily([code], limit=260)

        realtime = self._latest_quote(code)
        bars = self._history_bars(code, limit=120)
        events = [x for x in self.ingestion_store.announcements if str(x.get("stock_code", "")).upper() == code][-10:]
        trend = self._trend_metrics(bars) if bars else {}
        return {
            "stock_code": code,
            "realtime": realtime or {},
            "history": bars,
            "events": events,
            "trend": trend,
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

    def reports_list(self, token: str) -> list[dict[str, Any]]:
        return self.web.report_list(token)

    def report_versions(self, token: str, report_id: str) -> list[dict[str, Any]]:
        return self.web.report_versions(token, report_id)

    def report_export(self, token: str, report_id: str) -> dict[str, Any]:
        return self.web.report_export(token, report_id)

    def docs_list(self, token: str) -> list[dict[str, Any]]:
        return self.web.docs_list(token)

    def docs_review_queue(self, token: str) -> list[dict[str, Any]]:
        return self.web.docs_review_queue(token)

    def docs_review_action(self, token: str, doc_id: str, action: str, comment: str = "") -> dict[str, Any]:
        result = self.web.docs_review_action(token, doc_id, action, comment)
        # 审核动作需要同步到 chunk 生效状态，避免“文档状态已改但检索仍命中旧片段”。
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
        """会话级互斥：同一 session 同时仅允许一个 round 在执行。"""
        with self._deep_round_mutex:
            if session_id in self._deep_round_inflight:
                return False
            self._deep_round_inflight.add(session_id)
            return True

    def _deep_round_release(self, session_id: str) -> None:
        """释放会话执行锁，确保异常路径也不会永久占用。"""
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
        """统一补齐 DeepThink 流事件的元字段，避免前后端协议分叉。"""
        payload = dict(data)
        payload.setdefault("session_id", session_id)
        payload.setdefault("round_id", round_id)
        payload.setdefault("round_no", round_no)
        payload.setdefault("event_seq", event_seq)
        payload.setdefault("emitted_at", datetime.now(timezone.utc).isoformat())
        return {"event": event_name, "data": payload}

    def deep_think_run_round_stream_events(self, session_id: str, payload: dict[str, Any] | None = None):
        """V2 真流式执行：执行过程中逐步产出事件，并在结束时返回会话快照。"""
        payload = payload or {}
        started_at = time.perf_counter()
        session = self.web.deep_think_get_session(session_id)
        if not session:
            # 统一以 done 收口，便于前端和测试代码复用同一结束逻辑。
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
        archive_policy = self._resolve_deep_archive_retention_policy(
            session=session,
            requested_max_events=payload.get("archive_max_events"),
        )
        archive_max_events = int(archive_policy["max_events"])

        event_seq = 0
        first_event_ms: int | None = None
        stream_events: list[dict[str, Any]] = []

        def emit(event_name: str, data: dict[str, Any], *, persist: bool = True) -> dict[str, Any]:
            # 所有事件统一走此入口，保证 event_seq 与元字段完整。
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
                "counter_view": "无显著反方观点",
            }

            # 先发 round_started，确保前端在计算前就能拿到“已开始执行”的反馈。
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
                    "message": "任务拆解完成，开始执行多 Agent 协商。",
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
                        reason="预算已触顶，停止继续推理并输出保守结论。",
                        evidence_ids=[],
                        risk_tags=["budget_exceeded"],
                    )
                )
                arbitration = self._arbitrate_opinions(opinions)
            else:
                yield emit("progress", {"stage": "data_refresh", "message": "刷新行情与历史样本。"})
                if self._needs_quote_refresh(code):
                    self.ingest_market_daily([code])
                if self._needs_history_refresh(code):
                    self.ingestion.ingest_history_daily([code], limit=260)

                quote = self._latest_quote(code) or {}
                bars = self._history_bars(code, limit=180)
                trend = self._trend_metrics(bars) if len(bars) >= 30 else {}
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
                # 引入实时情报阶段：由模型侧 websearch 能力输出结构化情报。
                yield emit("progress", {"stage": "intel_search", "message": "正在检索宏观/行业/未来事件情报。"})
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
                    },
                )
                # 单独下发状态事件，便于前端或测试脚本快速判断是否命中外部实时检索。
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

                yield emit("progress", {"stage": "debate", "message": "生成多 Agent 观点。"})
                rule_core = self._build_rule_based_debate_opinions(question, trend, quote, quant_20)
                core_opinions = rule_core
                if self.settings.llm_external_enabled and self.llm_gateway.providers:
                    llm_core = self._build_llm_debate_opinions(question, code, trend, quote, quant_20, rule_core)
                    if llm_core:
                        core_opinions = llm_core

                # 将实时情报引用 URL 纳入证据链，后续导出与解释可追溯。
                intel_urls = [str(x.get("url", "")).strip() for x in list(intel_payload.get("citations", [])) if isinstance(x, dict)]
                evidence_ids = [x for x in {str(quote.get("source_id", "")), str((bars[-1] if bars else {}).get("source_id", "")), *intel_urls} if x]
                extra_opinions = [
                    self._normalize_deep_opinion(
                        agent="macro_agent",
                        signal="buy" if trend.get("ma60_slope", 0.0) > 0 and trend.get("momentum_20", 0.0) > 0 else "hold",
                        confidence=0.63,
                        reason=f"宏观侧评估：ma60_slope={trend.get('ma60_slope', 0.0):.4f}, momentum_20={trend.get('momentum_20', 0.0):.4f}",
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
                            f"执行层建议：pct_change={float(quote.get('pct_change', 0.0)):.2f}, "
                            f"volatility_20={trend.get('volatility_20', 0.0):.4f}"
                        ),
                        evidence_ids=evidence_ids,
                        risk_tags=["execution_timing"],
                    ),
                    self._normalize_deep_opinion(
                        agent="compliance_agent",
                        signal="reduce" if trend.get("max_drawdown_60", 0.0) > 0.2 else "hold",
                        confidence=0.78,
                        reason=f"合规与风控底线：max_drawdown_60={trend.get('max_drawdown_60', 0.0):.4f}",
                        evidence_ids=evidence_ids,
                        risk_tags=["compliance_guardrail"],
                    ),
                    self._normalize_deep_opinion(
                        agent="critic_agent",
                        signal="hold",
                        confidence=0.7,
                        reason="质检视角：已检查证据数量、信号冲突与风险标签完整性。",
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
                opinions = normalized_core + extra_opinions
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
                                reason=f"实时情报融合：{str(intel_decision.get('rationale', '未提供详细解释'))}",
                                evidence_ids=evidence_ids,
                                risk_tags=["intel_websearch"],
                            )
                        )

                pre_arb = self._arbitrate_opinions(opinions)
                if float(pre_arb["disagreement_score"]) >= 0.45 and round_no < max_rounds:
                    replan_triggered = True
                    task_graph.append(
                        {
                            "task_id": f"r{round_no}-replan-1",
                            "agent": "critic_agent",
                            "title": "触发补证重规划：查找冲突证据并解释分歧",
                            "priority": "high",
                        }
                    )
                    pre_arb["conflict_sources"] = list(dict.fromkeys(list(pre_arb["conflict_sources"]) + ["replan_triggered"]))

                supervisor = self._normalize_deep_opinion(
                    agent="supervisor_agent",
                    signal=str(pre_arb["consensus_signal"]),
                    confidence=round(1.0 - float(pre_arb["disagreement_score"]), 4),
                    reason=f"监督者仲裁：冲突源={','.join(pre_arb['conflict_sources']) or 'none'}",
                    evidence_ids=evidence_ids,
                    risk_tags=["supervisor_arbitration"],
                )
                opinions.append(supervisor)
                arbitration = self._arbitrate_opinions(opinions)
                if budget_usage["warn"]:
                    arbitration["conflict_sources"] = list(dict.fromkeys(list(arbitration["conflict_sources"]) + ["budget_warning"]))

            # 逐条输出 Agent 增量与最终观点，保证前端能持续刷新。
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
            # 业务摘要通过会话快照透传给同步接口调用方，便于前端直接展示。
            snapshot["business_summary"] = business_summary

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

            # 统一按生成顺序落库，确保重放事件与实时事件顺序一致。
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
            # 失败场景也发 error + done，前端可统一处理结束态。
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
        # V1 兼容路径：复用同一执行核心，但消费掉流事件，仅返回最终快照。
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
        # 为 Windows Excel 兼容写入 UTF-8 BOM，避免中文列值乱码。
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
        """导出业务可读摘要（面向投研/交易），与审计事件导出分离。"""
        export_format = str(format or "csv").strip().lower() or "csv"
        if export_format not in {"csv", "json"}:
            raise ValueError("format must be one of: csv, json")
        session = self.web.deep_think_get_session(session_id)
        if not session:
            return {"error": "not_found", "session_id": session_id}
        safe_limit = max(20, min(2000, int(limit)))
        round_id_clean = str(round_id or "").strip()
        rows: list[dict[str, Any]] = []
        # 优先使用业务摘要事件，确保导出的是对用户有意义的决策结果。
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
        # 若历史轮次无 business_summary 事件，兜底使用会话 round 快照生成可读摘要。
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
                        "trigger_condition": str(top_opinion.get("reason", "建议结合更多实时情报复核。")),
                        "invalidation_condition": "若风险因子放大或关键信号反转，则本结论失效。",
                        "review_time_hint": "T+1 日内复核",
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
            # 同样加 BOM，确保业务用户在 Excel 中直接可读。
            "content": "\ufeff" + output.getvalue(),
            "count": len(rows),
        }

    def deep_think_intel_self_test(self, *, stock_code: str, question: str = "") -> dict[str, Any]:
        """DeepThink 情报链路自检：验证外部开关/provider/websearch 与 fallback 原因。"""
        code = (stock_code or "SH600000").upper().replace(".", "")
        probe_question = (question or f"请做 {code} 的未来30日宏观+行业+事件情报自检").strip()
        provider_rows = [
            {
                "name": str(p.name),
                "enabled": bool(p.enabled),
                "api_style": str(p.api_style),
                "model": str(p.model),
            }
            for p in self.llm_gateway.providers
        ]
        # 尽量使用接近真实链路的数据快照，避免“自检通过但实战失败”的偏差。
        if self._needs_quote_refresh(code):
            self.ingest_market_daily([code])
        if self._needs_history_refresh(code):
            self.ingestion.ingest_history_daily([code], limit=260)
        quote = self._latest_quote(code) or {}
        bars = self._history_bars(code, limit=180)
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
        """输出 trace 事件明细，便于排查外部检索降级原因。"""
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

    def ops_deep_think_archive_metrics(self, token: str, *, window_hours: int = 24) -> dict[str, Any]:
        self.web.require_role(token, {"admin", "ops"})
        return self.web.deep_think_archive_audit_metrics(window_hours=window_hours)

    def ops_source_health(self, token: str) -> list[dict[str, Any]]:
        # 基于 scheduler 状态刷新 source health 快照
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
        """多 Agent 分歧分析：优先使用真实大模型并行辩论，失败回退规则引擎。"""
        code = stock_code.upper().replace(".", "")
        if self._needs_quote_refresh(code):
            self.ingest_market_daily([code])
        if self._needs_history_refresh(code):
            self.ingestion.ingest_history_daily([code], limit=260)

        quote = self._latest_quote(code) or {}
        bars = self._history_bars(code, limit=160)
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
            "market_snapshot": {
                "price": quote.get("price"),
                "pct_change": quote.get("pct_change"),
                "trend": trend,
            },
        }

    def ops_rag_quality(self) -> dict[str, Any]:
        """RAG 质量面板数据：聚合指标 + 用例明细。"""
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
        """向量索引重建入口：重建摘要索引并回传统计信息。"""
        self.web.require_role(token, {"admin", "ops"})
        del limit  # 当前实现按实时资产全集重建，后续可扩展增量重建窗口。
        result = self._refresh_summary_vector_index([], force=True)
        # 记录最近一次重建时间，便于前端业务看板展示运维状态。
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
        """Prompt 版本对比回放。"""
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
                "reason": f"基于趋势斜率与当日涨跌构建产品侧可解释观点，question={question or 'default'}",
            },
            {
                "agent": "quant_agent",
                "signal": quant_signal,
                "confidence": float(quant_20.get("up_probability", 0.5)),
                "reason": f"来自20日预测：{quant_20.get('rationale', '')}",
            },
            {
                "agent": "risk_agent",
                "signal": risk_signal,
                "confidence": 0.72,
                "reason": (
                    f"回撤={trend.get('max_drawdown_60', 0.0):.4f}, 波动={trend.get('volatility_20', 0.0):.4f}, "
                    f"动量={trend.get('momentum_20', 0.0):.4f}"
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
        """并行调用真实模型生成多角色观点。"""
        context = {
            "question": question or "请给出短中期观点",
            "stock_code": stock_code,
            "quote": {"price": quote.get("price"), "pct_change": quote.get("pct_change")},
            "trend": trend,
            "quant_20": quant_20,
        }
        prompts = [
            (
                "pm_agent",
                (
                    "你是资深产品经理。请根据输入给出一个交易信号。"
                    "严格返回 JSON: {\"signal\":\"buy|hold|reduce\",\"confidence\":0-1,\"reason\":\"...\"}\n"
                    f"context={json.dumps(context, ensure_ascii=False)}"
                ),
            ),
            (
                "quant_agent",
                (
                    "你是量化研究负责人。请根据输入给出一个交易信号。"
                    "严格返回 JSON: {\"signal\":\"buy|hold|reduce\",\"confidence\":0-1,\"reason\":\"...\"}\n"
                    f"context={json.dumps(context, ensure_ascii=False)}"
                ),
            ),
            (
                "risk_agent",
                (
                    "你是风控负责人。请根据输入给出一个交易信号。"
                    "严格返回 JSON: {\"signal\":\"buy|hold|reduce\",\"confidence\":0-1,\"reason\":\"...\"}\n"
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
        """释放数据库连接等资源。"""
        self.memory.close()
        self.prompts.close()
        self.web.close()




