from __future__ import annotations

import csv
import difflib
import io
import json
import queue
import threading
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
from backend.app.evals.service import EvalService
from backend.app.llm.gateway import MultiProviderLLMGateway
from backend.app.memory.store import MemoryStore
from backend.app.middleware.hooks import BudgetMiddleware, GuardrailMiddleware, MiddlewareStack
from backend.app.models import Citation, QueryRequest, QueryResponse, ReportRequest, ReportResponse
from backend.app.observability.tracing import TraceStore
from backend.app.prompt.registry import PromptRegistry
from backend.app.prompt.runtime import PromptRuntime
from backend.app.predict.service import PredictionService, PredictionStore
from backend.app.rag.evaluation import RetrievalEvaluator, default_retrieval_dataset
from backend.app.rag.graphrag import GraphRAGService
from backend.app.rag.retriever import HybridRetriever, RetrievalItem
from backend.app.state import AgentState
from backend.app.web.service import WebAppService
from backend.app.web.store import WebStore


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

        self.ingestion_store = IngestionStore()
        self.ingestion = IngestionService(
            # DATA-001: 真实源优先，失败后自动进入 mock 兜底回退。
            quote_service=QuoteService.build_default(),
            announcement_service=AnnouncementService(),
            store=self.ingestion_store,
        )
        self.prediction = PredictionService(
            quote_service=self.ingestion.quote_service,
            traces=self.traces,
            store=PredictionStore(),
            history_service=self.ingestion.history_service,
        )
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

    def query(self, payload: dict[str, Any]) -> dict[str, Any]:
        """执行问答主链路并返回标准化响应。"""
        req = QueryRequest(**payload)
        selected_runtime = self._select_runtime(str(payload.get("workflow_runtime", "")))

        # 真实数据优先：每次查询先尝试刷新一次行情与公告，再进入检索。
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
                # 摄取失败时不阻断问答，交由检索层使用已有数据继续降级运行。
                pass

        # 基于最新摄取数据动态构建检索语料，避免固定示例语料导致输出雷同。
        self.workflow.retriever = HybridRetriever(corpus=self._build_runtime_corpus(req.stock_codes))

        trace_id = self.traces.new_trace()
        state = AgentState(
            user_id=req.user_id,
            question=req.question,
            stock_codes=req.stock_codes,
            trace_id=trace_id,
        )

        memory_hint = self.memory.list_memory(req.user_id, limit=3)
        runtime_result = selected_runtime.run(state, memory_hint=memory_hint)
        state = runtime_result.state
        state.analysis["workflow_runtime"] = runtime_result.runtime
        self.traces.emit(trace_id, "workflow_runtime", {"runtime": runtime_result.runtime})
        answer, merged_citations = self._build_evidence_rich_answer(req.question, req.stock_codes, state.report, state.citations)
        analysis_brief = self._build_analysis_brief(req.stock_codes, merged_citations)
        state.report = answer
        state.citations = merged_citations

        # 写入任务记忆，便于后续对话复用
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
        return resp.model_dump(mode="json")

    def query_stream_events(self, payload: dict[str, Any], chunk_size: int = 80):
        """以事件流形式输出问答结果，便于前端逐段渲染。"""
        _ = chunk_size
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
        self.workflow.retriever = HybridRetriever(corpus=self._build_runtime_corpus(req.stock_codes))
        trace_id = self.traces.new_trace()
        state = AgentState(
            user_id=req.user_id,
            question=req.question,
            stock_codes=req.stock_codes,
            trace_id=trace_id,
        )
        memory_hint = self.memory.list_memory(req.user_id, limit=3)
        runtime_name = selected_runtime.runtime_name
        self.traces.emit(trace_id, "workflow_runtime", {"runtime": runtime_name})
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
        yield {"event": "analysis_brief", "data": self._build_analysis_brief(req.stock_codes, citations)}

    def _build_evidence_rich_answer(
        self,
        question: str,
        stock_codes: list[str],
        base_answer: str,
        citations: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        """生成带数据支撑的增强回答（实时+历史+双Agent讨论）。"""
        lines: list[str] = []
        merged = list(citations)

        lines.append("## Conclusion Summary")
        lines.append("## 结论摘要")
        lines.append(base_answer)
        lines.append("")
        lines.append("## Data Snapshot and History Trend")
        lines.append("## 数据证据分析")
        for code in stock_codes:
            realtime = self._latest_quote(code)
            bars = self._history_bars(code, limit=120)
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
        lines.append("")
        lines.append("## Evidence References")
        lines.append("## 引用清单")
        for idx, c in enumerate(merged[:10], start=1):
            lines.append(
                f"- [{idx}] `{c.get('source_id','unknown')}` | {c.get('source_url','')} | "
                f"score={float(c.get('reliability_score',0.0)):.2f} | {c.get('excerpt','')}"
            )
        return "\n".join(lines), merged[:10]

    def _build_analysis_brief(self, stock_codes: list[str], citations: list[dict[str, Any]]) -> dict[str, Any]:
        """构造结构化证据摘要，供前端可视化展示。"""
        by_code: list[dict[str, Any]] = []
        now = datetime.now(timezone.utc)
        for code in stock_codes:
            realtime = self._latest_quote(code)
            bars = self._history_bars(code, limit=120)
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
        return {
            "confidence_level": confidence,
            "confidence_reason": f"citations={citation_coverage}, avg_reliability={avg_score}",
            "stocks": by_code,
            "citation_count": citation_coverage,
            "citation_avg_reliability": avg_score,
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
        for b in self.ingestion_store.history_bars[-2000:]:
            code = str(b.get("stock_code", ""))
            if symbols and code not in symbols:
                continue
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

    def _needs_history_refresh(self, stock_code: str, max_age_seconds: int = 60 * 60 * 8) -> bool:
        code = stock_code.upper().replace(".", "")
        now = datetime.now(timezone.utc)
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
        query_result = self.query(
            {
                "user_id": req.user_id,
                "question": f"请生成{req.stock_code} {req.period} 的{req.report_type}报告",
                "stock_codes": [req.stock_code],
            }
        )
        report_id = str(uuid.uuid4())
        markdown = (
            f"# {req.stock_code} 分析报告\n\n"
            f"## 结论\n{query_result['answer']}\n\n"
            f"## 证据\n"
            + "\n".join(f"- {c['source_id']}: {c['excerpt']}" for c in query_result["citations"])
        )
        self._reports[report_id] = {
            "report_id": report_id,
            "trace_id": query_result["trace_id"],
            "markdown": markdown,
            "citations": query_result["citations"],
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
        )
        resp = ReportResponse(
            report_id=report_id,
            trace_id=query_result["trace_id"],
            markdown=markdown,
            citations=[Citation(**c) for c in query_result["citations"]],
        )
        return resp.model_dump(mode="json")

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
            needs_review=bool(doc.get("parse_confidence", 0.0) < 0.7),
        )
        return result

    def docs_index(self, doc_id: str) -> dict[str, Any]:
        """执行文档分块索引。"""
        result = self.ingestion.index_doc(doc_id)
        doc = self.ingestion.store.docs.get(doc_id, {})
        if doc:
            self.web.doc_upsert(
                doc_id=doc_id,
                filename=doc.get("filename", ""),
                parse_confidence=float(doc.get("parse_confidence", 0.0)),
                needs_review=bool(float(doc.get("parse_confidence", 0.0)) < 0.7),
            )
        return result

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
        horizons = payload.get("horizons") or ["5d", "20d"]
        as_of_date = payload.get("as_of_date")
        return self.prediction.run_prediction(stock_codes=stock_codes, horizons=horizons, as_of_date=as_of_date)

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

    def dashboard_overview(self, token: str) -> dict[str, Any]:
        return self.web.dashboard_overview(token)

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
        return self.web.docs_review_action(token, doc_id, action, comment)

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
            task_graph = self._deep_plan_tasks(question, round_no)
            budget = session.get("budget", {}) if isinstance(session.get("budget", {}), dict) else {}
            budget_usage = self._deep_budget_snapshot(budget, round_no, len(task_graph))
            replan_triggered = False
            stop_reason = ""

            evidence_ids: list[str] = []
            opinions: list[dict[str, Any]] = []
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
                pred = self.predict_run({"stock_codes": [code], "horizons": ["20d"]})
                horizon_map = {h["horizon"]: h for h in pred["results"][0]["horizons"]} if pred.get("results") else {}
                quant_20 = horizon_map.get("20d", {})

                yield emit("progress", {"stage": "debate", "message": "生成多 Agent 观点。"})
                rule_core = self._build_rule_based_debate_opinions(question, trend, quote, quant_20)
                core_opinions = rule_core
                if self.settings.llm_external_enabled and self.llm_gateway.providers:
                    llm_core = self._build_llm_debate_opinions(question, code, trend, quote, quant_20, rule_core)
                    if llm_core:
                        core_opinions = llm_core

                evidence_ids = [x for x in {str(quote.get("source_id", "")), str((bars[-1] if bars else {}).get("source_id", ""))} if x]
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
        csv_content = output.getvalue()
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
        retriever = HybridRetriever(corpus=self._build_runtime_corpus([]))
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
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        return self.web.stock_universe_search(
            token,
            keyword=keyword,
            exchange=exchange,
            market_tier=market_tier,
            listing_board=listing_board,
            industry_l1=industry_l1,
            limit=limit,
        )

    def stock_universe_filters(self, token: str) -> dict[str, list[str]]:
        return self.web.stock_universe_filters(token)

    def close(self) -> None:
        """释放数据库连接等资源。"""
        self.memory.close()
        self.prompts.close()
        self.web.close()



