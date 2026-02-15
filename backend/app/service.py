from __future__ import annotations

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

        resp = QueryResponse(
            trace_id=trace_id,
            intent=state.intent,  # type: ignore[arg-type]
            answer=state.report,
            citations=[Citation(**c) for c in state.citations],
            risk_flags=state.risk_flags,
            mode=state.mode,  # type: ignore[arg-type]
            workflow_runtime=runtime_result.runtime,
        )
        return resp.model_dump(mode="json")

    def query_stream_events(self, payload: dict[str, Any], chunk_size: int = 80):
        """以事件流形式输出问答结果，便于前端逐段渲染。"""
        _ = chunk_size
        req = QueryRequest(**payload)
        selected_runtime = self._select_runtime(str(payload.get("workflow_runtime", "")))
        # 与同步 query 保持一致：先做数据刷新与动态语料装载。
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
                pass
        self.workflow.retriever = HybridRetriever(corpus=self._build_runtime_corpus(req.stock_codes))
        yield {"event": "start", "data": {"status": "started"}}
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
        for event in selected_runtime.run_stream(state, memory_hint=memory_hint):
            yield event

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

    def ops_capabilities(self) -> dict[str, Any]:
        return build_capability_snapshot(
            self.settings,
            workflow_runtime=self.workflow_runtime.runtime_name,
            llm_external_enabled=self.settings.llm_external_enabled,
        )

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
