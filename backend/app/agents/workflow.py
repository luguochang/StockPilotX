from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Iterator

from backend.app.agents.tools import (
    AnnouncementToolInput,
    GraphToolInput,
    LangChainToolRunner,
    QuoteToolInput,
    RetrieveToolInput,
    ToolAccessController,
)
from backend.app.middleware.hooks import MiddlewareStack
from backend.app.rag.graphrag import GraphRAGService
from backend.app.rag.retriever import HybridRetriever, RetrievalItem
from backend.app.state import AgentState


def route_intent(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ("对比", "比较")):
        return "compare"
    if any(k in q for k in ("文档", "pdf", "报告")):
        return "doc_qa"
    if any(k in q for k in ("深入", "深度", "归因", "风险")):
        return "deep"
    return "fact"


class AgentWorkflow:
    """多 Agent 工作流（含 Deep Agents 与工具 ACL）。"""

    def __init__(
        self,
        retriever: HybridRetriever,
        graph_rag: GraphRAGService,
        middleware_stack: MiddlewareStack,
        trace_emit: callable,
        tool_acl: ToolAccessController | None = None,
        prompt_renderer: callable | None = None,
        external_model_call: callable | None = None,
        external_model_stream_call: callable | None = None,
        enable_local_fallback: bool = True,
    ) -> None:
        self.retriever = retriever
        self.graph_rag = graph_rag
        self.middleware = middleware_stack
        self.trace_emit = trace_emit
        self.tool_acl = tool_acl or ToolAccessController()
        self.tool_runner = LangChainToolRunner(self.tool_acl)
        self.prompt_renderer = prompt_renderer
        self.external_model_call = external_model_call
        self.external_model_stream_call = external_model_stream_call
        self.enable_local_fallback = enable_local_fallback
        self._register_default_tools()

    def run(self, state: AgentState, memory_hint: list[dict[str, Any]] | None = None) -> AgentState:
        prompt = self.prepare_prompt(state, memory_hint=memory_hint)
        prompt = self.apply_before_model(state, prompt)
        output = self.invoke_model(state, prompt)
        output = self.apply_after_model(state, output)
        return self.finalize_with_output(state, output)

    def run_stream(self, state: AgentState, memory_hint: list[dict[str, Any]] | None = None) -> Iterator[dict[str, Any]]:
        """流式执行：优先透传上游LLM增量。"""
        prompt = self.prepare_prompt(state, memory_hint=memory_hint)
        prompt = self.apply_before_model(state, prompt)
        yield {"event": "meta", "data": {"trace_id": state.trace_id, "intent": state.intent, "mode": state.mode}}

        output, stream_events = self.stream_model_collect(state, prompt)
        for event in stream_events:
            yield event

        output = self.apply_after_model(state, output)
        self.finalize_with_output(state, output)
        yield {"event": "citations", "data": {"citations": state.citations}}
        yield {"event": "done", "data": {"ok": True, "trace_id": state.trace_id}}

    # ---------------- Public stage methods ----------------
    def prepare_prompt(self, state: AgentState, memory_hint: list[dict[str, Any]] | None = None) -> str:
        """阶段1：准备状态与证据并构造 prompt。"""
        return self._prepare_state(state, memory_hint=memory_hint)

    def apply_before_model(self, state: AgentState, prompt: str) -> str:
        """阶段2：执行模型前中间件。"""
        return self.middleware.run_before_model(state, prompt)

    def invoke_model(self, state: AgentState, prompt: str) -> str:
        """阶段3：执行模型调用（含回退链）。"""
        return self.middleware.call_model(state, prompt, self._model_call_with_fallback)

    def apply_after_model(self, state: AgentState, output: str) -> str:
        """阶段4：执行模型后中间件。"""
        return self.middleware.run_after_model(state, output)

    def finalize_with_output(self, state: AgentState, output: str) -> AgentState:
        """阶段5：写入报告并执行引用与收尾。"""
        state.report = output
        return self._finalize_state(state)

    def stream_model_collect(self, state: AgentState, prompt: str, chunk_size: int = 80) -> tuple[str, list[dict[str, Any]]]:
        """阶段3(流式)：收集模型增量并返回完整输出及事件列表。"""
        events: list[dict[str, Any]] = []
        # 先触发预算中间件模型调用计数与阈值检查。
        self.middleware.call_model(state, prompt, lambda s, p: "")
        output = ""
        try:
            if self.external_model_stream_call is None:
                raise RuntimeError("external stream model is not configured")
            chunks: list[str] = []
            for delta in self.external_model_stream_call(state, prompt):
                if not delta:
                    continue
                chunks.append(delta)
                events.append({"event": "answer_delta", "data": {"delta": delta}})
            output = "".join(chunks)
            if not output.strip():
                raise RuntimeError("external stream returned empty output")
            events.append(
                {
                    "event": "stream_source",
                    "data": {
                        "source": "external_llm_stream",
                        "provider": state.analysis.get("llm_provider", ""),
                        "model": state.analysis.get("llm_model", ""),
                        "api_style": state.analysis.get("llm_api_style", ""),
                    },
                }
            )
        except Exception as ex:  # noqa: BLE001
            self.trace_emit(state.trace_id, "external_model_failed", {"error": str(ex)})
            state.risk_flags.append("external_model_failed")
            if not self.enable_local_fallback:
                raise
            output = self._synthesize_model_output(state, prompt)
            events.append(
                {
                    "event": "stream_source",
                    "data": {
                        "source": "local_fallback_stream",
                        "provider": "local_fallback",
                        "model": "rule_based_local",
                        "api_style": "local",
                    },
                }
            )
            for idx in range(0, len(output), chunk_size):
                events.append({"event": "answer_delta", "data": {"delta": output[idx : idx + chunk_size]}})
        return output, events

    def _prepare_state(self, state: AgentState, memory_hint: list[dict[str, Any]] | None = None) -> str:
        self.middleware.run_before_agent(state)
        self.trace_emit(state.trace_id, "before_agent", {"question": state.question})
        state.intent = route_intent(state.question)
        self.trace_emit(state.trace_id, "router", {"intent": state.intent})
        state.retrieval_plan = {
            "top_k_vector": 12,
            "top_k_bm25": 20,
            "rerank_top_n": 10,
            "memory_hint_count": len(memory_hint or []),
        }
        _ = self._call_tool("data", "quote_tool", {"stock_codes": state.stock_codes or ["SH600000"]})
        self.trace_emit(state.trace_id, "data_plan", state.retrieval_plan)

        question = state.question
        if state.stock_codes:
            question = f"{question} {' '.join(state.stock_codes)}"
        if memory_hint:
            question = f"{question}\n历史记忆摘要:{memory_hint[0]['content']}"

        if self._should_use_graphrag(state):
            state.mode = "graph_rag"
            graph_payload = self.graph_rag.query_subgraph(question, state.stock_codes)
            state.evidence_pack = [
                {
                    "text": graph_payload["summary"],
                    "source_id": "graph",
                    "source_url": "neo4j://local",
                    "event_time": datetime.now(timezone.utc).isoformat(),
                    "reliability_score": 0.8,
                }
            ]
            for rel in graph_payload.get("relations", [])[:10]:
                state.evidence_pack.append(
                    {
                        "text": f"{rel['from']} -> {rel['to']} ({rel['type']})",
                        "source_id": rel.get("source_id", "graph"),
                        "source_url": rel.get("source_url", "neo4j://local"),
                        "event_time": datetime.now(timezone.utc).isoformat(),
                        "reliability_score": 0.85,
                    }
                )
        else:
            state.mode = "agentic_rag"
            if self._should_use_deep_agents(state):
                items = self._deep_retrieve(question, state)
            else:
                items = self.retriever.retrieve(
                    question,
                    top_k_vector=state.retrieval_plan["top_k_vector"],
                    top_k_bm25=state.retrieval_plan["top_k_bm25"],
                    rerank_top_n=state.retrieval_plan["rerank_top_n"],
                )
            state.evidence_pack = [
                {
                    "text": i.text,
                    "source_id": i.source_id,
                    "source_url": i.source_url,
                    "event_time": i.event_time.isoformat(),
                    "reliability_score": i.reliability_score,
                }
                for i in items
            ]
        self.trace_emit(state.trace_id, "retrieval", {"mode": state.mode, "evidence": len(state.evidence_pack)})
        state.analysis = self._analyze(state)
        self.trace_emit(state.trace_id, "analysis", state.analysis)
        return self._build_prompt(state)

    def _finalize_state(self, state: AgentState) -> AgentState:
        state.citations = self._build_citations(state)
        if not state.citations:
            state.risk_flags.append("missing_citation")
        self.trace_emit(
            state.trace_id,
            "critic",
            {"citation_count": len(state.citations), "risk_flags": state.risk_flags},
        )
        self.middleware.run_after_agent(state)
        self.trace_emit(state.trace_id, "after_agent", {"middleware_logs": self.middleware.ctx.logs})
        return state

    def _should_use_graphrag(self, state: AgentState) -> bool:
        return any(k in state.question for k in ("关系", "演化", "关联", "产业链", "股权"))

    def _should_use_deep_agents(self, state: AgentState) -> bool:
        return state.intent in ("deep", "compare") or any(k in state.question for k in ("多维", "并行", "长期", "对比"))

    def _plan_subtasks(self, question: str) -> list[str]:
        return [f"{question}：财务维度", f"{question}：行业维度", f"{question}：风险维度"]

    def _deep_retrieve(self, question: str, state: AgentState) -> list[RetrievalItem]:
        subtasks = self._plan_subtasks(question)
        state.analysis["deep_subtasks"] = subtasks
        merged: list[RetrievalItem] = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(self.retriever.retrieve, q, 8, 12, 5): q for q in subtasks}
            for fut in as_completed(futures):
                merged.extend(fut.result())
        uniq: dict[tuple[str, str], RetrievalItem] = {}
        for item in merged:
            uniq[(item.source_id, item.text)] = item
        items = list(uniq.values())
        items.sort(key=lambda x: x.score, reverse=True)
        self.trace_emit(state.trace_id, "deep_agents", {"subtask_count": len(subtasks), "merged_items": len(items)})
        return items[: state.retrieval_plan["rerank_top_n"]]

    def _analyze(self, state: AgentState) -> dict[str, Any]:
        positives = [e for e in state.evidence_pack if e.get("reliability_score", 0) >= 0.9]
        risks = [e for e in state.evidence_pack if e.get("reliability_score", 0) < 0.75]
        return {
            "fact_count": len(state.evidence_pack),
            "high_confidence_count": len(positives),
            "low_confidence_count": len(risks),
            "summary": "证据显示公司存在业绩改善线索，但仍需关注行业波动风险。",
            "deep_subtasks": state.analysis.get("deep_subtasks", []),
        }

    def _build_prompt(self, state: AgentState) -> str:
        evidence = "\n".join(
            f"- [{e['source_id']}] {e['text']} (score={e['reliability_score']})" for e in state.evidence_pack[:6]
        )
        if self.prompt_renderer:
            rendered, prompt_meta = self.prompt_renderer(
                {
                    "question": state.question,
                    "stock_codes": state.stock_codes,
                    "evidence": evidence,
                }
            )
            self.trace_emit(state.trace_id, "prompt_meta", prompt_meta)
            return rendered
        return (
            "你是A股分析助手。\n"
            f"问题：{state.question}\n"
            f"模式：{state.mode}\n"
            f"分析摘要：{state.analysis.get('summary', '')}\n"
            f"证据：\n{evidence}\n"
            "请输出简明结论并附免责声明。"
        )

    def _synthesize_model_output(self, state: AgentState, prompt: str) -> str:
        _ = prompt
        symbols = ",".join(state.stock_codes) if state.stock_codes else "该标的"
        high = state.analysis.get("high_confidence_count", 0)
        low = state.analysis.get("low_confidence_count", 0)
        fact_count = state.analysis.get("fact_count", 0)
        top_facts = [e["text"] for e in state.evidence_pack[:2] if e.get("text")]
        top_fact_text = "；".join(top_facts) if top_facts else "暂无高质量事实片段"

        if high >= max(1, low):
            stance = "证据一致性较好，短期可跟踪结构性机会"
        elif low > high:
            stance = "证据分歧偏高，建议先观察并等待更多披露"
        else:
            stance = "当前证据有限，建议保持中性观察"

        risk_line = "；".join(state.risk_flags) if state.risk_flags else "未触发高风险风控词"
        return (
            f"结论：{symbols} 当前共采集 {fact_count} 条证据，{stance}。\n"
            f"关键事实：{top_fact_text}。\n"
            f"风险提示：{risk_line}。\n"
            "仅供研究参考，不构成投资建议。"
        )

    def _model_call_with_fallback(self, state: AgentState, prompt: str) -> str:
        """优先调用外部LLM，失败时回退本地模型。"""
        if self.external_model_call is None:
            return self._synthesize_model_output(state, prompt)
        try:
            text = self.external_model_call(state, prompt)
            if text and text.strip():
                return text
            raise RuntimeError("external llm returned empty text")
        except Exception as ex:  # noqa: BLE001
            self.trace_emit(state.trace_id, "external_model_failed", {"error": str(ex)})
            state.risk_flags.append("external_model_failed")
            if self.enable_local_fallback:
                return self._synthesize_model_output(state, prompt)
            raise

    def _build_citations(self, state: AgentState) -> list[dict[str, Any]]:
        citations: list[dict[str, Any]] = []
        for e in state.evidence_pack[:5]:
            citations.append(
                {
                    "source_id": e["source_id"],
                    "source_url": e["source_url"],
                    "event_time": e.get("event_time"),
                    "reliability_score": e.get("reliability_score", 0.5),
                    "excerpt": e["text"][:120],
                }
            )
        return citations

    def _register_default_tools(self) -> None:
        self._register_tool(
            "quote_tool",
            lambda payload: {"status": "ok", "queried": payload.get("stock_codes", [])},
            "query market quote",
            QuoteToolInput,
        )
        self._register_tool(
            "announcement_tool",
            lambda payload: {"status": "ok", "stock_codes": payload.get("stock_codes", [])},
            "query company announcements",
            AnnouncementToolInput,
        )
        self._register_tool(
            "retrieve_tool",
            lambda payload: {"status": "ok", "query": payload.get("query", "")},
            "retrieve rag docs",
            RetrieveToolInput,
        )
        self._register_tool(
            "graph_tool",
            lambda payload: {"status": "ok", "symbol": payload.get("symbol", "")},
            "query graph relations",
            GraphToolInput,
        )
        self._register_tool("analysis_tool", lambda payload: {"status": "ok"}, "analysis step")
        self._register_tool("format_tool", lambda payload: {"status": "ok"}, "format report")
        self._register_tool("citation_check_tool", lambda payload: {"status": "ok"}, "check citations")
        self._register_tool("intent_hint_tool", lambda payload: {"status": "ok"}, "intent classification hint")

    def _register_tool(self, name: str, fn: callable, description: str, args_schema: type | None = None) -> None:
        self.tool_acl.register(name, fn)
        self.tool_runner.register(name, fn, description, args_schema=args_schema)

    def _call_tool(self, role: str, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        def call_next(name: str, data: dict[str, Any]) -> dict[str, Any]:
            return self.tool_runner.call(role, name, data)

        return self.middleware.call_tool(tool_name, payload, call_next)
