from __future__ import annotations

from concurrent.futures import TimeoutError as FuturesTimeoutError
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
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


@dataclass(slots=True)
class IntentRoutingResult:
    """Rule-route result with confidence and explainable keyword hits."""

    intent: str
    confidence: float
    matched: dict[str, list[str]]
    conflict: bool


def route_intent(question: str) -> str:
    """Backward-compatible helper used by legacy callers."""
    return route_intent_with_confidence(question).intent


def route_intent_with_confidence(question: str) -> IntentRoutingResult:
    """Route intent via keyword rules and expose confidence for runtime telemetry."""
    q = str(question or "").strip().lower()
    compare_keywords = (
        "\u5bf9\u6bd4",
        "\u6bd4\u8f83",
        "vs",
        "versus",
        "compare",
    )
    doc_keywords = (
        "\u6587\u6863",
        "\u6587\u4ef6",
        "pdf",
        "\u62a5\u544a",
        "doc",
        "docx",
    )
    deep_keywords = (
        "\u6df1\u5165",
        "\u6df1\u5ea6",
        "\u5f52\u56e0",
        "\u98ce\u9669",
        "deep",
        "analysis",
    )
    matched = {
        "compare": [k for k in compare_keywords if k in q],
        "doc_qa": [k for k in doc_keywords if k in q],
        "deep": [k for k in deep_keywords if k in q],
    }
    # Priority is explicit: compare > doc_qa > deep > fact.
    # Score weights keep compare fast path robust for "A vs B" phrasing.
    weighted = {
        "compare": len(matched["compare"]) * 1.0,
        "doc_qa": len(matched["doc_qa"]) * 0.92,
        "deep": len(matched["deep"]) * 0.88,
    }
    ranked = sorted(weighted.items(), key=lambda x: x[1], reverse=True)
    top_intent, top_score = ranked[0]
    second_score = ranked[1][1]
    if top_score <= 0:
        return IntentRoutingResult(intent="fact", confidence=0.58, matched=matched, conflict=False)

    conflict = second_score > 0 and abs(top_score - second_score) <= 0.2
    # Confidence grows with hit count + margin; clamp keeps trace values stable.
    margin = max(0.0, top_score - second_score)
    confidence = min(0.98, max(0.62, 0.64 + min(0.22, top_score * 0.08) + min(0.12, margin * 0.08)))
    return IntentRoutingResult(intent=top_intent, confidence=round(confidence, 4), matched=matched, conflict=conflict)


class AgentWorkflow:
    """Multi-agent workflow with Deep Agents and tool ACL."""

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
        """Streaming execution that forwards upstream LLM deltas."""
        prompt = self.prepare_prompt(state, memory_hint=memory_hint)
        prompt = self.apply_before_model(state, prompt)
        yield {"event": "meta", "data": {"trace_id": state.trace_id, "intent": state.intent, "mode": state.mode}}
        # Must emit while consuming; collecting all events first causes fake streaming in the UI.
        stream_iter = self.stream_model_iter(state, prompt)
        output = ""
        while True:
            try:
                event = next(stream_iter)
                yield event
            except StopIteration as stop:
                output = str(stop.value or "")
                break

        output = self.apply_after_model(state, output)
        self.finalize_with_output(state, output)
        yield {"event": "citations", "data": {"citations": state.citations}}
        yield {"event": "done", "data": {"ok": True, "trace_id": state.trace_id}}

    # ---------------- Public stage methods ----------------
    def prepare_prompt(self, state: AgentState, memory_hint: list[dict[str, Any]] | None = None) -> str:
        """Stage 1: prepare state/evidence and construct the prompt."""
        return self._prepare_state(state, memory_hint=memory_hint)

    def apply_before_model(self, state: AgentState, prompt: str) -> str:
        """Stage 2: run before-model middleware."""
        return self.middleware.run_before_model(state, prompt)

    def invoke_model(self, state: AgentState, prompt: str) -> str:
        """Stage 3: invoke model with fallback chain."""
        return self.middleware.call_model(state, prompt, self._model_call_with_fallback)

    def apply_after_model(self, state: AgentState, output: str) -> str:
        """Stage 4: run after-model middleware."""
        return self.middleware.run_after_model(state, output)

    def finalize_with_output(self, state: AgentState, output: str) -> AgentState:
        """Stage 5: store report, generate citations, and finalize state."""
        state.report = output
        return self._finalize_state(state)

    def stream_model_collect(self, state: AgentState, prompt: str, chunk_size: int = 80) -> tuple[str, list[dict[str, Any]]]:
        """Stage 3 (streaming): collect deltas and return full output plus events."""
        # Keep collect mode for tests and non-realtime flows. Realtime should use stream_model_iter.
        events: list[dict[str, Any]] = []
        stream_iter = self.stream_model_iter(state, prompt, chunk_size=chunk_size)
        output = ""
        while True:
            try:
                events.append(next(stream_iter))
            except StopIteration as stop:
                output = str(stop.value or "")
                break
        return output, events

    def stream_model_iter(self, state: AgentState, prompt: str, chunk_size: int = 80) -> Iterator[dict[str, Any]]:
        """Stage 3 (true streaming): emit answer deltas and return final output on completion."""
        # Trigger budget middleware model-call counting and threshold checks up front.
        self.middleware.call_model(state, prompt, lambda s, p: "")
        output = ""
        # Trigger budget middleware model-call counting and threshold checks up front.
        try:
            if self.external_model_stream_call is None:
                raise RuntimeError("external stream model is not configured")
            chunks: list[str] = []
            for delta in self.external_model_stream_call(state, prompt):
                if not delta:
                    continue
                chunks.append(delta)
                yield {"event": "answer_delta", "data": {"delta": delta}}
            output = "".join(chunks)
            if not output.strip():
                raise RuntimeError("external stream returned empty output")
            # Gateway writes provider/model after upstream stream ends, then we emit source metadata.
            yield (
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
            yield (
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
                yield {"event": "answer_delta", "data": {"delta": output[idx : idx + chunk_size]}}
        return output

    def _prepare_state(self, state: AgentState, memory_hint: list[dict[str, Any]] | None = None) -> str:
        self.middleware.run_before_agent(state)
        self.trace_emit(state.trace_id, "before_agent", {"question": state.question})
        intent_result = route_intent_with_confidence(state.question)
        state.intent = intent_result.intent
        state.analysis["intent_confidence"] = intent_result.confidence
        self.trace_emit(
            state.trace_id,
            "router",
            {
                "intent": state.intent,
                "intent_confidence": intent_result.confidence,
                "intent_rule_conflict": intent_result.conflict,
                "intent_matched_keywords": intent_result.matched,
            },
        )
        # Preserve caller-side metadata and only fill retrieval defaults when absent.
        plan = dict(state.retrieval_plan or {})
        plan.setdefault("top_k_vector", 12)
        plan.setdefault("top_k_bm25", 20)
        plan.setdefault("rerank_top_n", 10)
        plan["memory_hint_count"] = len(memory_hint or [])
        state.retrieval_plan = plan
        _ = self._call_tool("data", "quote_tool", {"stock_codes": state.stock_codes or ["SH600000"]})
        self.trace_emit(state.trace_id, "data_plan", state.retrieval_plan)

        question = state.question
        if state.stock_codes:
            question = f"{question} {' '.join(state.stock_codes)}"
        if memory_hint:
            question = f"{question}\nMemory summary: {memory_hint[0]['content']}"

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
                    "retrieval_track": "graph_summary",
                    "metadata": {"retrieval_track": "graph_summary", "rerank_score": 0.8},
                    "rerank_score": 0.8,
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
                        "retrieval_track": "graph_relation",
                        "metadata": {"retrieval_track": "graph_relation", "rerank_score": 0.78},
                        "rerank_score": 0.78,
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
                    # Preserve rerank metadata for downstream citation explainability/audit.
                    "retrieval_track": str((i.metadata or {}).get("retrieval_track", "")),
                    "metadata": dict(i.metadata or {}),
                    "rerank_score": float(i.score),
                }
                for i in items
            ]
        self.trace_emit(state.trace_id, "retrieval", {"mode": state.mode, "evidence": len(state.evidence_pack)})
        # Preserve pre-analysis telemetry fields (intent confidence/timeout metadata)
        # and merge structured analysis snapshot afterwards.
        preserved_analysis = dict(state.analysis)
        preserved_analysis.update(self._analyze(state))
        state.analysis = preserved_analysis
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
        question = str(state.question or "").lower()
        graph_keywords = (
            "\u5173\u7cfb",
            "\u6f14\u5316",
            "\u5173\u8054",
            "\u4ea7\u4e1a\u94fe",
            "\u80a1\u6743",
            "graph",
            "network",
        )
        return any(k in question for k in graph_keywords)

    def _should_use_deep_agents(self, state: AgentState) -> bool:
        question = str(state.question or "").lower()
        deep_keywords = (
            "\u591a\u7ef4",
            "\u5e76\u884c",
            "\u957f\u671f",
            "\u5bf9\u6bd4",
            "deep",
            "multi-step",
        )
        return state.intent in ("deep", "compare") or any(k in question for k in deep_keywords)

    def _plan_subtasks(self, question: str) -> list[str]:
        base = str(question or "").strip()
        return [
            f"{base}: financial dimension",
            f"{base}: industry dimension",
            f"{base}: risk dimension",
        ]

    def _deep_retrieve(self, question: str, state: AgentState) -> list[RetrievalItem]:
        subtasks = self._plan_subtasks(question)
        state.analysis["deep_subtasks"] = subtasks
        subtask_timeout_seconds = max(0.2, float(getattr(self.middleware.ctx.settings, "deep_subtask_timeout_seconds", 2.5)))
        timeout_subtasks: list[str] = []
        failed_subtasks: list[dict[str, str]] = []
        merged: list[RetrievalItem] = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {q: pool.submit(self.retriever.retrieve, q, 8, 12, 5) for q in subtasks}
            for subtask, fut in futures.items():
                try:
                    merged.extend(fut.result(timeout=subtask_timeout_seconds))
                except FuturesTimeoutError:
                    timeout_subtasks.append(subtask)
                    fut.cancel()
                except Exception as ex:  # noqa: BLE001
                    failed_subtasks.append({"subtask": subtask, "error": str(ex)[:200]})
        uniq: dict[tuple[str, str], RetrievalItem] = {}
        for item in merged:
            uniq[(item.source_id, item.text)] = item
        items = list(uniq.values())
        items.sort(key=lambda x: x.score, reverse=True)
        state.analysis["timeout_subtasks"] = timeout_subtasks
        if failed_subtasks:
            state.analysis["failed_subtasks"] = failed_subtasks
        self.trace_emit(
            state.trace_id,
            "deep_agents",
            {
                "subtask_count": len(subtasks),
                "merged_items": len(items),
                "timeout_subtasks": timeout_subtasks,
                "failed_subtasks": len(failed_subtasks),
                "subtask_timeout_seconds": subtask_timeout_seconds,
            },
        )
        return items[: state.retrieval_plan["rerank_top_n"]]

    def _analyze(self, state: AgentState) -> dict[str, Any]:
        positives = [e for e in state.evidence_pack if e.get("reliability_score", 0) >= 0.9]
        risks = [e for e in state.evidence_pack if e.get("reliability_score", 0) < 0.75]
        regime_ctx = state.market_regime_context if isinstance(state.market_regime_context, dict) else {}
        return {
            "fact_count": len(state.evidence_pack),
            "high_confidence_count": len(positives),
            "low_confidence_count": len(risks),
            "summary": "Evidence suggests possible earnings improvement, but sector volatility remains a key risk.",
            "deep_subtasks": state.analysis.get("deep_subtasks", []),
            "market_regime": {
                "label": str(regime_ctx.get("regime_label", "")),
                "confidence": float(regime_ctx.get("regime_confidence", 0.0) or 0.0),
                "risk_bias": str(regime_ctx.get("risk_bias", "")),
            },
        }

    def _build_prompt(self, state: AgentState) -> str:
        regime_ctx = state.market_regime_context if isinstance(state.market_regime_context, dict) else {}
        regime_line = (
            f"- [a_share_regime] label={regime_ctx.get('regime_label', 'unknown')}, "
            f"risk_bias={regime_ctx.get('risk_bias', 'neutral')}, "
            f"confidence={float(regime_ctx.get('regime_confidence', 0.0) or 0.0):.3f}"
        )
        evidence = "\n".join(
            f"- [{e['source_id']}] {e['text']} (score={e['reliability_score']})" for e in state.evidence_pack[:6]
        )
        evidence = f"{evidence}\n{regime_line}" if evidence else regime_line
        if self.prompt_renderer:
            rendered, prompt_meta = self.prompt_renderer(
                {
                    "question": state.question,
                    "stock_codes": state.stock_codes,
                    "evidence": evidence,
                    # Keep a structured regime payload so prompt templates can consume it directly.
                    "market_regime": json.dumps(regime_ctx, ensure_ascii=False),
                }
            )
            self.trace_emit(state.trace_id, "prompt_meta", prompt_meta)
            return rendered
        return (
            "You are an A-share analysis assistant.\n"
            f"Question: {state.question}\n"
            f"Mode: {state.mode}\n"
            f"Analysis summary: {state.analysis.get('summary', '')}\n"
            f"Evidence:\n{evidence}\n"
            "Return a concise conclusion and include a disclaimer."
        )

    def _synthesize_model_output(self, state: AgentState, prompt: str) -> str:
        _ = prompt
        symbols = ",".join(state.stock_codes) if state.stock_codes else "the target symbol"
        high = state.analysis.get("high_confidence_count", 0)
        low = state.analysis.get("low_confidence_count", 0)
        fact_count = state.analysis.get("fact_count", 0)
        top_facts = [e["text"] for e in state.evidence_pack[:2] if e.get("text")]
        top_fact_text = "; ".join(top_facts) if top_facts else "No high-quality evidence snippets available"

        if high >= max(1, low):
            stance = "evidence consistency is strong and near-term structural opportunities can be tracked"
        elif low > high:
            stance = "evidence divergence is elevated, so observation is advised until more disclosures arrive"
        else:
            stance = "current evidence is limited and a neutral stance is recommended"

        risk_line = "; ".join(state.risk_flags) if state.risk_flags else "No high-risk control flags triggered"
        return (
            f"Conclusion: {symbols} currently has {fact_count} evidence items; {stance}.\n"
            f"Key facts: {top_fact_text}.\n"
            f"Risk notes: {risk_line}.\n"
            "For research reference only. This is not investment advice."
        )

    def _model_call_with_fallback(self, state: AgentState, prompt: str) -> str:
        """Prefer external LLM and fall back to local synthesis on failure."""
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
            meta = e.get("metadata", {})
            if not isinstance(meta, dict):
                meta = {}
            retrieval_track = str(e.get("retrieval_track", "")).strip() or str(meta.get("retrieval_track", "")).strip()
            citations.append(
                {
                    "source_id": e["source_id"],
                    "source_url": e["source_url"],
                    "event_time": e.get("event_time"),
                    "reliability_score": e.get("reliability_score", 0.5),
                    "excerpt": e["text"][:120],
                    "retrieval_track": retrieval_track or "unknown_track",
                    "rerank_score": float(e.get("rerank_score", 0.0) or 0.0),
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
