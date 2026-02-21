from __future__ import annotations

from .shared import *

class QueryMixin:
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

            memory_hint = self._build_memory_hint(req.user_id, req.question)
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
            analysis_brief["intent_confidence"] = float(state.analysis.get("intent_confidence", 0.0) or 0.0)
            analysis_brief["retrieval_track"] = self._summarize_retrieval_tracks(normalized_citations)
            analysis_brief["model_call_count"] = int(self.workflow.middleware.ctx.model_call_count)
            analysis_brief["tool_call_count"] = int(self.workflow.middleware.ctx.tool_call_count)
            analysis_brief["timeout_reason"] = ""
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
            self.traces.emit(
                trace_id,
                "runtime_observability",
                {
                    "intent_confidence": analysis_brief["intent_confidence"],
                    "retrieval_track": analysis_brief["retrieval_track"],
                    "model_call_count": analysis_brief["model_call_count"],
                    "tool_call_count": analysis_brief["tool_call_count"],
                    "timeout_reason": "",
                },
            )
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
                "intent_confidence": 0.0,
                "retrieval_track": {},
                "model_call_count": 0,
                "tool_call_count": 0,
                "timeout_reason": error_code,
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
        memory_hint = self._build_memory_hint(req.user_id, req.question)
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
                "intent_confidence": float(state.analysis.get("intent_confidence", 0.0) or 0.0),
                "retrieval_track": self._summarize_retrieval_tracks(citations),
                "model_call_count": int(self.workflow.middleware.ctx.model_call_count),
                "tool_call_count": int(self.workflow.middleware.ctx.tool_call_count),
                "timeout_reason": "",
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

    def _build_memory_hint(self, user_id: str, question: str) -> list[dict[str, Any]]:
        """Use similarity retrieval first, then fallback to latest memory records."""
        similar = self.memory.similarity_search(user_id=user_id, query=question, top_k=3)
        if similar:
            return similar
        return self.memory.list_memory(user_id, limit=3)

    @staticmethod
    def _summarize_retrieval_tracks(citations: list[dict[str, Any]]) -> dict[str, int]:
        """Aggregate retrieval attribution tracks for observability dashboards."""
        counter = Counter()
        for row in citations:
            if not isinstance(row, dict):
                continue
            track = str(row.get("retrieval_track", "")).strip() or "unknown_track"
            counter[track] += 1
        return dict(counter)

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

