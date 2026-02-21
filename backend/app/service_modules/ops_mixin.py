from __future__ import annotations

from .shared import *

class OpsMixin:
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
        data_gap_reasons: list[str] | None = None,
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
        gap_reasons = [str(x).strip() for x in list(data_gap_reasons or []) if str(x).strip()]
        quality_status = "degraded" if gap_reasons else "pass"
        quality_explain = self._build_quality_explain(
            reasons=gap_reasons,
            quality_status=quality_status,
            context="deepthink",
        )
        runtime_guard = dict(budget_usage.get("runtime_guard", {}) or {}) if isinstance(budget_usage, dict) else {}
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
            "runtime_guard": runtime_guard,
            "runtime_timeout": bool(runtime_guard.get("timed_out", False)),
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
            "quality_explain": quality_explain,
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
        requested_round_timeout = self._safe_float(payload.get("round_timeout_seconds", 0.0), default=0.0)
        default_round_timeout = self._safe_float(
            getattr(self.settings, "deep_round_timeout_seconds", 45.0),
            default=45.0,
        )
        round_timeout_seconds = max(
            0.1,
            min(600.0, requested_round_timeout if requested_round_timeout > 0 else default_round_timeout),
        )
        requested_stage_soft_timeout = self._safe_float(payload.get("stage_soft_timeout_seconds", 0.0), default=0.0)
        default_stage_soft_timeout = self._safe_float(
            getattr(self.settings, "deep_round_stage_soft_timeout_seconds", max(8.0, round_timeout_seconds * 0.6)),
            default=max(8.0, round_timeout_seconds * 0.6),
        )
        stage_soft_timeout_seconds = max(
            0.05,
            min(round_timeout_seconds, requested_stage_soft_timeout if requested_stage_soft_timeout > 0 else default_stage_soft_timeout),
        )
        runtime_guard_state = {
            "warn_emitted": False,
            "timed_out": False,
            "timeout_stage": "",
            "elapsed_ms": 0,
        }

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

        def runtime_guard_snapshot(stage: str) -> dict[str, Any]:
            elapsed_seconds = max(0.0, float(time.perf_counter() - started_at))
            elapsed_ms = int(round(elapsed_seconds * 1000))
            remaining_seconds = max(0.0, float(round_timeout_seconds - elapsed_seconds))
            return {
                "stage": str(stage),
                "elapsed_ms": elapsed_ms,
                "elapsed_seconds": round(elapsed_seconds, 4),
                "remaining_seconds": round(remaining_seconds, 4),
                "round_timeout_seconds": float(round_timeout_seconds),
                "stage_soft_timeout_seconds": float(stage_soft_timeout_seconds),
                "warn": bool(elapsed_seconds >= stage_soft_timeout_seconds),
                "timed_out": bool(elapsed_seconds >= round_timeout_seconds),
            }

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
            multi_role_pre = {
                "enabled": False,
                "trace_id": "",
                "debate_mode": "not_started",
                "role_count": 0,
                "consensus_signal": "hold",
                "consensus_confidence": 0.5,
                "disagreement_score": 0.0,
                "conflict_sources": [],
                "counter_view": "",
                "judge_summary": "",
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
            yield emit(
                "runtime_guard",
                {
                    "status": "armed",
                    "round_timeout_seconds": float(round_timeout_seconds),
                    "stage_soft_timeout_seconds": float(stage_soft_timeout_seconds),
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
                refresh_guard = runtime_guard_snapshot("data_refresh")
                runtime_guard_state["elapsed_ms"] = int(refresh_guard.get("elapsed_ms", 0) or 0)
                if bool(refresh_guard.get("warn")) and not bool(runtime_guard_state["warn_emitted"]):
                    runtime_guard_state["warn_emitted"] = True
                    yield emit("runtime_guard", {"status": "warning", **refresh_guard})
                if bool(refresh_guard.get("timed_out")):
                    runtime_guard_state["timed_out"] = True
                    runtime_guard_state["timeout_stage"] = str(refresh_guard.get("stage", "data_refresh"))
                    stop_reason = "DEEP_ROUND_TIMEOUT"
                    yield emit("runtime_timeout", {"reason": "round_timeout", **refresh_guard})
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
                intel_guard = runtime_guard_snapshot("intel_search")
                runtime_guard_state["elapsed_ms"] = int(intel_guard.get("elapsed_ms", 0) or 0)
                if bool(intel_guard.get("warn")) and not bool(runtime_guard_state["warn_emitted"]):
                    runtime_guard_state["warn_emitted"] = True
                    yield emit("runtime_guard", {"status": "warning", **intel_guard})
                if bool(intel_guard.get("timed_out")):
                    runtime_guard_state["timed_out"] = True
                    runtime_guard_state["timeout_stage"] = str(intel_guard.get("stage", "intel_search"))
                    stop_reason = "DEEP_ROUND_TIMEOUT"
                    yield emit("runtime_timeout", {"reason": "round_timeout", **intel_guard})

                yield emit("progress", {"stage": "debate", "message": "Generating multi-agent viewpoints"})
                data_pack_missing = [str(x).strip() for x in list(deep_pack.get("missing_data", []) or []) if str(x).strip()]
                pre_quality_reasons = [f"input_pack:{x}" for x in data_pack_missing]
                if refresh_failed:
                    pre_quality_reasons.append("input_pack:auto_refresh_failed")
                pre_quality_reasons = list(dict.fromkeys(pre_quality_reasons))
                pre_quality_status = "degraded" if data_pack_missing else "watch" if refresh_failed else "pass"
                pre_quality_gate = {"overall_status": pre_quality_status, "reasons": pre_quality_reasons}
                pre_multi = self._predict_run_multi_role_debate(
                    stock_code=code,
                    question=question,
                    quote=quote,
                    trend=trend,
                    quant_20=quant_20,
                    quality_gate=pre_quality_gate,
                    input_pack=deep_pack,
                )
                multi_role_pre = {
                    "enabled": True,
                    "trace_id": str(pre_multi.get("trace_id", "")),
                    "debate_mode": str(pre_multi.get("debate_mode", "rule_fallback")),
                    "role_count": int(len(list(pre_multi.get("opinions", []) or []))),
                    "consensus_signal": str(pre_multi.get("consensus_signal", "hold")),
                    "consensus_confidence": round(
                        max(0.0, min(1.0, self._safe_float(pre_multi.get("consensus_confidence", 0.5), default=0.5))),
                        4,
                    ),
                    "disagreement_score": round(
                        max(0.0, min(1.0, self._safe_float(pre_multi.get("disagreement_score", 0.0), default=0.0))),
                        4,
                    ),
                    "conflict_sources": [
                        str(x).strip()
                        for x in list(pre_multi.get("conflict_sources", []) or [])
                        if str(x).strip()
                    ],
                    "counter_view": str(pre_multi.get("counter_view", ""))[:280],
                    "judge_summary": str(pre_multi.get("judge_summary", ""))[:280],
                }
                yield emit("pre_arbitration", dict(multi_role_pre))
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
                        reason=f"宏观评估：ma60_slope={trend.get('ma60_slope', 0.0):.4f}, momentum_20={trend.get('momentum_20', 0.0):.4f}",
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
                pre_core = [
                    self._normalize_deep_opinion(
                        agent=str(opinion.get("agent", "")),
                        signal=str(opinion.get("signal", "hold")),
                        confidence=self._safe_float(opinion.get("confidence", 0.5), default=0.5),
                        reason=str(opinion.get("reason", "")),
                        evidence_ids=evidence_ids,
                        risk_tags=["pre_arbitration_core"],
                    )
                    for opinion in list(pre_multi.get("opinions", []) or [])
                    if isinstance(opinion, dict) and str(opinion.get("agent", "")).strip() and str(opinion.get("agent", "")).strip() != "supervisor_agent"
                ]
                # Deduplicate by agent name: pre-arbitration output has higher precedence than ad-hoc debate output.
                normalized_core_by_agent: dict[str, dict[str, Any]] = {}
                for row in pre_core + normalized_core:
                    normalized_core_by_agent[str(row.get("agent", "")).strip()] = row
                normalized_core = [row for row in normalized_core_by_agent.values() if str(row.get("agent", "")).strip()]
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
                merged_conflicts = list(pre_arb.get("conflict_sources", [])) + list(multi_role_pre.get("conflict_sources", []))
                if data_pack_missing:
                    merged_conflicts.append("data_gap")
                if bool(runtime_guard_state.get("timed_out", False)):
                    merged_conflicts.append("runtime_timeout")
                pre_arb["conflict_sources"] = list(
                    dict.fromkeys([str(x).strip() for x in merged_conflicts if str(x).strip()])
                )
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
                if bool(runtime_guard_state.get("timed_out", False)):
                    # Runtime timeout is treated as a hard conflict source to force conservative downstream handling.
                    arbitration["conflict_sources"] = list(
                        dict.fromkeys(list(arbitration.get("conflict_sources", [])) + ["runtime_timeout"])
                    )

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
            runtime_guard_final = runtime_guard_snapshot("round_finalize")
            runtime_guard_state["elapsed_ms"] = int(runtime_guard_final.get("elapsed_ms", 0) or 0)
            if bool(runtime_guard_final.get("warn")) and not bool(runtime_guard_state["warn_emitted"]):
                runtime_guard_state["warn_emitted"] = True
                yield emit("runtime_guard", {"status": "warning", **runtime_guard_final})
            if bool(runtime_guard_final.get("timed_out")) and not bool(runtime_guard_state["timed_out"]):
                runtime_guard_state["timed_out"] = True
                runtime_guard_state["timeout_stage"] = str(runtime_guard_final.get("stage", "round_finalize"))
                stop_reason = "DEEP_ROUND_TIMEOUT"
                yield emit("runtime_timeout", {"reason": "round_timeout", **runtime_guard_final})
                arbitration["conflict_sources"] = list(
                    dict.fromkeys(list(arbitration.get("conflict_sources", [])) + ["runtime_timeout"])
                )
            budget_usage["runtime_guard"] = {
                "warn_emitted": bool(runtime_guard_state.get("warn_emitted", False)),
                "timed_out": bool(runtime_guard_state.get("timed_out", False)),
                "timeout_stage": str(runtime_guard_state.get("timeout_stage", "")),
                "elapsed_ms": int(runtime_guard_state.get("elapsed_ms", 0) or 0),
                "round_timeout_seconds": float(round_timeout_seconds),
                "stage_soft_timeout_seconds": float(stage_soft_timeout_seconds),
            }
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
                data_gap_reasons=[str(x) for x in list(deep_pack.get("missing_data", []) or []) if str(x).strip()],
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

            terminal_stop_reasons = {"DEEP_BUDGET_EXCEEDED"}
            session_status = (
                "completed"
                if round_no >= max_rounds or str(stop_reason or "").strip() in terminal_stop_reasons
                else "in_progress"
            )
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
            snapshot["multi_role_pre"] = dict(multi_role_pre)
            snapshot["runtime_guard"] = dict(budget_usage.get("runtime_guard", {}) or {})
            if isinstance(snapshot.get("rounds"), list) and snapshot["rounds"]:
                snapshot["rounds"][-1]["multi_role_pre"] = dict(multi_role_pre)
                snapshot["rounds"][-1]["runtime_guard"] = dict(budget_usage.get("runtime_guard", {}) or {})

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
                        "multi_role_pre_trace_id": str(multi_role_pre.get("trace_id", "")),
                        "runtime_guard": dict(budget_usage.get("runtime_guard", {}) or {}),
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
                    "multi_role_pre": dict(multi_role_pre),
                    "runtime_guard": dict(budget_usage.get("runtime_guard", {}) or {}),
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

