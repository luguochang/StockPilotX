from __future__ import annotations

from .shared import *

class PredictMixin:
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
            # Only hard-gap reasons should force degraded quality.
            # Watch-level gaps (for example research_insufficient) keep the run usable.
            has_hard_gap = any(self._predict_reason_severity(code) == "degraded" for code in merged_reasons)
            if has_hard_gap:
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
        latest_metrics = dict(latest_eval.get("metrics", {}) or {})
        # Keep metric sources explicit so frontend can avoid mixed-confidence rendering.
        result["metrics_live"] = latest_metrics if metric_mode == "live" else {}
        result["metrics_backtest"] = latest_metrics if metric_mode == "backtest_proxy" else {}
        result["metrics_simulated"] = latest_metrics if metric_mode == "simulated" else {}
        result["eval_provenance"] = {
            "coverage_rows": int(latest_metrics.get("coverage", 0) or 0),
            "evaluated_stocks": int(latest_eval.get("evaluated_stocks", 0) or 0),
            "skipped_stocks": list(latest_eval.get("skipped_stocks", []) or []),
            "history_modes": dict(latest_eval.get("history_modes", {}) or {}),
            "fallback_reason": str(latest_eval.get("fallback_reason", "")),
            "run_data_quality": str(quality.get("data_quality", "unknown")),
        }
        result["quality_gate"] = self._predict_build_quality_gate(result, latest_eval)
        result["quality_gate_summary"] = self._predict_build_quality_gate_summary(result["quality_gate"])

        # Multi-role arbitration is now a first-class output for predict.
        # We cap per-run debate fan-out to keep latency bounded for large pools.
        pack_by_code = {str(pack.get("stock_code", "")).strip().upper(): pack for pack in data_packs}
        debate_cap = min(8, len(list(result.get("results", []))))
        debate_rows: list[dict[str, Any]] = []
        for item in list(result.get("results", []))[:debate_cap]:
            code = str(item.get("stock_code", "")).strip().upper()
            quote = self._latest_quote(code) or {}
            bars = self._history_bars(code, limit=160)
            trend = self._trend_metrics(bars) if len(bars) >= 30 else {}
            horizons_rows = list(item.get("horizons", []) or [])
            quant_20 = next(
                (x for x in horizons_rows if str(x.get("horizon", "")).strip().lower() == "20d"),
                horizons_rows[0] if horizons_rows else {},
            )
            debate_rows.append(
                self._predict_run_multi_role_debate(
                    stock_code=code,
                    question=str(payload.get("question", "")).strip() or f"predict:{code}",
                    quote=quote,
                    trend=trend,
                    quant_20=quant_20,
                    quality_gate=dict(result.get("quality_gate", {}) or {}),
                    input_pack=pack_by_code.get(code),
                )
            )

        primary_debate = debate_rows[0] if debate_rows else {}
        result["multi_role_enabled"] = True
        result["multi_role_trace_id"] = str(primary_debate.get("trace_id", ""))
        result["multi_role_truncated"] = len(list(result.get("results", []))) > debate_cap
        result["multi_role_debate"] = debate_rows
        # Keep compatibility with existing frontend by exposing a top-level primary view.
        result["role_opinions"] = list(primary_debate.get("opinions", []))
        result["judge_summary"] = str(primary_debate.get("judge_summary", ""))
        result["conflict_sources"] = list(primary_debate.get("conflict_sources", []))
        result["consensus_signal"] = str(primary_debate.get("consensus_signal", "hold"))
        result["consensus_confidence"] = float(primary_debate.get("consensus_confidence", 0.5) or 0.5)
        result["engine_profile"] = {
            "prediction_engine": "quant_rule_v1",
            "llm_used_in_scoring": False,
            "llm_used_in_explain": bool(self.settings.llm_external_enabled and self.llm_gateway.providers),
            "latency_mode": "fast_local_compute",
        }
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

    @staticmethod
    def _predict_reason_dimension(reason: str) -> str:
        raw = str(reason or "").strip().lower()
        if raw.startswith("input_pack:"):
            return "evidence_quality"
        if raw.startswith("history_") or raw.startswith("quote_") or raw in {"empty_prediction_result"}:
            return "history_quality"
        if raw.startswith("metrics_") or raw.startswith("eval:") or raw in {"insufficient_backtest_rows"}:
            return "metric_quality"
        return "evidence_quality"

    @staticmethod
    def _predict_merge_status(current: str, incoming: str) -> str:
        rank = {"pass": 0, "watch": 1, "degraded": 2}
        left = str(current or "pass").strip().lower()
        right = str(incoming or "pass").strip().lower()
        if left not in rank:
            left = "pass"
        if right not in rank:
            right = "degraded"
        return left if rank[left] >= rank[right] else right

    def _predict_reason_severity(self, reason: str) -> str:
        severity = str(self._predict_reason_detail(reason).get("severity", "degraded")).strip().lower()
        if severity not in {"pass", "watch", "degraded"}:
            return "degraded"
        return severity

    def _predict_reason_detail(self, reason: str) -> dict[str, str]:
        raw = str(reason or "").strip()
        key = raw.lower()
        dimension = self._predict_reason_dimension(raw)
        if key == "history_not_real":
            return {
                "code": raw,
                "dimension": dimension,
                "title": "历史行情非真实样本",
                "impact": "预测可靠性下降，适合做方向性参考，不宜用于高置信结论。",
                "action": "补齐真实历史K线后重新运行。",
                "severity": "degraded",
            }
        if key == "history_sample_insufficient":
            return {
                "code": raw,
                "dimension": dimension,
                "title": "历史样本量不足",
                "impact": "周期统计不稳定，短中期信号波动会放大。",
                "action": "扩充历史窗口并重跑预测。",
                "severity": "degraded",
            }
        if key == "history_fetch_degraded":
            return {
                "code": raw,
                "dimension": dimension,
                "title": "历史数据抓取发生降级",
                "impact": "部分关键因子可能使用兜底序列。",
                "action": "检查数据源连通性后重跑。",
                "severity": "degraded",
            }
        if key == "quote_source_mock":
            return {
                "code": raw,
                "dimension": dimension,
                "title": "实时行情来源为兜底源",
                "impact": "价格与成交强度可能存在偏差。",
                "action": "确认实时行情源状态后再决策。",
                "severity": "degraded",
            }
        if key == "metrics_simulated":
            return {
                "code": raw,
                "dimension": dimension,
                "title": "评测为模拟模式",
                "impact": "指标仅能做相对排序，不代表真实收益能力。",
                "action": "补齐真实回测样本后复核。",
                "severity": "watch",
            }
        if key.startswith("eval:"):
            return {
                "code": raw,
                "dimension": dimension,
                "title": "评测链路已降级",
                "impact": f"回退原因: {raw.split(':', 1)[1] if ':' in raw else 'unknown'}",
                "action": "补齐评测样本并重新触发评测。",
                "severity": "watch",
            }
        if key.startswith("input_pack:"):
            missing_code = key.split(":", 1)[1] if ":" in key else ""
            missing_map: dict[str, tuple[str, str, str, str]] = {
                "quote_missing": ("实时行情缺失", "缺少最新价格上下文，信号时效性下降。", "检查行情抓取链路并补抓。", "degraded"),
                "history_insufficient": ("历史行情不足", "趋势因子样本不足，稳定性变弱。", "补齐历史K线后重跑。", "degraded"),
                "history_30d_insufficient": ("近30日样本不足", "短周期研判依据不充分。", "补齐近30日样本。", "degraded"),
                "financial_missing": ("财务快照缺失", "估值与盈利验证链路不完整。", "补齐财务数据后重跑。", "degraded"),
                "announcement_missing": ("公告样本缺失", "公司事件影响难以校验。", "补抓公告并刷新。", "watch"),
                "news_insufficient": ("新闻样本不足", "事件驱动信息覆盖不足。", "补抓新闻数据。", "watch"),
                # Business decision: research shortage should not hard-block predict outputs.
                "research_insufficient": ("研报证据不足", "机构观点与一致预期覆盖不足。", "补齐研报摘要后重跑。", "watch"),
                "fund_missing": ("资金面样本缺失", "资金流驱动判断弱化。", "补抓资金面数据。", "watch"),
                "macro_insufficient": ("宏观样本不足", "宏观扰动未充分纳入解释。", "补齐宏观数据。", "watch"),
            }
            title, impact, action, severity = missing_map.get(
                missing_code,
                ("输入数据包存在缺口", f"缺失项: {missing_code or 'unknown'}。", "补齐缺失数据后复核。", "watch"),
            )
            return {
                "code": raw,
                "dimension": dimension,
                "title": title,
                "impact": impact,
                "action": action,
                "severity": severity,
            }
        return {
            "code": raw,
            "dimension": dimension,
            "title": "存在待处理降级项",
            "impact": f"降级代码: {raw or 'unknown'}",
            "action": "建议复核数据覆盖并重新运行。",
            "severity": "watch",
        }

    def _predict_build_quality_gate(self, result: dict[str, Any], latest_eval: dict[str, Any]) -> dict[str, Any]:
        reasons = [str(x).strip() for x in list(result.get("degrade_reasons", []) or []) if str(x).strip()]
        metric_mode = str(result.get("metric_mode", latest_eval.get("metric_mode", "simulated"))).strip() or "simulated"
        fallback_reason = str((result.get("eval_provenance", {}) or {}).get("fallback_reason", latest_eval.get("fallback_reason", ""))).strip()
        coverage_rows = int((result.get("eval_provenance", {}) or {}).get("coverage_rows", (latest_eval.get("metrics", {}) or {}).get("coverage", 0)) or 0)
        evaluated_stocks = int((result.get("eval_provenance", {}) or {}).get("evaluated_stocks", latest_eval.get("evaluated_stocks", 0)) or 0)
        if metric_mode == "simulated":
            reasons.append("metrics_simulated")
        if fallback_reason:
            reasons.append(f"eval:{fallback_reason}")
        dedup_reasons = list(dict.fromkeys(reasons))

        dims: dict[str, dict[str, Any]] = {
            "history_quality": {"status": "pass", "reasons": []},
            "evidence_quality": {"status": "pass", "reasons": []},
            "metric_quality": {"status": "pass", "reasons": []},
        }
        reason_details: list[dict[str, str]] = []
        actions: list[str] = []
        for reason in dedup_reasons:
            detail = self._predict_reason_detail(reason)
            dim_key = str(detail.get("dimension", "evidence_quality"))
            if dim_key not in dims:
                dim_key = "evidence_quality"
            dims[dim_key]["reasons"].append(str(detail.get("code", reason)))
            dims[dim_key]["status"] = self._predict_merge_status(
                str(dims[dim_key].get("status", "pass")),
                str(detail.get("severity", "degraded")),
            )
            reason_details.append(detail)
            action = str(detail.get("action", "")).strip()
            if action:
                actions.append(action)

        # backtest_proxy can be used for research, but low sample coverage should be treated as watch.
        if dims["metric_quality"]["status"] == "pass" and metric_mode == "backtest_proxy":
            if coverage_rows < 80 or evaluated_stocks < 2:
                dims["metric_quality"]["status"] = "watch"
                actions.append("扩大真实回测样本后复核关键信号。")

        statuses = [str(dims[k]["status"]) for k in ("history_quality", "evidence_quality", "metric_quality")]
        if "degraded" in statuses:
            overall_status = "degraded"
        elif "watch" in statuses:
            overall_status = "watch"
        else:
            overall_status = "pass"

        rank = {"pass": 0, "watch": 1, "degraded": 2}
        sorted_details = sorted(
            reason_details,
            key=lambda item: rank.get(str(item.get("severity", "degraded")).strip().lower(), 2),
            reverse=True,
        )

        if overall_status == "degraded":
            primary = sorted_details[0] if sorted_details else {
                "title": "存在降级项",
                "impact": "当前结果可信度下降。",
                "action": "建议补齐样本后重跑。",
            }
            user_message = f"当前预测处于降级模式：{primary['title']}。{primary['impact']} 建议：{primary['action']}"
        elif overall_status == "watch":
            user_message = "当前结果可用于研究参考，但评测覆盖偏低，建议补齐样本后再使用高置信结论。"
        else:
            user_message = "数据与评测门禁通过，可用于研究参考（不构成投资建议）。"

        return {
            "overall_status": overall_status,
            "dimensions": {
                key: {
                    "status": str(val.get("status", "pass")),
                    "reason_count": len(list(val.get("reasons", []))),
                    "reasons": list(val.get("reasons", [])),
                }
                for key, val in dims.items()
            },
            "reasons": dedup_reasons,
            "reason_details": reason_details,
            "user_message": user_message,
            "actions": list(dict.fromkeys([x for x in actions if str(x).strip()])),
        }

    def _predict_build_quality_gate_summary(self, quality_gate: dict[str, Any]) -> dict[str, Any]:
        details = list(quality_gate.get("reason_details", []) or [])
        primary = details[0] if details else {}
        return {
            "status": str(quality_gate.get("overall_status", "pass")),
            "headline": str(primary.get("title", "质量门禁通过")),
            "impact": str(primary.get("impact", "当前结果可用于研究参考。")),
            "action": str(primary.get("action", "")),
            "reason_count": len(list(quality_gate.get("reasons", []) or [])),
        }

    def _predict_run_multi_role_debate(
        self,
        *,
        stock_code: str,
        question: str,
        quote: dict[str, Any],
        trend: dict[str, Any],
        quant_20: dict[str, Any],
        quality_gate: dict[str, Any],
        input_pack: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one predictable multi-role arbitration for predict/report/deepthink shared usage."""
        trace_id = self.traces.new_trace()
        rule_defaults = self._build_rule_based_debate_opinions(question, trend, quote, quant_20)
        base_opinions = list(rule_defaults)
        debate_mode = "rule_fallback"
        # Predict module prioritizes deterministic low-latency responses.
        # Keep LLM debate as a future opt-in, not the default request path.
        if False and self.settings.llm_external_enabled and self.llm_gateway.providers:
            llm_rows = self._build_llm_debate_opinions(question, stock_code, trend, quote, quant_20, rule_defaults)
            if llm_rows:
                base_opinions = llm_rows
                debate_mode = "llm_parallel"

        core_map = {str(row.get("agent", "")): row for row in base_opinions}
        # `predict` uses `overall_status`, while `report` currently uses `status`.
        # Accept both keys so the same arbitration helper can be reused cross-module.
        quality_status = str(quality_gate.get("overall_status", quality_gate.get("status", "pass"))).strip().lower() or "pass"
        quality_reasons = [str(x).strip() for x in list(quality_gate.get("reasons", []) or []) if str(x).strip()]
        missing_data_rows = list((input_pack or {}).get("missing_data", []))
        missing_data = [str(x).strip() for x in missing_data_rows if str(x).strip()]

        pm = self._normalize_deep_opinion(
            agent="pm_agent",
            signal=str(core_map.get("pm_agent", {}).get("signal", "hold")),
            confidence=float(core_map.get("pm_agent", {}).get("confidence", 0.62) or 0.62),
            reason=str(core_map.get("pm_agent", {}).get("reason", "Theme narrative view from PM role.")),
            evidence_ids=[],
            risk_tags=[],
        )
        quant = self._normalize_deep_opinion(
            agent="quant_agent",
            signal=str(core_map.get("quant_agent", {}).get("signal", "hold")),
            confidence=float(core_map.get("quant_agent", {}).get("confidence", 0.64) or 0.64),
            reason=str(core_map.get("quant_agent", {}).get("reason", "Quant factor and probability signal.")),
            evidence_ids=[],
            risk_tags=[],
        )
        risk = self._normalize_deep_opinion(
            agent="risk_agent",
            signal=str(core_map.get("risk_agent", {}).get("signal", "hold")),
            confidence=float(core_map.get("risk_agent", {}).get("confidence", 0.68) or 0.68),
            reason=str(core_map.get("risk_agent", {}).get("reason", "Drawdown and volatility constraints.")),
            evidence_ids=[],
            risk_tags=[],
        )

        momentum_20 = float(trend.get("momentum_20", 0.0) or 0.0)
        macro_signal = "hold"
        if momentum_20 > 0.03:
            macro_signal = "buy"
        elif momentum_20 < -0.03:
            macro_signal = "reduce"
        macro = self._normalize_deep_opinion(
            agent="macro_agent",
            signal=macro_signal,
            confidence=0.60,
            reason=(
                "Macro role reviewed regime proxy from trend and data freshness; "
                f"momentum_20={momentum_20:.4f}, quality_status={quality_status}."
            ),
            evidence_ids=[],
            risk_tags=(["macro_data_gap"] if "input_pack:macro_insufficient" in quality_reasons else []),
        )

        execution_signal = "hold" if str(risk.get("signal")) == "hold" else str(risk.get("signal"))
        execution = self._normalize_deep_opinion(
            agent="execution_agent",
            signal=execution_signal,
            confidence=0.61,
            reason=(
                "Execution role converts risk posture into sizing cadence. "
                f"risk_signal={risk.get('signal')}, quality_status={quality_status}."
            ),
            evidence_ids=[],
            risk_tags=[],
        )

        compliance_signal = "reduce" if quality_status == "degraded" else "hold"
        compliance = self._normalize_deep_opinion(
            agent="compliance_agent",
            signal=compliance_signal,
            confidence=0.72 if compliance_signal == "reduce" else 0.58,
            reason=(
                "Compliance role enforces conservative output under quality stress. "
                f"quality_reasons={','.join(quality_reasons) or 'none'}."
            ),
            evidence_ids=[],
            risk_tags=(["quality_block"] if compliance_signal == "reduce" else []),
        )

        critic_signal = "hold" if quality_status in {"pass", "watch"} else "reduce"
        critic = self._normalize_deep_opinion(
            agent="critic_agent",
            signal=critic_signal,
            confidence=0.66,
            reason=(
                "Critic role checks evidence coverage and logic consistency. "
                f"missing_data={','.join(missing_data) or 'none'}."
            ),
            evidence_ids=[],
            risk_tags=(["evidence_gap"] if missing_data else []),
        )

        opinions = [pm, quant, risk, macro, execution, compliance, critic]
        arbitration = self._arbitrate_opinions(opinions)
        consensus_conf = max(
            0.3,
            min(
                0.92,
                (1.0 - float(arbitration.get("disagreement_score", 0.0) or 0.0)) * 0.7
                + float(sum(float(x.get("confidence", 0.0)) for x in opinions) / max(1, len(opinions))) * 0.3,
            ),
        )
        supervisor = self._normalize_deep_opinion(
            agent="supervisor_agent",
            signal=str(arbitration.get("consensus_signal", "hold")),
            confidence=consensus_conf,
            reason=(
                "Supervisor arbitration merged role conflicts into one actionable stance; "
                f"conflicts={','.join(list(arbitration.get('conflict_sources', []))[:4]) or 'none'}."
            ),
            evidence_ids=[],
            risk_tags=[],
        )
        final_opinions = opinions + [supervisor]
        final_arb = self._arbitrate_opinions(final_opinions)
        final_arb["consensus_confidence"] = round(float(supervisor.get("confidence", 0.5) or 0.5), 4)

        judge_summary = (
            f"{stock_code} 多角色裁决：{final_arb['consensus_signal']}，"
            f"置信度 {final_arb['consensus_confidence']:.2f}，"
            f"冲突源 {','.join(final_arb['conflict_sources']) or 'none'}。"
        )
        self.traces.emit(
            trace_id,
            "predict_multi_role_done",
            {
                "stock_code": stock_code,
                "debate_mode": debate_mode,
                "consensus_signal": final_arb["consensus_signal"],
                "consensus_confidence": final_arb["consensus_confidence"],
                "conflict_sources": final_arb["conflict_sources"],
            },
        )
        return {
            "trace_id": trace_id,
            "stock_code": stock_code,
            "debate_mode": debate_mode,
            "opinions": final_opinions,
            "consensus_signal": str(final_arb.get("consensus_signal", "hold")),
            "consensus_confidence": float(final_arb.get("consensus_confidence", 0.5) or 0.5),
            "disagreement_score": float(final_arb.get("disagreement_score", 0.0) or 0.0),
            "conflict_sources": list(final_arb.get("conflict_sources", [])),
            "counter_view": str(final_arb.get("counter_view", "")),
            "judge_summary": judge_summary,
        }

    def predict_explain(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Build prediction explanation with optional LLM overlay (scoring remains quant-only)."""
        run_id = str(payload.get("run_id", "")).strip()
        if not run_id:
            raise ValueError("run_id is required")

        run = self.predict_get(run_id)
        if "error" in run:
            raise ValueError(f"predict run not found: {run_id}")

        items = list(run.get("results", []) or [])
        if not items:
            raise ValueError("prediction result is empty")

        target_code = str(payload.get("stock_code", "")).strip().upper().replace(".", "")
        target_item = next((x for x in items if str(x.get("stock_code", "")).upper() == target_code), None) if target_code else items[0]
        if not target_item:
            raise ValueError(f"stock_code not found in run: {target_code}")

        target_code = str(target_item.get("stock_code", "")).strip().upper()
        horizon = str(payload.get("horizon", "20d")).strip().lower()
        horizons = list(target_item.get("horizons", []) or [])
        target_h = next((x for x in horizons if str(x.get("horizon", "")).strip().lower() == horizon), None)
        if not target_h:
            target_h = next((x for x in horizons if str(x.get("horizon", "")).strip().lower() == "20d"), horizons[0] if horizons else None)
        if not target_h:
            raise ValueError("horizon result not found")

        latest_eval = self.prediction.eval_latest()
        quality_gate = dict(run.get("quality_gate", {}) or self._predict_build_quality_gate(run, latest_eval))

        signal = str(target_h.get("signal", "hold")).strip().lower()
        risk_tier = str(target_h.get("risk_tier", "medium")).strip().lower()
        up_prob = float(target_h.get("up_probability", 0.0) or 0.0)
        expected_excess = float(target_h.get("expected_excess_return", 0.0) or 0.0)
        horizon_name = str(target_h.get("horizon", horizon)).strip() or "20d"

        signal_label = {
            "strong_buy": "强增配",
            "buy": "增配",
            "hold": "持有",
            "reduce": "减配",
            "strong_reduce": "强减配",
        }.get(signal, signal or "持有")
        risk_label = {"low": "低", "medium": "中", "high": "高"}.get(risk_tier, risk_tier or "中")

        base_payload = {
            "summary": (
                f"{target_code} 在 {horizon_name} 维度给出「{signal_label}」信号，"
                f"上涨概率约 {up_prob * 100:.1f}% ，预期超额收益约 {expected_excess * 100:.2f}% 。"
            ),
            "drivers": [],
            "risks": [],
            "actions": [],
        }
        rationale = str(target_h.get("rationale", "")).strip()
        if rationale:
            base_payload["drivers"].append(f"量化因子依据：{rationale[:180]}")
        base_payload["drivers"].append(f"概率信号：up_probability={up_prob * 100:.1f}%")
        base_payload["drivers"].append(f"收益信号：expected_excess_return={expected_excess * 100:.2f}%")

        base_payload["risks"].append(f"风险分层：{risk_label}风险。")
        for detail in list(quality_gate.get("reason_details", []) or [])[:2]:
            title = str((detail or {}).get("title", "")).strip()
            impact = str((detail or {}).get("impact", "")).strip()
            if title or impact:
                base_payload["risks"].append(f"{title}：{impact}".strip("："))
        if not base_payload["risks"]:
            base_payload["risks"].append("当前未发现显著数据降级风险。")

        signal_action = {
            "strong_buy": "优先关注回调后的分批增配节奏，避免一次性重仓。",
            "buy": "可考虑小步增配，并设置触发复核条件。",
            "hold": "维持观察，等待新证据或更强信号。",
            "reduce": "优先控制仓位，关注风险释放节奏。",
            "strong_reduce": "降低敞口并优先防守，等待风险信号缓解。",
        }.get(signal, "先做小仓位试探，再根据新数据复核。")
        base_payload["actions"].append(signal_action)
        for action in list(quality_gate.get("actions", []) or [])[:3]:
            if str(action).strip():
                base_payload["actions"].append(str(action).strip())
        base_payload["actions"] = list(dict.fromkeys(base_payload["actions"]))[:4]

        llm_used = False
        provider = ""
        model = ""
        degraded_reason = ""
        trace_id = self.traces.new_trace()
        if not self.settings.llm_external_enabled:
            degraded_reason = "external_llm_disabled"
        elif not self.llm_gateway.providers:
            degraded_reason = "llm_provider_not_configured"
        else:
            prompt_payload = {
                "stock_code": target_code,
                "horizon": horizon_name,
                "signal": signal,
                "signal_label": signal_label,
                "risk_tier": risk_tier,
                "up_probability": round(up_prob, 6),
                "expected_excess_return": round(expected_excess, 6),
                "quality_gate": quality_gate,
                "rationale": rationale,
            }
            prompt_text = (
                "你是A股研究助手。请基于输入生成简洁、可执行的研究解释。\n"
                "要求：\n"
                "1) 不得承诺收益，不得给确定性结论；\n"
                "2) 必须体现数据质量限制；\n"
                "3) 输出严格 JSON 对象，字段仅包含 summary, drivers, risks, actions；\n"
                "4) drivers/risks/actions 各输出 2-4 条短句。\n\n"
                f"输入:\n{json.dumps(prompt_payload, ensure_ascii=False)}"
            )
            state = AgentState(
                user_id="predict-explainer",
                question=f"predict explain {target_code}",
                stock_codes=[target_code],
                trace_id=trace_id,
                mode="predict_explain",
            )
            try:
                raw = self.llm_gateway.generate(state, prompt_text)
                parsed = self._deep_safe_json_loads(raw)
                summary = str(parsed.get("summary", "")).strip()
                drivers = [str(x).strip() for x in list(parsed.get("drivers", []) or []) if str(x).strip()][:4]
                risks = [str(x).strip() for x in list(parsed.get("risks", []) or []) if str(x).strip()][:4]
                actions = [str(x).strip() for x in list(parsed.get("actions", []) or []) if str(x).strip()][:4]
                if summary:
                    base_payload["summary"] = summary[:260]
                if drivers:
                    base_payload["drivers"] = drivers
                if risks:
                    base_payload["risks"] = risks
                if actions:
                    base_payload["actions"] = actions
                llm_used = True
                provider = str(state.analysis.get("llm_provider", ""))
                model = str(state.analysis.get("llm_model", ""))
            except Exception as ex:  # noqa: BLE001
                degraded_reason = f"llm_provider_failed:{str(ex)[:160]}"

        return {
            "run_id": run_id,
            "trace_id": trace_id,
            "stock_code": target_code,
            "horizon": horizon_name,
            "signal": signal,
            "risk_tier": risk_tier,
            "expected_excess_return": expected_excess,
            "up_probability": up_prob,
            "summary": str(base_payload.get("summary", "")),
            "drivers": list(base_payload.get("drivers", [])),
            "risks": list(base_payload.get("risks", [])),
            "actions": list(base_payload.get("actions", [])),
            "llm_used": llm_used,
            "provider": provider,
            "model": model,
            "degraded_reason": degraded_reason,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def predict_get(self, run_id: str) -> dict[str, Any]:
        """Get prediction task details."""
        return self.prediction.get_prediction(run_id)

    def predict_eval_latest(self) -> dict[str, Any]:
        """Get latest prediction evaluation summary."""
        return self.prediction.eval_latest()

    def predict_self_test(self, *, stock_code: str = "SH600000", question: str = "") -> dict[str, Any]:
        """Run one predict chain self-test and return traceable diagnostics."""
        code = str(stock_code or "SH600000").strip().upper().replace(".", "")
        probe_question = str(question or "").strip() or f"predict self-test for {code}"

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

        started = time.perf_counter()
        run_payload = {
            "stock_codes": [code],
            "horizons": ["5d", "20d"],
            "question": probe_question,
        }
        run = self.predict_run(run_payload)
        latency_ms = int((time.perf_counter() - started) * 1000)

        run_id = str(run.get("run_id", "")).strip()
        explain_payload: dict[str, Any] = {}
        explain_error = ""
        if run_id:
            try:
                explain_payload = self.predict_explain({"run_id": run_id, "stock_code": code, "horizon": "20d"})
            except Exception as ex:  # noqa: BLE001
                explain_error = str(ex)[:220]

        trace_id = str(run.get("multi_role_trace_id", run.get("trace_id", ""))).strip()
        trace_rows = self.multi_role_trace_events(trace_id, limit=120).get("events", []) if trace_id else []
        quality_gate = dict(run.get("quality_gate", {}) or {})
        return {
            "ok": True,
            "stock_code": code,
            "question": probe_question,
            "latency_ms": latency_ms,
            "run_id": run_id,
            "quality_gate": quality_gate,
            "quality_gate_summary": dict(run.get("quality_gate_summary", {}) or {}),
            "degrade_reasons": list(run.get("degrade_reasons", []) or []),
            "multi_role_trace_id": trace_id,
            "multi_role_enabled": bool(run.get("multi_role_enabled", False)),
            "consensus_signal": str(run.get("consensus_signal", "hold")),
            "consensus_confidence": float(run.get("consensus_confidence", 0.5) or 0.5),
            "conflict_sources": list(run.get("conflict_sources", []) or []),
            "judge_summary": str(run.get("judge_summary", "")),
            "trace_events": trace_rows,
            "explain_status": "ok" if explain_payload else "failed",
            "explain_error": explain_error,
            "explain_preview": {
                "llm_used": bool(explain_payload.get("llm_used", False)) if explain_payload else False,
                "degraded_reason": str(explain_payload.get("degraded_reason", "")) if explain_payload else "",
                "summary": str(explain_payload.get("summary", "")) if explain_payload else "",
            },
        }

