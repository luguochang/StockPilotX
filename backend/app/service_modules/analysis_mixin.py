from __future__ import annotations

from .shared import *

class AnalysisMixin:
    def factor_snapshot(self, stock_code: str) -> dict[str, Any]:
        """Get factor snapshot for one stock."""
        return self.prediction.get_factor_snapshot(stock_code)

    def multi_role_trace_events(self, trace_id: str, *, limit: int = 120) -> dict[str, Any]:
        """Return predict multi-role trace rows. Reuses common trace store formatter."""
        return self.deep_think_trace_events(trace_id, limit=limit)

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

        positive_keywords = ("增长", "增持", "中标", "回购", "改善", "突破", "上调", "buy", "outperform")
        negative_keywords = ("下滑", "减持", "诉讼", "处罚", "亏损", "风险", "下调", "sell", "underperform")

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
                f"主力={float(fund_row.get('main_inflow', 0.0) or 0.0):.2f}, "
                f"大单={float(fund_row.get('large_inflow', 0.0) or 0.0):.2f}, "
                f"小单={float(fund_row.get('small_inflow', 0.0) or 0.0):.2f}"
            )
            add_evidence(
                kind="fund",
                title="资金流向",
                summary=fund_summary,
                source_id=str(fund_row.get("source_id", "")),
                source_url=str(fund_row.get("source_url", "")),
                event_time=str(fund_row.get("ts", fund_row.get("trade_date", ""))),
                reliability_score=float(fund_row.get("reliability_score", 0.0) or 0.0),
                retrieval_track="fund_flow",
            )
            classify_row("资金流向", fund_summary, fund_row, kind="fund", default_signal="neutral")

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
                    "title": f"{str(row.get('metric_name', '宏观指标'))} {str(row.get('metric_value', ''))}".strip(),
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

