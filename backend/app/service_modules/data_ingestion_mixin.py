from __future__ import annotations

from .shared import *

class DataIngestionMixin:
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

