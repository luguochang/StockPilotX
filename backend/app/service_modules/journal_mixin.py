from __future__ import annotations

from .shared import *

class JournalMixin:
    def _journal_normalize_items(self, values: Any, *, max_items: int = 6, max_len: int = 200) -> list[str]:
        if not isinstance(values, list):
            return []
        normalized: list[str] = []
        for raw in values:
            text = str(raw or "").strip()
            if not text:
                continue
            text = text[:max_len]
            if text in normalized:
                continue
            normalized.append(text)
            if len(normalized) >= max_items:
                break
        return normalized

    def _journal_build_ai_reflection_prompt(
        self,
        *,
        journal: dict[str, Any],
        reflections: list[dict[str, Any]],
        focus: str,
    ) -> str:
        """Build strict JSON prompt so downstream API can parse deterministic fields."""
        reflection_brief = []
        for item in reflections[:5]:
            reflection_brief.append(str(item.get("reflection_content", ""))[:120])
        return (
            "你是A股投资复盘助手。仅输出 JSON，不要输出 markdown。\n"
            "JSON schema:\n"
            "{\n"
            '  "summary": "一句话总结，<=120字",\n'
            '  "insights": ["洞察1", "洞察2"],\n'
            '  "lessons": ["改进行动1", "改进行动2"],\n'
            '  "confidence": 0.0\n'
            "}\n\n"
            f"日志类型: {journal.get('journal_type', '')}\n"
            f"股票: {journal.get('stock_code', '')}\n"
            f"决策方向: {journal.get('decision_type', '')}\n"
            f"日志标题: {journal.get('title', '')}\n"
            f"日志正文: {journal.get('content', '')}\n"
            f"历史复盘摘要: {json.dumps(reflection_brief, ensure_ascii=False)}\n"
            f"关注重点: {focus}\n"
            "要求:\n"
            "1) 聚焦执行偏差、证据缺口、下次可操作动作，不给买卖指令。\n"
            "2) 每条洞察/改进动作控制在 60 字内。\n"
            "3) confidence range must be within [0,1]."
        )

    def _journal_validate_ai_reflection_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("ai reflection payload must be object")
        summary = str(payload.get("summary", "")).strip()
        if not summary:
            raise ValueError("ai reflection summary is empty")
        insights = self._journal_normalize_items(payload.get("insights", []))
        lessons = self._journal_normalize_items(payload.get("lessons", []))
        confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.0) or 0.0)))
        if not insights:
            insights = ["Review evidence completeness and trigger conditions before making decisions."]
        if not lessons:
            lessons = ["Define invalidation conditions and re-check schedule before execution."]
        return {
            "summary": summary[:600],
            "insights": insights,
            "lessons": lessons,
            "confidence": confidence,
        }

    def _journal_ai_reflection_fallback(self, *, journal: dict[str, Any]) -> dict[str, Any]:
        stock = str(journal.get("stock_code", "")).strip()
        decision = str(journal.get("decision_type", "")).strip() or "unknown"
        return {
            "summary": f"Journal for {stock or 'this ticker'} ({decision}) recorded. Re-check trigger and invalidation conditions.",
            "insights": [
                "Break decision basis into verifiable indicators instead of narrative-first reasoning.",
                "Add position constraints and stop-loss thresholds to reduce execution bias.",
            ],
            "lessons": [
                "Compare expected vs actual outcomes and update the next decision template.",
                "Capture failed samples and boundary conditions to prioritize high-impact fixes.",
            ],
            "confidence": 0.46,
        }

    @staticmethod
    def _journal_default_title(journal_type: str, stock_code: str) -> str:
        """Create a readable title when frontend only sends minimal fields."""
        label_map = {
            "decision": "交易决策",
            "reflection": "交易复盘",
            "learning": "学习记录",
        }
        jt = str(journal_type or "decision").strip().lower()
        label = label_map.get(jt, "投资日志")
        code = str(stock_code or "").strip().upper() or "UNKNOWN"
        return f"{label} {code}"

    def _journal_default_content(self, payload: dict[str, Any]) -> str:
        """Generate fallback content so users can submit quickly without verbose input."""
        stock_code = str(payload.get("stock_code", "")).strip().upper() or "UNKNOWN"
        journal_type = str(payload.get("journal_type", "decision")).strip().lower() or "decision"
        decision_type = str(payload.get("decision_type", "hold")).strip().lower() or "hold"
        thesis = str(payload.get("thesis", "")).strip()
        if not thesis:
            thesis = (
                f"{stock_code} 当前采用 {journal_type}/{decision_type} 模板记录。"
                "后续将结合数据覆盖提升观点完整性。"
            )
        return "\n".join(
            [
                f"模板类型: {journal_type}",
                f"核心观点: {thesis}",
                "触发条件: （系统默认）价格与成交量共振，且关键风险未恶化。",
                "失效条件: （系统默认）关键风险事件落地偏负面或趋势结构被破坏。",
                "执行计划: （系统默认）分批验证，不单次重仓。",
            ]
        )

    # Investment Journal: keep orchestration in service layer so API payload remains thin.

    def journal_create(self, token: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = dict(payload or {})
        journal_type = str(body.get("journal_type", "decision")).strip().lower() or "decision"
        stock_code = str(body.get("stock_code", "")).strip().upper()
        title = str(body.get("title", "")).strip()
        content = str(body.get("content", "")).strip()
        if not title:
            title = self._journal_default_title(journal_type, stock_code)
        if not content:
            # Support fast-create mode where frontend only sends template + code.
            content = self._journal_default_content(body)
        tags = body.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        if not tags:
            tags = [journal_type, "auto_generated"]

        return self.web.journal_create(
            token,
            journal_type=journal_type,
            title=title,
            content=content,
            stock_code=stock_code,
            decision_type=str(body.get("decision_type", "")),
            related_research_id=str(body.get("related_research_id", "")),
            related_portfolio_id=body.get("related_portfolio_id"),
            tags=tags,
            sentiment=str(body.get("sentiment", "")),
            source_type=str(body.get("source_type", "manual")),
            source_ref_id=str(body.get("source_ref_id", "")),
            status=str(body.get("status", "open")),
            review_due_at=str(body.get("review_due_at", "")),
            executed_as_planned=body.get("executed_as_planned", False),
            outcome_rating=str(body.get("outcome_rating", "")),
            outcome_note=str(body.get("outcome_note", "")),
            deviation_reason=str(body.get("deviation_reason", "")),
            closed_at=str(body.get("closed_at", "")),
        )

    def journal_create_quick(self, token: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = dict(payload or {})
        stock_code = str(body.get("stock_code", "")).strip().upper()
        if not stock_code:
            raise ValueError("stock_code is required")
        event_type = str(body.get("event_type", "watch")).strip().lower() or "watch"
        if event_type not in {"buy", "sell", "rebalance", "watch"}:
            raise ValueError("event_type must be one of buy/sell/rebalance/watch")
        review_days = max(1, min(365, int(body.get("review_days", 5) or 5)))
        thesis = str(body.get("thesis", "")).strip()
        custom_tags = body.get("tags", [])
        tags = [event_type, "quick_log", stock_code]
        if isinstance(custom_tags, list):
            tags.extend([str(x).strip() for x in custom_tags if str(x).strip()])
        tags = list(dict.fromkeys(tags))[:12]

        event_label = {
            "buy": "买入",
            "sell": "卖出",
            "rebalance": "调仓",
            "watch": "观察",
        }.get(event_type, event_type)
        decision_type = "buy" if event_type == "buy" else "reduce" if event_type == "sell" else "hold"
        due_at = (datetime.now() + timedelta(days=review_days)).strftime("%Y-%m-%d %H:%M:%S")
        content = "\n".join(
            [
                f"事件类型: {event_type}",
                f"标的: {stock_code}",
                f"观点: {thesis or '（可选，后补）'}",
                "执行记录: （后续补充）",
                "偏差原因: （后续补充）",
                "改进动作: （后续补充）",
            ]
        )
        return self.web.journal_create(
            token,
            journal_type="decision",
            title=f"{event_label}记录 {stock_code}",
            content=content,
            stock_code=stock_code,
            decision_type=decision_type,
            tags=tags,
            sentiment="neutral",
            source_type="manual",
            source_ref_id="",
            status="open",
            review_due_at=due_at,
        )

    def journal_create_from_transaction(self, token: str, payload: dict[str, Any]) -> dict[str, Any]:
        portfolio_id = int(payload.get("portfolio_id", 0) or 0)
        transaction_id = int(payload.get("transaction_id", 0) or 0)
        review_days = int(payload.get("review_days", 5) or 5)
        if portfolio_id <= 0:
            raise ValueError("portfolio_id must be > 0")
        if transaction_id <= 0:
            raise ValueError("transaction_id must be > 0")
        return self.web.journal_create_from_transaction(
            token,
            portfolio_id=portfolio_id,
            transaction_id=transaction_id,
            review_days=review_days,
        )

    def journal_review_queue(
        self,
        token: str,
        *,
        status: str = "",
        stock_code: str = "",
        limit: int = 60,
    ) -> list[dict[str, Any]]:
        return self.web.journal_review_queue(
            token,
            status=status,
            stock_code=stock_code,
            limit=limit,
        )

    def journal_outcome_update(self, token: str, journal_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        return self.web.journal_outcome_update(
            token,
            journal_id=journal_id,
            executed_as_planned=payload.get("executed_as_planned"),
            outcome_rating=payload.get("outcome_rating"),
            outcome_note=payload.get("outcome_note"),
            deviation_reason=payload.get("deviation_reason"),
            close=payload.get("close"),
        )

    def journal_execution_board(self, token: str, *, window_days: int = 30) -> dict[str, Any]:
        return self.web.journal_execution_board(token, window_days=window_days)

    def journal_list(
        self,
        token: str,
        *,
        journal_type: str = "",
        stock_code: str = "",
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        return self.web.journal_list(
            token,
            journal_type=journal_type,
            stock_code=stock_code,
            limit=limit,
            offset=offset,
        )

    def journal_reflection_add(self, token: str, journal_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        return self.web.journal_reflection_add(
            token,
            journal_id=journal_id,
            reflection_content=str(payload.get("reflection_content", "")),
            ai_insights=str(payload.get("ai_insights", "")),
            lessons_learned=str(payload.get("lessons_learned", "")),
        )

    def journal_reflection_list(self, token: str, journal_id: int, *, limit: int = 50) -> list[dict[str, Any]]:
        return self.web.journal_reflection_list(token, journal_id=journal_id, limit=limit)

    def journal_ai_reflection_get(self, token: str, journal_id: int) -> dict[str, Any]:
        return self.web.journal_ai_reflection_get(token, journal_id=journal_id)

    def journal_ai_reflection_generate(self, token: str, journal_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        journal = self.web.journal_get(token, journal_id=journal_id)
        reflections = self.web.journal_reflection_list(token, journal_id=journal_id, limit=8)
        focus = str(payload.get("focus", "")).strip()[:200]
        trace_id = self.traces.new_trace()

        generated = self._journal_ai_reflection_fallback(journal=journal)
        status = "fallback"
        error_code = ""
        error_message = ""
        provider = ""
        model = ""
        started_at = time.perf_counter()

        if self.settings.llm_external_enabled and self.llm_gateway.providers:
            state = AgentState(
                user_id=str(journal.get("user_id", "journal_user")),
                question=f"journal-ai-reflection:{journal_id}",
                stock_codes=[str(journal.get("stock_code", "")).strip().upper()] if str(journal.get("stock_code", "")).strip() else [],
                trace_id=trace_id,
            )
            prompt = self._journal_build_ai_reflection_prompt(
                journal=journal,
                reflections=reflections,
                focus=focus or "Please emphasize execution bias, evidence gaps, and actionable improvements.",
            )
            try:
                raw = self.llm_gateway.generate(state, prompt)
                parsed = self._deep_safe_json_loads(raw)
                generated = self._journal_validate_ai_reflection_payload(parsed)
                status = "ready"
                provider = str(state.analysis.get("llm_provider", ""))
                model = str(state.analysis.get("llm_model", ""))
            except Exception as ex:  # noqa: BLE001
                status = "fallback"
                error_code = "journal_ai_generation_failed"
                error_message = str(ex)[:380]

        result = self.web.journal_ai_reflection_upsert(
            token,
            journal_id=journal_id,
            status=status,
            summary=str(generated.get("summary", "")),
            insights=generated.get("insights", []),
            lessons=generated.get("lessons", []),
            confidence=float(generated.get("confidence", 0.0) or 0.0),
            provider=provider,
            model=model,
            trace_id=trace_id,
            error_code=error_code,
            error_message=error_message,
        )
        # 杈撳嚭鐢熸垚鑰楁椂锛屾柟渚垮悗缁帓鏌モ€淎I澶嶇洏鎱?澶辫触鈥濈殑鍏蜂綋閾捐矾銆?
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        result["latency_ms"] = latency_ms
        # 姣忔鐢熸垚閮借褰曡川閲忔棩蹇楋紝渚夸簬杩愮淮缁熻 fallback/failed 姣斾緥涓庡欢杩熷垎甯冦€?
        try:
            _ = self.web.journal_ai_generation_log_add(
                token,
                journal_id=journal_id,
                status=status,
                provider=provider,
                model=model,
                trace_id=trace_id,
                error_code=error_code,
                error_message=error_message,
                latency_ms=latency_ms,
            )
        except Exception:
            pass
        return result

    def _journal_counter_breakdown(self, counter: Counter[str], *, total: int, top_n: int) -> list[dict[str, Any]]:
        """Convert Counter into deterministic [{key,count,ratio}] payload for UI/API."""
        items = sorted(counter.items(), key=lambda x: (-int(x[1]), str(x[0])))
        output: list[dict[str, Any]] = []
        for key, count in items[: max(1, int(top_n))]:
            ratio = float(count) / float(total) if total > 0 else 0.0
            output.append({"key": str(key), "count": int(count), "ratio": round(ratio, 4)})
        return output

    def _journal_extract_keywords(self, rows: list[dict[str, Any]], *, top_n: int = 12) -> list[dict[str, Any]]:
        """Extract coarse keywords from title/content/tags for quick journal topic profiling."""
        stopwords = {
            "以及",
            "因为",
            "所以",
            "我们",
            "他们",
            "如果",
            "但是",
            "然后",
            "这个",
            "那个",
            "需要",
            "已经",
            "当前",
            "计划",
            "复盘",
            "日志",
            "记录",
            "分析",
            "执行",
            "策略",
            "stock",
            "journal",
            "decision",
            "reflection",
            "learning",
        }
        pattern = re.compile(r"[A-Za-z][A-Za-z0-9_+-]{2,24}|[\u4e00-\u9fff]{2,8}")
        counter: Counter[str] = Counter()
        for row in rows:
            tags = row.get("tags", [])
            joined_tags = " ".join(str(x) for x in tags if str(x).strip()) if isinstance(tags, list) else ""
            corpus = " ".join(
                [
                    str(row.get("title", "")),
                    str(row.get("content", "")),
                    joined_tags,
                ]
            )
            for token in pattern.findall(corpus):
                term = str(token).strip().lower()
                if not term:
                    continue
                if term in stopwords:
                    continue
                if re.fullmatch(r"\d+", term):
                    continue
                counter[term] += 1
        items = sorted(counter.items(), key=lambda x: (-int(x[1]), str(x[0])))
        return [{"keyword": str(term), "count": int(count)} for term, count in items[: max(1, int(top_n))]]

    def journal_insights(
        self,
        token: str,
        *,
        window_days: int = 90,
        limit: int = 400,
        timeline_days: int = 30,
    ) -> dict[str, Any]:
        # 鑱氬悎灞傜粺涓€杈撳嚭涓氬姟娲炲療锛屽墠绔棤闇€鍐嶆嫾鎺ュ涓帴鍙ｃ€?
        safe_window_days = max(7, min(3650, int(window_days)))
        safe_limit = max(20, min(2000, int(limit)))
        safe_timeline_days = max(7, min(safe_window_days, int(timeline_days)))

        rows = self.web.journal_insights_rows(token, days=safe_window_days, limit=safe_limit)
        timeline_rows = self.web.journal_insights_timeline(token, days=safe_timeline_days)

        total_journals = len(rows)
        type_counter: Counter[str] = Counter()
        decision_counter: Counter[str] = Counter()
        stock_counter: Counter[str] = Counter()

        reflection_covered = 0
        ai_reflection_covered = 0
        total_reflection_records = 0
        for row in rows:
            journal_type = str(row.get("journal_type", "")).strip() or "unknown"
            decision_type = str(row.get("decision_type", "")).strip() or "none"
            stock_code = str(row.get("stock_code", "")).strip().upper() or "UNASSIGNED"
            reflection_count = max(0, int(row.get("reflection_count", 0) or 0))
            has_ai_reflection = bool(row.get("has_ai_reflection", False))

            type_counter[journal_type] += 1
            decision_counter[decision_type] += 1
            stock_counter[stock_code] += 1
            total_reflection_records += reflection_count
            if reflection_count > 0:
                reflection_covered += 1
            if has_ai_reflection:
                ai_reflection_covered += 1

        avg_reflections_per_journal = float(total_reflection_records) / float(total_journals) if total_journals else 0.0
        reflection_coverage_rate = float(reflection_covered) / float(total_journals) if total_journals else 0.0
        ai_reflection_coverage_rate = float(ai_reflection_covered) / float(total_journals) if total_journals else 0.0

        return {
            "status": "ok",
            "window_days": safe_window_days,
            "timeline_days": safe_timeline_days,
            "total_journals": total_journals,
            "type_distribution": self._journal_counter_breakdown(type_counter, total=total_journals, top_n=8),
            "decision_distribution": self._journal_counter_breakdown(decision_counter, total=total_journals, top_n=8),
            "stock_activity": self._journal_counter_breakdown(stock_counter, total=total_journals, top_n=10),
            "reflection_coverage": {
                "with_reflection": int(reflection_covered),
                "with_ai_reflection": int(ai_reflection_covered),
                "reflection_coverage_rate": round(reflection_coverage_rate, 4),
                "ai_reflection_coverage_rate": round(ai_reflection_coverage_rate, 4),
                "total_reflection_records": int(total_reflection_records),
                "avg_reflections_per_journal": round(avg_reflections_per_journal, 4),
            },
            "keyword_profile": self._journal_extract_keywords(rows, top_n=12),
            "timeline": timeline_rows,
        }

