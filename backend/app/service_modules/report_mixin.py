from __future__ import annotations

from .shared import *

class ReportMixin:
    def _build_report_committee_notes(
        self,
        *,
        final_decision: dict[str, Any],
        quality_gate: dict[str, Any],
        intel: dict[str, Any],
        quality_reasons: list[str],
        analysis_nodes: list[dict[str, Any]] | None = None,
    ) -> dict[str, str]:
        """Build committee-style summary for product display."""
        signal = str(final_decision.get("signal", "hold"))
        confidence = self._safe_float(final_decision.get("confidence", 0.5), default=0.5)
        risk_level = str(intel.get("risk_level", "medium") or "medium")
        catalysts = len(list(intel.get("key_catalysts", []) or []))
        risk_watch = len(list(intel.get("risk_watch", []) or []))
        node_by_id = {
            str(row.get("node_id", "")).strip().lower(): row
            for row in list(analysis_nodes or [])
            if isinstance(row, dict) and str(row.get("node_id", "")).strip()
        }
        research_node = dict(node_by_id.get("research_summarizer", {}) or {})
        risk_node = dict(node_by_id.get("risk_arbiter", {}) or {})
        research_note = (
            f"研究汇总：当前倾向 `{signal}`（置信度 {confidence:.2f}），"
            f"已识别催化 {catalysts} 项、风险观察 {risk_watch} 项。"
        )
        if str(research_node.get("summary", "")).strip():
            research_note = str(research_node.get("summary", "")).strip()
        risk_note = (
            f"风险仲裁：风险等级 `{risk_level}`，质量门控 `{quality_gate.get('status', 'unknown')}`，"
            f"质量原因 {', '.join(quality_reasons) if quality_reasons else 'none'}。"
        )
        if str(risk_node.get("summary", "")).strip():
            risk_note = str(risk_node.get("summary", "")).strip()
        return {"research_note": research_note, "risk_note": risk_note}

    def _build_report_research_summarizer_node(
        self,
        *,
        code: str,
        query_result: dict[str, Any],
        overview: dict[str, Any],
        intel: dict[str, Any],
        final_decision: dict[str, Any],
        quality_gate: dict[str, Any],
    ) -> dict[str, Any]:
        """Build nodeized research synthesizer output for downstream display/bridging."""
        citations = list(query_result.get("citations", []) or [])
        citation_refs = [str(row.get("source_id", "")).strip() for row in citations if str(row.get("source_id", "")).strip()]
        news_rows = list(overview.get("news", []) or [])
        catalysts = list(intel.get("key_catalysts", []) or [])
        quality_reasons = [str(x).strip() for x in list(quality_gate.get("reasons", []) or []) if str(x).strip()]
        summary_parts = [
            f"研究汇总器：{code} 当前信号 `{self._normalize_report_signal(str(final_decision.get('signal', 'hold')))}`",
            f"证据引用 {len(citation_refs)} 条，新闻样本 {len(news_rows)} 条，催化事件 {len(catalysts)} 条。",
        ]
        if quality_reasons:
            summary_parts.append(f"质量门控提示：{','.join(quality_reasons[:3])}")
        highlights: list[str] = []
        if str(query_result.get("answer", "")).strip():
            highlights.append(str(query_result.get("answer", "")).strip()[:220])
        for row in catalysts[:2]:
            item = str(row.get("summary", row.get("title", ""))).strip()
            if item:
                highlights.append(item[:180])
        for row in news_rows[-2:]:
            title = str(row.get("title", "")).strip()
            if title:
                highlights.append(title[:160])
        return {
            "node_id": "research_summarizer",
            "title": "研究汇总器",
            "status": "ready" if citation_refs else "degraded",
            "signal": self._normalize_report_signal(str(final_decision.get("signal", "hold"))),
            "confidence": round(
                max(
                    0.2,
                    min(
                        self._safe_float(final_decision.get("confidence", 0.5), default=0.5),
                        self._safe_float(quality_gate.get("score", 0.5), default=0.5),
                    ),
                ),
                4,
            ),
            "summary": " ".join(summary_parts)[:320],
            "highlights": highlights[:6],
            "evidence_refs": citation_refs[:8],
            "coverage": {
                "citation_count": int(len(citation_refs)),
                "news_count": int(len(news_rows)),
                "catalyst_count": int(len(catalysts)),
            },
            "degrade_reason": quality_reasons,
        }

    def _build_report_risk_arbiter_node(
        self,
        *,
        intel: dict[str, Any],
        quality_gate: dict[str, Any],
        final_decision: dict[str, Any],
    ) -> dict[str, Any]:
        """Build nodeized risk arbiter output with actionable guardrails."""
        risk_level = str(intel.get("risk_level", "medium") or "medium").strip().lower() or "medium"
        risk_watch = list(intel.get("risk_watch", []) or [])
        quality_status = str(quality_gate.get("status", "unknown") or "unknown")
        quality_score = self._safe_float(quality_gate.get("score", 0.0), default=0.0)
        quality_reasons = [str(x).strip() for x in list(quality_gate.get("reasons", []) or []) if str(x).strip()]
        signal = self._normalize_report_signal(str(final_decision.get("signal", "hold")))

        guardrails: list[str] = [
            "若单日波动显著放大，降低新增仓位节奏。",
            "遇到关键风险事件落地，先执行风险限额再更新观点。",
        ]
        if signal == "buy":
            guardrails.insert(0, "采用分批建仓，避免单次重仓暴露。")
        if signal == "reduce":
            guardrails.insert(0, "优先削减高波动仓位并保留观察仓。")
        if quality_status != "pass":
            guardrails.append("当前处于质量降级，补齐缺失数据前不放大风险敞口。")

        risk_signal = "warning" if risk_level in {"high", "very_high"} or quality_status != "pass" else "normal"
        veto = bool(risk_signal == "warning" and (signal == "buy" and quality_status != "pass"))
        summary = (
            f"风险仲裁器：风险等级 `{risk_level}`，风险观察 {len(risk_watch)} 项，"
            f"质量门控 `{quality_status}`（{quality_score:.2f}）。"
        )
        if quality_reasons:
            summary += f" 主要降级原因：{','.join(quality_reasons[:3])}。"

        return {
            "node_id": "risk_arbiter",
            "title": "风险仲裁器",
            "status": "ready",
            "signal": risk_signal,
            "confidence": round(max(0.2, min(1.0, max(quality_score, 0.35))), 4),
            "summary": summary[:320],
            "highlights": [
                str(row.get("summary", row.get("title", ""))).strip()[:180]
                for row in risk_watch[:3]
                if str(row.get("summary", row.get("title", ""))).strip()
            ],
            "coverage": {
                "risk_watch_count": int(len(risk_watch)),
                "quality_status": quality_status,
            },
            "degrade_reason": quality_reasons,
            "guardrails": guardrails[:6],
            "veto": veto,
        }

    def _normalize_report_analysis_nodes(self, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize analysis node payload to stable schema."""
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in nodes:
            if not isinstance(row, dict):
                continue
            node_id = str(row.get("node_id", "")).strip().lower()
            if not node_id or node_id in seen:
                continue
            seen.add(node_id)
            highlights = [self._sanitize_report_text(str(x).strip()) for x in list(row.get("highlights", []) or []) if str(x).strip()]
            evidence_refs = [str(x).strip() for x in list(row.get("evidence_refs", []) or []) if str(x).strip()]
            degrade_reason = [str(x).strip() for x in list(row.get("degrade_reason", []) or []) if str(x).strip()]
            guardrails = [self._sanitize_report_text(str(x).strip()) for x in list(row.get("guardrails", []) or []) if str(x).strip()]
            normalized.append(
                {
                    "node_id": node_id,
                    "title": self._sanitize_report_text(str(row.get("title", node_id)).strip() or node_id),
                    "status": str(row.get("status", "ready")).strip().lower() or "ready",
                    "signal": self._normalize_report_signal(str(row.get("signal", "hold"))),
                    "confidence": round(max(0.0, min(1.0, self._safe_float(row.get("confidence", 0.5), default=0.5))), 4),
                    "summary": self._sanitize_report_text(str(row.get("summary", "")).strip()),
                    "highlights": highlights[:8],
                    "evidence_refs": evidence_refs[:12],
                    "coverage": dict(row.get("coverage", {}) or {}),
                    "degrade_reason": degrade_reason,
                    "guardrails": guardrails[:8],
                    "veto": bool(row.get("veto", False)),
                }
            )
        return normalized

    def _build_report_quality_dashboard(
        self,
        *,
        report_modules: list[dict[str, Any]],
        quality_gate: dict[str, Any],
        final_decision: dict[str, Any],
        analysis_nodes: list[dict[str, Any]] | None = None,
        evidence_refs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build module-level quality board with coverage/evidence/consistency metrics."""
        modules = self._normalize_report_modules(report_modules)
        node_rows = self._normalize_report_analysis_nodes(list(analysis_nodes or []))
        evidence_rows = self._normalize_report_evidence_refs(list(evidence_refs or []))
        module_count = len(modules)
        module_scores = [self._safe_float(row.get("module_quality_score", 0.0), default=0.0) for row in modules]
        avg_module_score = sum(module_scores) / float(module_count) if module_count else 0.0
        min_module_score = min(module_scores) if module_scores else 0.0
        coverage_full = sum(1 for row in modules if str((row.get("coverage", {}) or {}).get("status", "")).strip().lower() == "full")
        coverage_ratio = float(coverage_full) / float(module_count) if module_count else 0.0
        unique_evidence = {
            str(item).strip()
            for row in modules
            for item in list(row.get("evidence_refs", []) or [])
            if str(item).strip()
        }
        evidence_density = float(len(unique_evidence)) / float(module_count) if module_count else 0.0
        freshness_scores = [self._safe_float(row.get("freshness_score", 0.35), default=0.35) for row in evidence_rows]
        freshness_score = sum(freshness_scores) / float(len(freshness_scores)) if freshness_scores else 0.35
        stale_evidence_count = sum(1 for row in freshness_scores if row < 0.55)
        stale_evidence_ratio = float(stale_evidence_count) / float(len(freshness_scores)) if freshness_scores else 0.0
        decision_confidence = self._safe_float(final_decision.get("confidence", 0.5), default=0.5)
        consistency_score = max(0.0, min(1.0, 1.0 - abs(decision_confidence - avg_module_score)))
        node_veto = any(bool(row.get("veto", False)) for row in node_rows)
        quality_gate_score = self._safe_float(quality_gate.get("score", 0.0), default=0.0)

        overall_score = (
            avg_module_score * 0.34
            + coverage_ratio * 0.20
            + min(1.0, evidence_density / 2.0) * 0.14
            + consistency_score * 0.15
            + quality_gate_score * 0.09
            + freshness_score * 0.08
        )
        overall_score = max(0.0, min(1.0, overall_score))
        if overall_score >= 0.75 and not node_veto:
            status = "pass"
        elif overall_score >= 0.5:
            status = "watch"
        else:
            status = "degraded"
        if node_veto and status != "pass":
            status = "degraded"
        if overall_score < 0.35:
            status = "critical"
        low_quality_modules = [str(row.get("module_id", "")) for row in modules if self._safe_float(row.get("module_quality_score", 0.0), default=0.0) < 0.45]
        reasons: list[str] = [str(x).strip() for x in list(quality_gate.get("reasons", []) or []) if str(x).strip()]
        if node_veto:
            reasons.append("risk_arbiter_veto")
        if freshness_score < 0.45:
            reasons.append("evidence_stale")
        reasons = list(dict.fromkeys(reasons))
        return {
            "status": status,
            "overall_score": round(overall_score, 4),
            "module_count": int(module_count),
            "avg_module_quality": round(avg_module_score, 4),
            "min_module_quality": round(min_module_score, 4),
            "coverage_ratio": round(coverage_ratio, 4),
            "evidence_ref_count": int(len(unique_evidence)),
            "evidence_density": round(evidence_density, 4),
            "evidence_freshness_score": round(freshness_score, 4),
            "stale_evidence_ratio": round(stale_evidence_ratio, 4),
            "consistency_score": round(consistency_score, 4),
            "low_quality_modules": low_quality_modules[:8],
            "reasons": reasons,
            "node_veto": node_veto,
        }

    def _render_report_module_markdown(self, report_payload: dict[str, Any]) -> str:
        """Render report payload into moduleized markdown for export."""
        report_id = str(report_payload.get("report_id", "")).strip()
        stock_code = str(report_payload.get("stock_code", "")).strip().upper() or "UNKNOWN"
        report_type = str(report_payload.get("report_type", "")).strip() or "report"
        final_decision = dict(report_payload.get("final_decision", {}) or {})
        committee = dict(report_payload.get("committee", {}) or {})
        quality_dashboard = dict(report_payload.get("quality_dashboard", {}) or {})
        modules = self._normalize_report_modules(list(report_payload.get("report_modules", []) or []))
        lines = [
            f"# {stock_code} 模块化报告导出",
            "",
            f"- report_id: {report_id or 'n/a'}",
            f"- report_type: {report_type}",
            f"- signal: {self._normalize_report_signal(str(final_decision.get('signal', 'hold')))}",
            f"- confidence: {self._safe_float(final_decision.get('confidence', 0.0), default=0.0):.2f}",
            "",
            "## 委员会纪要",
            f"- 研究汇总: {self._sanitize_report_text(str(committee.get('research_note', '')).strip()) or 'n/a'}",
            f"- 风险仲裁: {self._sanitize_report_text(str(committee.get('risk_note', '')).strip()) or 'n/a'}",
            "",
            "## 质量看板",
            f"- status: {str(quality_dashboard.get('status', 'unknown'))}",
            f"- overall_score: {self._safe_float(quality_dashboard.get('overall_score', 0.0), default=0.0):.2f}",
            f"- coverage_ratio: {self._safe_float(quality_dashboard.get('coverage_ratio', 0.0), default=0.0):.2f}",
            f"- consistency_score: {self._safe_float(quality_dashboard.get('consistency_score', 0.0), default=0.0):.2f}",
            f"- low_quality_modules: {','.join(list(quality_dashboard.get('low_quality_modules', []) or [])) or 'none'}",
            "",
        ]
        for module in modules:
            module_id = str(module.get("module_id", "")).strip() or "module"
            title = self._sanitize_report_text(str(module.get("title", module_id)).strip() or module_id)
            content = self._sanitize_report_text(str(module.get("content", "")).strip())
            coverage = dict(module.get("coverage", {}) or {})
            lines.extend(
                [
                    f"## {title} ({module_id})",
                    f"- confidence: {self._safe_float(module.get('confidence', 0.0), default=0.0):.2f}",
                    f"- module_quality_score: {self._safe_float(module.get('module_quality_score', 0.0), default=0.0):.2f}",
                    f"- coverage: {str(coverage.get('status', 'unknown'))}/{int(coverage.get('data_points', 0) or 0)}",
                    f"- degrade_reason: {','.join(list(module.get('degrade_reason', []) or [])) or 'none'}",
                    "",
                    content or "该模块暂无可用内容。",
                    "",
                ]
            )
        return "\n".join(lines).strip() + "\n"

    def _build_fallback_report_modules(
        self,
        *,
        code: str,
        query_result: dict[str, Any],
        overview: dict[str, Any],
        predict_snapshot: dict[str, Any],
        intel: dict[str, Any],
        report_input_pack: dict[str, Any],
        quality_gate: dict[str, Any],
        quality_reasons: list[str],
        final_decision: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Build deterministic moduleized report payload used as baseline."""
        trend = dict(overview.get("trend", {}) or {})
        financial = dict(overview.get("financial", {}) or {})
        news_rows = list(overview.get("news", []) or [])
        risk_watch = list(intel.get("risk_watch", []) or [])
        scenarios = list(intel.get("scenario_matrix", []) or [])
        citations = list(query_result.get("citations", []) or [])
        citation_ids = [str(x.get("source_id", "")) for x in citations[:8] if str(x.get("source_id", "")).strip()]
        top_news = [str(x.get("title", "")).strip() for x in news_rows[-3:] if str(x.get("title", "")).strip()]
        bull_points = [
            str(x.get("summary", x.get("title", ""))).strip()
            for x in list(intel.get("key_catalysts", []) or [])[:4]
            if str(x.get("summary", x.get("title", ""))).strip()
        ]
        bear_points = [
            str(x.get("summary", x.get("title", ""))).strip()
            for x in list(intel.get("risk_watch", []) or [])[:4]
            if str(x.get("summary", x.get("title", ""))).strip()
        ]
        risk_level = str(intel.get("risk_level", "medium")).strip().lower() or "medium"
        target_price = dict(final_decision.get("target_price", {}) or {})
        position_hint = str(final_decision.get("position_sizing_hint", "")).strip()

        modules = [
            {
                "module_id": "executive_summary",
                "title": "执行摘要",
                "content": str(query_result.get("answer", "")).strip()[:1600],
                "evidence_refs": citation_ids,
                "confidence": self._safe_float(final_decision.get("confidence", 0.5), default=0.5),
                "coverage": {"status": "partial" if quality_reasons else "full", "data_points": int(len(citations))},
                "degrade_reason": list(quality_reasons),
            },
            {
                "module_id": "market_technical",
                "title": "行情与技术",
                "content": (
                    f"MA20={self._safe_float(trend.get('ma20', 0.0)):.3f}, "
                    f"MA60={self._safe_float(trend.get('ma60', 0.0)):.3f}, "
                    f"20日动量={self._safe_float(trend.get('momentum_20', 0.0)) * 100:.2f}%, "
                    f"20日波动={self._safe_float(trend.get('volatility_20', 0.0)) * 100:.2f}%, "
                    f"60日最大回撤={self._safe_float(trend.get('max_drawdown_60', 0.0)) * 100:.2f}%。"
                ),
                "evidence_refs": citation_ids[:4],
                "confidence": round(max(0.3, self._safe_float(quality_gate.get("score", 0.5), default=0.5) - 0.05), 4),
                "coverage": {"status": "partial" if "history_30d_insufficient" in quality_reasons else "full", "data_points": int(len(list(overview.get("history", []) or [])))},
                "degrade_reason": [x for x in quality_reasons if "history" in x],
            },
            {
                "module_id": "fundamental_valuation",
                "title": "财务与估值",
                "content": (
                    f"PE={self._safe_float(financial.get('pe', 0.0)):.3f}, "
                    f"PB={self._safe_float(financial.get('pb', 0.0)):.3f}, "
                    f"ROE={self._safe_float(financial.get('roe', 0.0)):.2f}%, "
                    f"ROA={self._safe_float(financial.get('roa', 0.0)):.2f}%, "
                    f"营收同比={self._safe_float(financial.get('revenue_yoy', 0.0)):.2f}%, "
                    f"净利同比={self._safe_float(financial.get('net_profit_yoy', 0.0)):.2f}%。"
                ),
                "evidence_refs": citation_ids[:4],
                "confidence": round(max(0.28, self._safe_float(quality_gate.get("score", 0.5), default=0.5) - 0.08), 4),
                "coverage": {"status": "partial" if "financial_missing" in quality_reasons else "full", "data_points": int(1 if financial else 0)},
                "degrade_reason": [x for x in quality_reasons if "financial" in x],
            },
            {
                "module_id": "news_event",
                "title": "新闻事件与催化",
                "content": (
                    "近期重点新闻："
                    + ("；".join(top_news) if top_news else "暂无高质量新闻命中。")
                ),
                "evidence_refs": citation_ids[:6],
                "confidence": round(max(0.25, self._safe_float(quality_gate.get("score", 0.5), default=0.5) - 0.06), 4),
                "coverage": {"status": "partial" if "news_insufficient" in quality_reasons else "full", "data_points": int(len(news_rows))},
                "degrade_reason": [x for x in quality_reasons if "news" in x],
            },
            {
                "module_id": "sentiment_flow",
                "title": "情绪与资金",
                "content": (
                    f"综合情报信号={str(intel.get('overall_signal', 'hold'))}, "
                    f"置信度={self._safe_float(intel.get('confidence', 0.0)):.2f}, "
                    f"催化数={len(list(intel.get('key_catalysts', []) or []))}, "
                    f"风险观察数={len(risk_watch)}。"
                ),
                "evidence_refs": citation_ids[:6],
                "confidence": round(max(0.25, self._safe_float(intel.get("confidence", 0.45), default=0.45)), 4),
                "coverage": {"status": "partial" if "fund_missing" in quality_reasons else "full", "data_points": int(len(list(intel.get("evidence", []) or [])))},
                "degrade_reason": [x for x in quality_reasons if "fund" in x],
            },
            {
                "module_id": "risk_matrix",
                "title": "风险矩阵",
                "content": json.dumps(
                    [
                        {"risk": str(x.get("title", "")), "signal": str(x.get("signal", "negative")), "detail": str(x.get("summary", ""))[:160]}
                        for x in risk_watch[:6]
                    ]
                    or [{"risk": "data_quality", "signal": "warning", "detail": ",".join(quality_reasons) or "none"}],
                    ensure_ascii=False,
                ),
                "evidence_refs": citation_ids[:6],
                "confidence": round(max(0.2, self._safe_float(quality_gate.get("score", 0.5), default=0.5) - 0.1), 4),
                "coverage": {"status": "partial" if quality_reasons else "full", "data_points": int(len(risk_watch))},
                "degrade_reason": list(quality_reasons),
            },
            {
                "module_id": "scenario_plan",
                "title": "情景分析",
                "content": json.dumps(scenarios[:6], ensure_ascii=False) if scenarios else "暂无可用情景矩阵，需补齐实时情报与历史样本。",
                "evidence_refs": citation_ids[:6],
                "confidence": round(max(0.22, self._safe_float(intel.get("confidence", 0.45), default=0.45) - 0.08), 4),
                "coverage": {"status": "partial" if not scenarios else "full", "data_points": int(len(scenarios))},
                "degrade_reason": ["scenario_missing"] if not scenarios else [],
            },
            {
                # TradingAgents-style role module: explicit bullish thesis long text.
                "module_id": "bull_case",
                "title": "多头论证（Bull Case）",
                "content": (
                    "核心逻辑：趋势与催化共振时，优先考虑增配。"
                    + (
                        "\n" + "\n".join(f"- {line}" for line in bull_points[:6])
                        if bull_points
                        else "\n- 当前未提取到明确多头催化，需补齐新闻/研报样本。"
                    )
                    + f"\n- 目标价参考：{json.dumps(target_price, ensure_ascii=False)}"
                ),
                "evidence_refs": citation_ids[:8],
                "confidence": round(max(0.2, self._safe_float(final_decision.get("confidence", 0.5), default=0.5) - 0.03), 4),
                "coverage": {"status": "partial" if quality_reasons else "full", "data_points": int(len(bull_points))},
                "degrade_reason": list(quality_reasons),
            },
            {
                # TradingAgents-style role module: explicit bearish thesis long text.
                "module_id": "bear_case",
                "title": "空头论证（Bear Case）",
                "content": (
                    "核心逻辑：若风险事件持续发酵，应优先保护回撤。"
                    + (
                        "\n" + "\n".join(f"- {line}" for line in bear_points[:6])
                        if bear_points
                        else "\n- 当前未提取到明确空头风险，仍需跟踪事件时间轴。"
                    )
                    + f"\n- 当前风险等级：{risk_level}"
                ),
                "evidence_refs": citation_ids[:8],
                "confidence": round(max(0.2, self._safe_float(final_decision.get("confidence", 0.5), default=0.5) - 0.05), 4),
                "coverage": {"status": "partial" if quality_reasons else "full", "data_points": int(len(bear_points))},
                "degrade_reason": list(quality_reasons),
            },
            {
                "module_id": "risky_case",
                "title": "激进执行论证（Risky Case）",
                "content": (
                    "核心逻辑：在高胜率窗口以更快节奏执行，但严格遵守单步上限。"
                    f"\n- 仓位提示：{position_hint or '未提供'}"
                    f"\n- 执行路径：{'; '.join(list(final_decision.get('execution_plan', []) or [])[:3]) or '待补充'}"
                ),
                "evidence_refs": citation_ids[:6],
                "confidence": round(max(0.2, self._safe_float(final_decision.get("confidence", 0.5), default=0.5) - 0.06), 4),
                "coverage": {"status": "partial" if quality_reasons else "full", "data_points": int(len(list(final_decision.get("execution_plan", []) or [])))},
                "degrade_reason": list(quality_reasons),
            },
            {
                "module_id": "safe_case",
                "title": "保守风控论证（Safe Case）",
                "content": (
                    "核心逻辑：优先控制回撤与证据不确定性，分批验证再行动。"
                    f"\n- 门控状态：{str(quality_gate.get('status', 'unknown'))}"
                    f"\n- 风险触发：{'; '.join(list(final_decision.get('invalidation_conditions', []) or [])[:4]) or '待补充'}"
                ),
                "evidence_refs": citation_ids[:6],
                "confidence": round(max(0.2, self._safe_float(final_decision.get("confidence", 0.5), default=0.5) - 0.04), 4),
                "coverage": {"status": "partial" if quality_reasons else "full", "data_points": int(len(list(final_decision.get("invalidation_conditions", []) or [])))},
                "degrade_reason": list(quality_reasons),
            },
            {
                "module_id": "execution_plan",
                "title": "执行策略",
                "content": "\n".join(f"- {line}" for line in list(final_decision.get("execution_plan", []) or [])),
                "evidence_refs": citation_ids[:4],
                "confidence": self._safe_float(final_decision.get("confidence", 0.5), default=0.5),
                "coverage": {"status": "full", "data_points": int(len(list(final_decision.get("execution_plan", []) or [])))},
                "degrade_reason": [],
            },
            {
                "module_id": "final_decision",
                "title": "综合决策",
                "content": str(final_decision.get("rationale", ""))[:1800],
                "evidence_refs": citation_ids[:8],
                "confidence": self._safe_float(final_decision.get("confidence", 0.5), default=0.5),
                "coverage": {"status": "partial" if quality_reasons else "full", "data_points": int(len(citation_ids))},
                "degrade_reason": list(quality_reasons),
            },
        ]

        return self._normalize_report_modules(modules)

    def _normalize_report_modules(self, modules: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize module payload to stable schema for frontend rendering."""
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in modules:
            if not isinstance(row, dict):
                continue
            module_id = str(row.get("module_id", "")).strip().lower()
            if not module_id or module_id in seen:
                continue
            seen.add(module_id)
            title = str(row.get("title", module_id)).strip() or module_id
            content = self._sanitize_report_text(str(row.get("content", "")).strip())
            if not content:
                content = "该模块暂无可用内容。"
            evidence_refs = [str(x).strip() for x in list(row.get("evidence_refs", []) or []) if str(x).strip()]
            coverage_raw = dict(row.get("coverage", {}) or {})
            coverage = {
                "status": str(coverage_raw.get("status", "partial")).strip().lower() or "partial",
                "data_points": int(coverage_raw.get("data_points", 0) or 0),
            }
            confidence = max(0.0, min(1.0, self._safe_float(row.get("confidence", 0.5), default=0.5)))
            degrade_reason = [str(x).strip() for x in list(row.get("degrade_reason", []) or []) if str(x).strip()]
            coverage_factor = {"full": 1.0, "partial": 0.78, "missing": 0.4}.get(str(coverage["status"]).lower(), 0.7)
            degrade_penalty = min(0.35, 0.08 * float(len(degrade_reason)))
            module_quality_score = max(0.0, min(1.0, confidence * coverage_factor - degrade_penalty))
            normalized.append(
                {
                    "module_id": module_id,
                    "title": title,
                    "content": content,
                    "evidence_refs": evidence_refs,
                    "coverage": coverage,
                    "confidence": round(confidence, 4),
                    "degrade_reason": degrade_reason,
                    "module_quality_score": round(module_quality_score, 4),
                    "module_degrade_code": str(degrade_reason[0]) if degrade_reason else "",
                }
            )
        return normalized

    def report_generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Generate report based on query result and cache it for retrieval/export."""
        req = ReportRequest(**payload)
        run_id = str(payload.get("run_id", "")).strip()
        pool_snapshot_id = str(payload.get("pool_snapshot_id", "")).strip()
        template_id = str(payload.get("template_id", "default")).strip() or "default"
        code = str(req.stock_code or "").strip().upper()
        report_input_pack = self._build_llm_input_pack(
            code,
            question=f"report:{code}:{req.report_type}",
            scenario="report",
        )
        query_result = self.query(
            {
                "user_id": req.user_id,
                "question": f"请生成{req.stock_code} {req.period} 的{req.report_type}报告",
                "stock_codes": [req.stock_code],
            }
        )
        overview = self.market_overview(code)
        predict_snapshot = self.predict_run({"stock_codes": [code], "horizons": ["5d", "20d"]})
        try:
            intel = self.analysis_intel_card(code, horizon="30d", risk_profile="neutral")
        except Exception:
            intel = {}
        history_sample_size = len(list(overview.get("history", []) or []))
        refresh_actions = list(((report_input_pack.get("dataset", {}) or {}).get("refresh_actions", []) or []))
        refresh_failed_count = len([x for x in refresh_actions if str((x or {}).get("status", "")).strip().lower() != "ok"])
        quality_reasons: list[str] = []
        if len(list(query_result.get("citations", []) or [])) < 3:
            quality_reasons.append("citations_insufficient")
        if history_sample_size < 60:
            quality_reasons.append("history_sample_insufficient")
        if str(predict_snapshot.get("data_quality", "")).strip() != "real":
            quality_reasons.append("predict_degraded")
        if refresh_failed_count > 0:
            quality_reasons.append("auto_refresh_failed")
        quality_reasons.extend([str(x) for x in list(report_input_pack.get("missing_data", []) or []) if str(x).strip()])
        quality_reasons = list(dict.fromkeys(quality_reasons))
        # Weighted quality gate: avoid over-penalizing isolated soft gaps (e.g. research_only gap).
        reason_weights = {
            "quote_missing": 0.32,
            "history_insufficient": 0.30,
            "history_sample_insufficient": 0.20,
            "history_30d_insufficient": 0.14,
            "financial_missing": 0.16,
            "announcement_missing": 0.10,
            "news_insufficient": 0.12,
            "research_insufficient": 0.08,
            "macro_insufficient": 0.08,
            "fund_missing": 0.06,
            "predict_degraded": 0.06,
            "citations_insufficient": 0.10,
            "auto_refresh_failed": 0.12,
        }
        quality_penalty = 0.0
        for reason in quality_reasons:
            quality_penalty += float(reason_weights.get(reason, 0.06))
        quality_penalty = min(0.85, quality_penalty)
        quality_score = round(max(0.0, 1.0 - quality_penalty), 4)
        if not quality_reasons:
            quality_status = "pass"
        elif quality_penalty < 0.28:
            quality_status = "watch"
        else:
            quality_status = "degraded"
        quality_gate = {
            "status": quality_status,
            "score": quality_score,
            "reasons": quality_reasons,
            "penalty": round(quality_penalty, 4),
        }
        quality_explain = self._build_quality_explain(
            reasons=quality_reasons,
            quality_status=quality_status,
            context="report",
        )
        report_data_pack_summary = {
            "as_of": datetime.now(timezone.utc).isoformat(),
            "history_sample_size": history_sample_size,
            "history_30d_count": int(((report_input_pack.get("dataset", {}) or {}).get("coverage", {}) or {}).get("history_30d_count", 0) or 0),
            "history_90d_count": int(((report_input_pack.get("dataset", {}) or {}).get("coverage", {}) or {}).get("history_90d_count", 0) or 0),
            "history_252d_count": int(len(list(report_input_pack.get("history_daily_252", []) or []))),
            "history_1y_has_full": bool((report_input_pack.get("history_1y_summary", {}) or {}).get("has_full_1y", False)),
            "predict_quality": str(predict_snapshot.get("data_quality", "unknown")),
            "predict_degrade_reasons": list(predict_snapshot.get("degrade_reasons", []) or []),
            "intel_signal": str(intel.get("overall_signal", "")),
            "intel_confidence": float(intel.get("confidence", 0.0) or 0.0),
            "news_count": len(list(overview.get("news", []) or [])),
            "research_count": len(list(overview.get("research", []) or [])),
            "macro_count": len(list(overview.get("macro", []) or [])),
            "quarterly_fundamentals_count": int(len(list(report_input_pack.get("quarterly_fundamentals_summary", []) or []))),
            "event_timeline_count": int(len(list(report_input_pack.get("event_timeline_1y", []) or []))),
            "uncertainty_note_count": int(len(list(report_input_pack.get("evidence_uncertainty", []) or []))),
            "time_horizon_coverage": dict(report_input_pack.get("time_horizon_coverage", {}) or {}),
            "refresh_action_count": int(len(refresh_actions)),
            "refresh_failed_count": int(refresh_failed_count),
            "quality_explain_summary": str(quality_explain.get("summary", "")),
            "missing_data": list(report_input_pack.get("missing_data", []) or []),
            "data_quality": str(report_input_pack.get("data_quality", "ready")),
        }
        metric_snapshot = self._build_report_metric_snapshot(
            overview=overview,
            predict_snapshot=predict_snapshot,
            intel=intel,
            quality_gate=quality_gate,
            citation_count=len(list(query_result.get("citations", []) or [])),
        )
        final_decision = self._build_fallback_final_decision(
            code=code,
            report_type=str(req.report_type),
            intel=intel,
            quality_gate=quality_gate,
            quality_reasons=quality_reasons,
        )
        multi_role_decision: dict[str, Any] = {
            "enabled": False,
            "trace_id": "",
            "debate_mode": "disabled",
            "opinions": [],
            "consensus_signal": str(final_decision.get("signal", "hold")),
            "consensus_confidence": float(final_decision.get("confidence", 0.5) or 0.5),
            "disagreement_score": 0.0,
            "conflict_sources": [],
            "counter_view": "",
            "judge_summary": "",
        }
        # Reuse one arbitration kernel across Predict/Report to avoid signal drift between modules.
        try:
            history_rows = [dict(row) for row in list(overview.get("history", []) or []) if isinstance(row, dict)]
            trend_for_debate = dict(overview.get("trend", {}) or {})
            if not trend_for_debate and history_rows:
                trend_for_debate = self._trend_metrics(history_rows)

            quote_for_debate = dict(overview.get("realtime", {}) or {})
            if not quote_for_debate:
                latest_bar = history_rows[-1] if history_rows else {}
                quote_for_debate = {
                    "price": self._safe_float(latest_bar.get("close", 0.0), default=0.0),
                    "pct_change": self._safe_float(latest_bar.get("pct_change", 0.0), default=0.0),
                }

            predict_rows = [dict(row) for row in list(predict_snapshot.get("results", []) or []) if isinstance(row, dict)]
            first_predict = predict_rows[0] if predict_rows else {}
            horizon_rows = [dict(row) for row in list(first_predict.get("horizons", []) or []) if isinstance(row, dict)]
            quant_20 = next(
                (row for row in horizon_rows if str(row.get("horizon", "")).strip().lower() == "20d"),
                horizon_rows[0] if horizon_rows else {},
            )
            if not quant_20:
                quant_20 = {
                    "horizon": "20d",
                    "signal": "hold",
                    "up_probability": 0.5,
                    "rationale": "report_fallback_quant20_missing",
                }

            report_quality_for_debate = {
                "overall_status": str(quality_gate.get("status", "pass")),
                "reasons": list(quality_reasons),
            }
            raw_multi_role = self._predict_run_multi_role_debate(
                stock_code=code,
                question=f"report:{code}:{req.report_type}:{req.period}",
                quote=quote_for_debate,
                trend=trend_for_debate,
                quant_20=quant_20,
                quality_gate=report_quality_for_debate,
                input_pack=report_input_pack,
            )
            opinions = [
                self._normalize_deep_opinion(
                    agent=str(row.get("agent", "unknown")),
                    signal=str(row.get("signal", "hold")),
                    confidence=self._safe_float(row.get("confidence", 0.5), default=0.5),
                    reason=str(row.get("reason", "")),
                    evidence_ids=[str(x).strip() for x in list(row.get("evidence_ids", []) or []) if str(x).strip()],
                    risk_tags=[str(x).strip() for x in list(row.get("risk_tags", []) or []) if str(x).strip()],
                )
                for row in list(raw_multi_role.get("opinions", []) or [])
                if isinstance(row, dict)
            ]
            consensus_signal = self._normalize_report_signal(
                str(raw_multi_role.get("consensus_signal", final_decision.get("signal", "hold")))
            )
            consensus_confidence = max(
                0.2,
                min(0.95, self._safe_float(raw_multi_role.get("consensus_confidence", 0.5), default=0.5)),
            )
            multi_role_decision = {
                "enabled": bool(opinions),
                "trace_id": str(raw_multi_role.get("trace_id", "")),
                "debate_mode": str(raw_multi_role.get("debate_mode", "rule_fallback")),
                "opinions": opinions,
                "consensus_signal": consensus_signal,
                "consensus_confidence": round(consensus_confidence, 4),
                "disagreement_score": round(self._safe_float(raw_multi_role.get("disagreement_score", 0.0), default=0.0), 4),
                "conflict_sources": [
                    str(x).strip()
                    for x in list(raw_multi_role.get("conflict_sources", []) or [])
                    if str(x).strip()
                ],
                "counter_view": str(raw_multi_role.get("counter_view", "")).strip()[:320],
                "judge_summary": str(raw_multi_role.get("judge_summary", "")).strip()[:320],
            }
            if bool(multi_role_decision.get("enabled", False)):
                final_decision["signal"] = str(multi_role_decision.get("consensus_signal", final_decision.get("signal", "hold")))
                blended_confidence = (
                    self._safe_float(final_decision.get("confidence", 0.5), default=0.5) * 0.55
                    + float(multi_role_decision.get("consensus_confidence", 0.5) or 0.5) * 0.45
                )
                final_decision["confidence"] = round(
                    max(
                        0.25,
                        min(
                            self._safe_float(quality_gate.get("score", 0.5), default=0.5),
                            blended_confidence,
                        ),
                    ),
                    4,
                )
                judge_summary = str(multi_role_decision.get("judge_summary", "")).strip()
                if judge_summary:
                    final_decision["rationale"] = f"{str(final_decision.get('rationale', '')).strip()} 多角色裁决：{judge_summary}".strip()
                conflict_sources = [str(x).strip() for x in list(multi_role_decision.get("conflict_sources", []) or []) if str(x).strip()]
                if conflict_sources:
                    final_decision["invalidation_conditions"] = list(
                        dict.fromkeys(
                            list(final_decision.get("invalidation_conditions", []) or [])
                            + [f"角色冲突源持续存在：{', '.join(conflict_sources[:3])}"]
                        )
                    )[:10]
                counter_view = str(multi_role_decision.get("counter_view", "")).strip()
                if counter_view:
                    final_decision["execution_plan"] = list(
                        dict.fromkeys(
                            list(final_decision.get("execution_plan", []) or [])
                            + [f"反方观点复核：{counter_view[:120]}"]
                        )
                    )[:8]
        except Exception as ex:  # noqa: BLE001
            # Report should still render even if arbitration helper fails unexpectedly.
            multi_role_decision["judge_summary"] = f"multi_role_fallback:{str(ex)[:160]}"

        multi_role_node: dict[str, Any] | None = None
        if bool(multi_role_decision.get("enabled", False)):
            quality_status = str(quality_gate.get("status", "pass")).strip().lower() or "pass"
            node_status = "degraded" if quality_status == "degraded" else "watch" if quality_status == "watch" else "ready"
            multi_role_node = {
                "node_id": "multi_role_judge",
                "title": "多角色裁决器",
                "status": node_status,
                "signal": str(multi_role_decision.get("consensus_signal", "hold")),
                "confidence": self._safe_float(multi_role_decision.get("consensus_confidence", 0.5), default=0.5),
                "summary": str(multi_role_decision.get("judge_summary", "")).strip() or "多角色裁决已执行。",
                "highlights": [
                    f"冲突源：{', '.join(list(multi_role_decision.get('conflict_sources', [])) or ['none'])}",
                    f"分歧度：{self._safe_float(multi_role_decision.get('disagreement_score', 0.0), default=0.0):.2f}",
                ],
                "evidence_refs": [],
                "coverage": {"role_count": int(len(list(multi_role_decision.get("opinions", []) or [])))},
                "degrade_reason": [],
                "guardrails": (
                    [f"反方观点：{str(multi_role_decision.get('counter_view', '')).strip()[:120]}"]
                    if str(multi_role_decision.get("counter_view", "")).strip()
                    else []
                ),
                "veto": bool("risk_veto" in list(multi_role_decision.get("conflict_sources", []) or [])),
            }

        # Nodeized committee pipeline: explicitly separate research synthesis and risk arbitration.
        analysis_nodes = self._normalize_report_analysis_nodes(
            list(
                [
                self._build_report_research_summarizer_node(
                    code=code,
                    query_result=query_result,
                    overview=overview,
                    intel=intel,
                    final_decision=final_decision,
                    quality_gate=quality_gate,
                ),
                self._build_report_risk_arbiter_node(
                    intel=intel,
                    quality_gate=quality_gate,
                    final_decision=final_decision,
                ),
                ]
                + ([multi_role_node] if isinstance(multi_role_node, dict) else [])
            )
        )
        committee = self._build_report_committee_notes(
            final_decision=final_decision,
            quality_gate=quality_gate,
            intel=intel,
            quality_reasons=quality_reasons,
            analysis_nodes=analysis_nodes,
        )
        if bool(multi_role_decision.get("enabled", False)) and str(multi_role_decision.get("judge_summary", "")).strip():
            committee["research_note"] = (
                f"{str(committee.get('research_note', '')).strip()} "
                f"多角色裁决：{str(multi_role_decision.get('judge_summary', '')).strip()}"
            ).strip()
        report_modules = self._build_fallback_report_modules(
            code=code,
            query_result=query_result,
            overview=overview,
            predict_snapshot=predict_snapshot,
            intel=intel,
            report_input_pack=report_input_pack,
            quality_gate=quality_gate,
            quality_reasons=quality_reasons,
            final_decision=final_decision,
        )
        module_order = [
            "executive_summary",
            "market_technical",
            "fundamental_valuation",
            "news_event",
            "sentiment_flow",
            "risk_matrix",
            "scenario_plan",
            "bull_case",
            "bear_case",
            "risky_case",
            "safe_case",
            "execution_plan",
            "final_decision",
        ]
        generation_mode = "fallback"
        generation_error = ""
        report_id = str(uuid.uuid4())
        evidence_refs = [self._build_report_evidence_ref(dict(c)) for c in list(query_result.get("citations", []) or [])]
        report_sections = [
            {"section_id": "summary", "title": "结论摘要", "content": str(query_result["answer"])[:800]},
            {
                "section_id": "evidence",
                "title": "证据清单",
                "content": "\n".join(f"- {x['source_id']}: {x['excerpt']}" for x in evidence_refs[:8]),
            },
            {"section_id": "risk", "title": "风险与反证", "content": "结合估值、流动性与政策扰动进行反证校验。"},
            {"section_id": "action", "title": "操作建议", "content": "建议分批验证信号稳定性，避免一次性重仓。"},
        ]
        for row in list(intel.get("evidence", []) or [])[:6]:
            evidence_refs.append(self._build_report_evidence_ref(dict(row)))
        evidence_refs = self._normalize_report_evidence_refs(evidence_refs)[:14]

        report_sections.extend(
            [
                {
                    "section_id": "evidence_summary",
                    "title": "证据摘要",
                    "content": json.dumps(
                        {
                            "citation_count": int(len(evidence_refs)),
                            "top_citation_sources": [str(x.get("source_id", "")) for x in evidence_refs[:8] if str(x.get("source_id", "")).strip()],
                            "catalyst_count": int(len(list(intel.get("key_catalysts", []) or []))),
                            "risk_watch_count": int(len(list(intel.get("risk_watch", []) or []))),
                        },
                        ensure_ascii=False,
                    ),
                },
                {
                    "section_id": "uncertainty_notes",
                    "title": "不确定性清单",
                    "content": "\n".join(f"- {line}" for line in list(report_input_pack.get("evidence_uncertainty", []) or [])),
                },
                {
                    "section_id": "time_horizon_coverage",
                    "title": "时域覆盖度",
                    "content": json.dumps(dict(report_input_pack.get("time_horizon_coverage", {}) or {}), ensure_ascii=False),
                },
                {
                    "section_id": "data_quality_gate",
                    "title": "数据质量门控",
                    "content": f"status={quality_gate['status']}; score={quality_gate['score']}; reasons={','.join(quality_reasons) or 'none'}",
                },
                {
                    "section_id": "signal_context",
                    "title": "信号上下文",
                    "content": json.dumps(
                        {
                            "predict_quality": str(predict_snapshot.get("data_quality", "unknown")),
                            "predict_degrade_reasons": list(predict_snapshot.get("degrade_reasons", []) or []),
                            "intel_signal": str(intel.get("overall_signal", "")),
                            "intel_confidence": float(intel.get("confidence", 0.0) or 0.0),
                        },
                        ensure_ascii=False,
                    ),
                },
                {
                    "section_id": "multi_role_arbitration",
                    "title": "多角色仲裁摘要",
                    "content": json.dumps(
                        {
                            "enabled": bool(multi_role_decision.get("enabled", False)),
                            "trace_id": str(multi_role_decision.get("trace_id", "")),
                            "debate_mode": str(multi_role_decision.get("debate_mode", "")),
                            "consensus_signal": str(multi_role_decision.get("consensus_signal", "hold")),
                            "consensus_confidence": self._safe_float(
                                multi_role_decision.get("consensus_confidence", 0.5), default=0.5
                            ),
                            "disagreement_score": self._safe_float(
                                multi_role_decision.get("disagreement_score", 0.0), default=0.0
                            ),
                            "conflict_sources": list(multi_role_decision.get("conflict_sources", []) or []),
                            "judge_summary": str(multi_role_decision.get("judge_summary", "")),
                        },
                        ensure_ascii=False,
                    ),
                },
                {
                    "section_id": "scenario_matrix",
                    "title": "情景分析矩阵",
                    "content": json.dumps(list(intel.get("scenario_matrix", []) or [])[:6], ensure_ascii=False),
                },
                {
                    "section_id": "llm_input_pack",
                    "title": "模型输入数据包摘要",
                    "content": json.dumps(
                        {
                            "coverage": dict((report_input_pack.get("dataset", {}) or {}).get("coverage", {}) or {}),
                            "missing_data": list(report_input_pack.get("missing_data", []) or []),
                            "data_quality": str(report_input_pack.get("data_quality", "")),
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
        )
        if self.settings.llm_external_enabled and self.llm_gateway.providers:
            llm_prompt = (
                "你是A股研究报告助手。请基于给定context输出严格JSON，不要输出markdown或解释性文本。"
                "顶层JSON必须包含字段：modules, final_decision, committee, next_data_needed。"
                "modules 为数组，元素结构："
                "{module_id,title,content,evidence_refs,confidence,coverage,degrade_reason}。"
                "module_id 仅允许：executive_summary,market_technical,fundamental_valuation,news_event,"
                "sentiment_flow,risk_matrix,scenario_plan,bull_case,bear_case,risky_case,safe_case,execution_plan,final_decision。"
                "final_decision 结构：{signal,confidence,rationale,invalidation_conditions,execution_plan,target_price,risk_score,reward_risk_ratio,position_sizing_hint}，"
                "signal 仅允许 buy/hold/reduce。"
                "committee 结构：{research_note,risk_note}。"
                "要求："
                "1) 明确引用数据覆盖缺口，不得忽略 missing_data。"
                "2) 若数据不足，降低结论确信度并给出补数建议。"
                "3) 模块内容聚焦交易可执行性与风险触发条件。"
                "4) confidence 必须在 0 到 1。"
                "5) 必须显式使用 1y 证据包（history_daily_252 / quarterly_fundamentals_summary / event_timeline_1y）。"
                "6) 若1y样本不足，必须在内容里写出不确定性说明。"
                "context="
                + json.dumps(
                    {
                        "stock_code": code,
                        "report_type": req.report_type,
                        "query_answer": query_result.get("answer", ""),
                        "financial": overview.get("financial", {}),
                        "trend": overview.get("trend", {}),
                        "predict": {
                            "quality": predict_snapshot.get("data_quality", ""),
                            "horizons": ((predict_snapshot.get("results", []) or [{}])[0].get("horizons", []) if predict_snapshot.get("results") else []),
                        },
                        "intel": {
                            "signal": intel.get("overall_signal", ""),
                            "confidence": intel.get("confidence", 0.0),
                            "risk_watch": intel.get("risk_watch", []),
                            "catalysts": intel.get("key_catalysts", []),
                            "scenario_matrix": intel.get("scenario_matrix", []),
                        },
                        "quality_gate": quality_gate,
                        "metric_snapshot": metric_snapshot,
                        "input_pack": {
                            "coverage": dict((report_input_pack.get("dataset", {}) or {}).get("coverage", {}) or {}),
                            "missing_data": list(report_input_pack.get("missing_data", []) or []),
                            "data_quality": str(report_input_pack.get("data_quality", "")),
                            "history_daily_30": list(report_input_pack.get("history_daily_30", []) or []),
                            "history_daily_252": list(report_input_pack.get("history_daily_252", []) or []),
                            "history_1y_summary": dict(report_input_pack.get("history_1y_summary", {}) or {}),
                            "history_monthly_summary_12": list(report_input_pack.get("history_monthly_summary_12", []) or []),
                            "quarterly_fundamentals_summary": list(report_input_pack.get("quarterly_fundamentals_summary", []) or []),
                            "event_timeline_1y": list(report_input_pack.get("event_timeline_1y", []) or []),
                            "time_horizon_coverage": dict(report_input_pack.get("time_horizon_coverage", {}) or {}),
                            "evidence_uncertainty": list(report_input_pack.get("evidence_uncertainty", []) or []),
                        },
                    },
                    ensure_ascii=False,
                )
            )
            state = AgentState(
                user_id="report-generator",
                question=f"report:{code}",
                stock_codes=[code],
                trace_id=self.traces.new_trace(),
            )
            try:
                raw = self.llm_gateway.generate(state, llm_prompt)
                parsed = self._deep_safe_json_loads(raw)
                if isinstance(parsed, dict) and parsed:
                    generation_mode = "llm"
                    modules_map: dict[str, dict[str, Any]] = {
                        str(x.get("module_id", "")): dict(x)
                        for x in report_modules
                        if isinstance(x, dict) and str(x.get("module_id", "")).strip()
                    }

                    parsed_modules = list(parsed.get("modules", []) or [])
                    for row in parsed_modules:
                        if not isinstance(row, dict):
                            continue
                        module_id = str(row.get("module_id", "")).strip().lower()
                        if module_id not in module_order:
                            continue
                        base = dict(modules_map.get(module_id, {}))
                        base["module_id"] = module_id
                        if str(row.get("title", "")).strip():
                            base["title"] = str(row.get("title", "")).strip()
                        if str(row.get("content", "")).strip():
                            base["content"] = str(row.get("content", "")).strip()
                        if isinstance(row.get("evidence_refs"), list):
                            base["evidence_refs"] = [str(x).strip() for x in list(row.get("evidence_refs", [])) if str(x).strip()]
                        if row.get("confidence") is not None:
                            base["confidence"] = self._safe_float(row.get("confidence", 0.5), default=0.5)
                        if isinstance(row.get("coverage"), dict):
                            base["coverage"] = dict(row.get("coverage", {}) or {})
                        if isinstance(row.get("degrade_reason"), list):
                            base["degrade_reason"] = [str(x).strip() for x in list(row.get("degrade_reason", [])) if str(x).strip()]
                        modules_map[module_id] = base

                    # Backward compatibility for older prompt schema.
                    if "modules" not in parsed:
                        legacy_summary = str(parsed.get("executive_summary", "")).strip()
                        if legacy_summary:
                            modules_map["executive_summary"]["content"] = legacy_summary
                        core_logic = list(parsed.get("core_logic", []) or [])
                        if core_logic:
                            modules_map["market_technical"]["content"] = "\n".join(f"- {str(x)}" for x in core_logic[:10])
                        legacy_risk = list(parsed.get("risk_matrix", []) or [])
                        if legacy_risk:
                            modules_map["risk_matrix"]["content"] = json.dumps(legacy_risk[:8], ensure_ascii=False)
                        legacy_scenario = list(parsed.get("scenario_analysis", []) or [])
                        if legacy_scenario:
                            modules_map["scenario_plan"]["content"] = json.dumps(legacy_scenario[:8], ensure_ascii=False)
                        legacy_exec = list(parsed.get("execution_plan", []) or [])
                        if legacy_exec:
                            modules_map["execution_plan"]["content"] = "\n".join(f"- {str(x)}" for x in legacy_exec[:10])

                    parsed_decision = parsed.get("final_decision")
                    if isinstance(parsed_decision, dict):
                        final_decision["signal"] = self._normalize_report_signal(str(parsed_decision.get("signal", final_decision.get("signal", "hold"))))
                        final_decision["confidence"] = round(
                            max(0.2, min(1.0, self._safe_float(parsed_decision.get("confidence", final_decision.get("confidence", 0.5)), default=0.5))),
                            4,
                        )
                        if str(parsed_decision.get("rationale", "")).strip():
                            final_decision["rationale"] = str(parsed_decision.get("rationale", "")).strip()
                        if isinstance(parsed_decision.get("invalidation_conditions"), list):
                            final_decision["invalidation_conditions"] = [
                                str(x).strip()
                                for x in list(parsed_decision.get("invalidation_conditions", []))
                                if str(x).strip()
                            ][:8]
                        if isinstance(parsed_decision.get("execution_plan"), list):
                            final_decision["execution_plan"] = [
                                str(x).strip()
                                for x in list(parsed_decision.get("execution_plan", []))
                                if str(x).strip()
                            ][:8]
                        target_raw = parsed_decision.get("target_price")
                        if isinstance(target_raw, dict):
                            final_decision["target_price"] = {
                                "low": round(self._safe_float(target_raw.get("low", 0.0), default=0.0), 3),
                                "base": round(self._safe_float(target_raw.get("base", 0.0), default=0.0), 3),
                                "high": round(self._safe_float(target_raw.get("high", 0.0), default=0.0), 3),
                            }
                        elif isinstance(target_raw, (int, float)):
                            base_target = round(float(target_raw), 3)
                            final_decision["target_price"] = {
                                "low": round(base_target * 0.95, 3),
                                "base": base_target,
                                "high": round(base_target * 1.05, 3),
                            }
                        if parsed_decision.get("risk_score") is not None:
                            final_decision["risk_score"] = round(
                                max(0.0, min(100.0, self._safe_float(parsed_decision.get("risk_score", 50.0), default=50.0))),
                                2,
                            )
                        if parsed_decision.get("reward_risk_ratio") is not None:
                            final_decision["reward_risk_ratio"] = round(
                                max(0.0, min(10.0, self._safe_float(parsed_decision.get("reward_risk_ratio", 1.0), default=1.0))),
                                3,
                            )
                        if str(parsed_decision.get("position_sizing_hint", "")).strip():
                            final_decision["position_sizing_hint"] = str(parsed_decision.get("position_sizing_hint", "")).strip()[:80]

                    parsed_committee = parsed.get("committee")
                    if isinstance(parsed_committee, dict):
                        if str(parsed_committee.get("research_note", "")).strip():
                            committee["research_note"] = str(parsed_committee.get("research_note", "")).strip()
                        if str(parsed_committee.get("risk_note", "")).strip():
                            committee["risk_note"] = str(parsed_committee.get("risk_note", "")).strip()

                    if isinstance(parsed.get("next_data_needed"), list):
                        llm_missing = [str(x).strip() for x in list(parsed.get("next_data_needed", [])) if str(x).strip()]
                        if llm_missing:
                            final_decision["invalidation_conditions"] = list(dict.fromkeys(list(final_decision.get("invalidation_conditions", [])) + llm_missing))[:10]

                    report_modules = [modules_map.get(mid, {"module_id": mid, "title": mid, "content": "该模块暂无可用内容。"}) for mid in module_order]
                    report_modules = self._normalize_report_modules(report_modules)
            except Exception as ex:
                generation_error = str(ex)[:240]

        # Keep final decision confidence bounded by quality gate even if LLM outputs larger value.
        final_decision["confidence"] = round(
            min(
                self._safe_float(final_decision.get("confidence", 0.5), default=0.5),
                max(0.25, self._safe_float(quality_gate.get("score", 0.5), default=0.5)),
            ),
            4,
        )
        final_decision["signal"] = self._normalize_report_signal(str(final_decision.get("signal", "hold")))
        committee = {
            "research_note": self._sanitize_report_text(str(committee.get("research_note", "")).strip() or "研究汇总暂不可用。"),
            "risk_note": self._sanitize_report_text(str(committee.get("risk_note", "")).strip() or "风险仲裁暂不可用。"),
        }
        analysis_nodes = self._normalize_report_analysis_nodes(list(analysis_nodes or []))
        quality_dashboard = self._build_report_quality_dashboard(
            report_modules=report_modules,
            quality_gate=quality_gate,
            final_decision=final_decision,
            analysis_nodes=analysis_nodes,
            evidence_refs=evidence_refs,
        )

        # Project module payload into report_sections for backward-compatible renderers.
        for module in report_modules:
            if not isinstance(module, dict):
                continue
            report_sections.append(
                {
                    "section_id": f"module_{str(module.get('module_id', '')).strip() or 'unknown'}",
                    "title": str(module.get("title", "模块内容")),
                    "content": str(module.get("content", ""))[:2200],
                }
            )
        report_sections.append(
            {
                "section_id": "committee_notes",
                "title": "委员会纪要",
                "content": json.dumps(committee, ensure_ascii=False),
            }
        )
        report_sections.append(
            {
                "section_id": "final_decision",
                "title": "综合决策",
                "content": json.dumps(final_decision, ensure_ascii=False),
            }
        )
        report_sections.append(
            {
                "section_id": "metric_snapshot",
                "title": "指标快照",
                "content": json.dumps(metric_snapshot, ensure_ascii=False),
            }
        )
        report_sections.append(
            {
                "section_id": "analysis_nodes",
                "title": "研究与风险节点",
                "content": json.dumps(analysis_nodes, ensure_ascii=False),
            }
        )
        report_sections.append(
            {
                "section_id": "quality_dashboard",
                "title": "质量评估看板",
                "content": json.dumps(quality_dashboard, ensure_ascii=False),
            }
        )

        markdown = (
            f"# {req.stock_code} 分析报告\n\n"
            f"## 结论\n{query_result['answer']}\n\n"
            "## 证据\n"
            + "\n".join(f"- {c['source_id']}: {c['excerpt']}" for c in query_result["citations"])
            + "\n\n## 风险与反证\n结合估值、流动性与政策扰动进行反证校验。"
            + "\n\n## 操作建议\n建议分批验证信号稳定性，避免一次性重仓。"
        )
        markdown += (
            "\n\n## 最终决策\n"
            f"- signal: {str(final_decision.get('signal', 'hold'))}\n"
            f"- confidence: {self._safe_float(final_decision.get('confidence', 0.5), default=0.5):.2f}\n"
            f"- target_price: {json.dumps(dict(final_decision.get('target_price', {}) or {}), ensure_ascii=False)}\n"
            f"- risk_score: {self._safe_float(final_decision.get('risk_score', 0.0), default=0.0):.2f}\n"
            f"- reward_risk_ratio: {self._safe_float(final_decision.get('reward_risk_ratio', 0.0), default=0.0):.3f}\n"
            f"- position_sizing_hint: {str(final_decision.get('position_sizing_hint', ''))}\n"
            f"- rationale: {str(final_decision.get('rationale', ''))[:400]}\n"
        )
        if bool(multi_role_decision.get("enabled", False)):
            markdown += (
                "\n\n## 多角色仲裁\n"
                f"- trace_id: {str(multi_role_decision.get('trace_id', ''))}\n"
                f"- consensus_signal: {str(multi_role_decision.get('consensus_signal', 'hold'))}\n"
                f"- consensus_confidence: {self._safe_float(multi_role_decision.get('consensus_confidence', 0.5), default=0.5):.2f}\n"
                f"- conflict_sources: {','.join(list(multi_role_decision.get('conflict_sources', []) or [])) or 'none'}\n"
                f"- judge_summary: {str(multi_role_decision.get('judge_summary', ''))[:320]}\n"
            )
        markdown += (
            "\n\n## Data Quality Gate\n"
            f"- status: {quality_gate['status']}\n"
            f"- score: {float(quality_gate.get('score', 0.0) or 0.0):.2f}\n"
            f"- reasons: {','.join(quality_reasons) or 'none'}\n"
            f"- generation_mode: {generation_mode}\n"
            + (f"- generation_error: {generation_error}\n" if generation_error else "")
        )
        self._reports[report_id] = {
            "report_id": report_id,
            "schema_version": self._report_bundle_schema_version,
            "trace_id": query_result["trace_id"],
            "markdown": markdown,
            "citations": query_result["citations"],
            "run_id": run_id,
            "pool_snapshot_id": pool_snapshot_id,
            "template_id": template_id,
            "evidence_refs": evidence_refs,
            "report_sections": report_sections,
            "quality_gate": quality_gate,
            "report_data_pack_summary": report_data_pack_summary,
            "report_modules": report_modules,
            "committee": committee,
            "final_decision": final_decision,
            "metric_snapshot": metric_snapshot,
            "analysis_nodes": analysis_nodes,
            "quality_dashboard": quality_dashboard,
            "multi_role_enabled": bool(multi_role_decision.get("enabled", False)),
            "multi_role_trace_id": str(multi_role_decision.get("trace_id", "")),
            "multi_role_decision": dict(multi_role_decision),
            "role_opinions": list(multi_role_decision.get("opinions", []) or []),
            "judge_summary": str(multi_role_decision.get("judge_summary", "")),
            "conflict_sources": list(multi_role_decision.get("conflict_sources", []) or []),
            "consensus_signal": str(multi_role_decision.get("consensus_signal", final_decision.get("signal", "hold"))),
            "consensus_confidence": self._safe_float(
                multi_role_decision.get("consensus_confidence", final_decision.get("confidence", 0.5)),
                default=self._safe_float(final_decision.get("confidence", 0.5), default=0.5),
            ),
            "confidence_attribution": {
                "citation_count": len(list(query_result.get("citations", []) or [])),
                "history_sample_size": history_sample_size,
                "predict_quality": str(predict_snapshot.get("data_quality", "unknown")),
                "quality_score": float(quality_gate.get("score", 0.0) or 0.0),
            },
            "llm_input_pack": report_input_pack,
            "generation_mode": generation_mode,
            "generation_error": generation_error,
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
        version_payload = {
            "schema_version": self._report_bundle_schema_version,
            "report_id": report_id,
            "stock_code": str(req.stock_code),
            "report_type": str(req.report_type),
            "final_decision": dict(final_decision),
            "committee": dict(committee),
            "report_modules": list(report_modules),
            "analysis_nodes": list(analysis_nodes),
            "quality_dashboard": dict(quality_dashboard),
            "metric_snapshot": dict(metric_snapshot),
            "quality_gate": dict(quality_gate),
            "evidence_refs": list(evidence_refs),
            "multi_role_enabled": bool(multi_role_decision.get("enabled", False)),
            "multi_role_trace_id": str(multi_role_decision.get("trace_id", "")),
            "multi_role_decision": dict(multi_role_decision),
            "role_opinions": list(multi_role_decision.get("opinions", []) or []),
            "judge_summary": str(multi_role_decision.get("judge_summary", "")),
            "conflict_sources": list(multi_role_decision.get("conflict_sources", []) or []),
            "consensus_signal": str(multi_role_decision.get("consensus_signal", final_decision.get("signal", "hold"))),
            "consensus_confidence": self._safe_float(
                multi_role_decision.get("consensus_confidence", final_decision.get("confidence", 0.5)),
                default=self._safe_float(final_decision.get("confidence", 0.5), default=0.5),
            ),
        }
        self.web.save_report_index(
            report_id=report_id,
            user_id=user_id,
            tenant_id=tenant_id,
            stock_code=req.stock_code,
            report_type=req.report_type,
            markdown=markdown,
            run_id=run_id,
            pool_snapshot_id=pool_snapshot_id,
            template_id=template_id,
            payload_json=json.dumps(version_payload, ensure_ascii=False),
        )
        resp = ReportResponse(
            report_id=report_id,
            trace_id=query_result["trace_id"],
            markdown=markdown,
            citations=[Citation(**c) for c in query_result["citations"]],
        ).model_dump(mode="json")
        resp["run_id"] = run_id
        resp["pool_snapshot_id"] = pool_snapshot_id
        resp["template_id"] = template_id
        resp["evidence_refs"] = evidence_refs
        resp["report_sections"] = report_sections
        resp["quality_gate"] = quality_gate
        resp["report_data_pack_summary"] = report_data_pack_summary
        resp["report_modules"] = report_modules
        resp["committee"] = committee
        resp["final_decision"] = final_decision
        resp["metric_snapshot"] = metric_snapshot
        resp["analysis_nodes"] = analysis_nodes
        resp["quality_dashboard"] = quality_dashboard
        resp["multi_role_enabled"] = bool(multi_role_decision.get("enabled", False))
        resp["multi_role_trace_id"] = str(multi_role_decision.get("trace_id", ""))
        resp["multi_role_decision"] = dict(multi_role_decision)
        resp["role_opinions"] = list(multi_role_decision.get("opinions", []) or [])
        resp["judge_summary"] = str(multi_role_decision.get("judge_summary", ""))
        resp["conflict_sources"] = list(multi_role_decision.get("conflict_sources", []) or [])
        resp["consensus_signal"] = str(multi_role_decision.get("consensus_signal", final_decision.get("signal", "hold")))
        resp["consensus_confidence"] = self._safe_float(
            multi_role_decision.get("consensus_confidence", final_decision.get("confidence", 0.5)),
            default=self._safe_float(final_decision.get("confidence", 0.5), default=0.5),
        )
        resp["confidence_attribution"] = {
            "citation_count": len(list(query_result.get("citations", []) or [])),
            "history_sample_size": history_sample_size,
            "predict_quality": str(predict_snapshot.get("data_quality", "unknown")),
            "quality_score": float(quality_gate.get("score", 0.0) or 0.0),
        }
        resp["llm_input_pack"] = report_input_pack
        resp["generation_mode"] = generation_mode
        resp["generation_error"] = generation_error
        resp["stock_code"] = str(req.stock_code)
        resp["report_type"] = str(req.report_type)
        resp["schema_version"] = self._report_bundle_schema_version
        resp["result_level"] = "full"
        degrade_code = ""
        if quality_gate["status"] == "degraded":
            degrade_code = "quality_degraded"
        elif quality_gate["status"] == "watch":
            degrade_code = "quality_watch"
        severity = "low"
        if quality_gate["status"] == "degraded":
            severity = "high" if float(quality_gate.get("score", 0.0) or 0.0) < 0.5 else "medium"
        elif quality_gate["status"] == "watch":
            severity = "low"
        resp["degrade"] = {
            "active": bool(quality_reasons),
            "code": degrade_code if quality_reasons else "",
            "severity": severity,
            "reasons": list(dict.fromkeys(quality_reasons)),
            "missing_data": list(dict.fromkeys(quality_reasons)),
            "confidence_penalty": round(min(0.7, 0.15 * float(len(quality_reasons))), 4),
            "user_message": str(quality_explain.get("summary", "")),
            "explain": quality_explain,
        }
        sanitized = self._sanitize_report_payload(resp)
        self._reports[report_id] = dict(sanitized)
        return sanitized

    def report_get(self, report_id: str) -> dict[str, Any]:
        """Get report by report_id."""
        report = self._reports.get(report_id)
        if not report:
            return {"error": "not_found", "report_id": report_id}
        return self._sanitize_report_payload(dict(report))

    def report_self_test(
        self,
        *,
        stock_code: str = "SH600000",
        report_type: str = "research",
        period: str = "1y",
    ) -> dict[str, Any]:
        """Run report sync/async smoke checks and return deterministic diagnostics."""
        code = str(stock_code or "SH600000").strip().upper().replace(".", "")
        normalized_report_type = str(report_type or "research").strip() or "research"
        normalized_period = str(period or "1y").strip() or "1y"
        payload = {
            "user_id": "report-self-test",
            "stock_code": code,
            "period": normalized_period,
            "report_type": normalized_report_type,
        }

        started = time.perf_counter()
        generated = self.report_generate(payload)
        sync_latency_ms = int((time.perf_counter() - started) * 1000)
        report_id = str(generated.get("report_id", "")).strip()

        loaded = self.report_get(report_id) if report_id else {"error": "missing_report_id"}
        sync_ok = isinstance(loaded, dict) and "error" not in loaded and bool(loaded.get("report_modules"))

        task = self.report_task_create(payload)
        task_id = str(task.get("task_id", "")).strip()
        final_status = str(task.get("status", "")).strip().lower() or "created"
        snapshot = dict(task)
        for _ in range(120):
            if final_status in {"completed", "failed", "partial_ready"}:
                break
            time.sleep(0.05)
            snapshot = self.report_task_get(task_id)
            final_status = str(snapshot.get("status", "")).strip().lower() or final_status
        async_result = self.report_task_result(task_id) if task_id else {"error": "missing_task_id"}
        async_payload = dict(async_result.get("result", {}) or {}) if isinstance(async_result, dict) else {}
        async_ok = bool(async_payload.get("report_modules")) and str(async_result.get("status", "")).strip().lower() in {
            "completed",
            "partial_ready",
            "failed",
        }

        return {
            "ok": bool(sync_ok and async_ok),
            "stock_code": code,
            "report_type": normalized_report_type,
            "period": normalized_period,
            "sync_latency_ms": sync_latency_ms,
            "sync": {
                "ok": bool(sync_ok),
                "report_id": report_id,
                "quality_status": str((generated.get("quality_gate", {}) or {}).get("status", "")),
                "multi_role_enabled": bool(generated.get("multi_role_enabled", False)),
                "multi_role_trace_id": str(generated.get("multi_role_trace_id", "")),
                "consensus_signal": str(generated.get("consensus_signal", "")),
                "consensus_confidence": self._safe_float(generated.get("consensus_confidence", 0.0), default=0.0),
                "module_count": int(len(list(generated.get("report_modules", []) or []))),
            },
            "async": {
                "ok": bool(async_ok),
                "task_id": task_id,
                "final_status": final_status,
                "result_level": str(async_result.get("result_level", "")) if isinstance(async_result, dict) else "",
                "display_ready": bool(async_result.get("display_ready", False)) if isinstance(async_result, dict) else False,
                "quality_status": str((async_payload.get("quality_gate", {}) or {}).get("status", "")),
                "multi_role_enabled": bool(async_payload.get("multi_role_enabled", False)),
                "multi_role_trace_id": str(async_payload.get("multi_role_trace_id", "")),
                "module_count": int(len(list(async_payload.get("report_modules", []) or []))),
            },
        }

    def _latest_report_context(self, stock_code: str) -> dict[str, Any]:
        """Get latest in-memory report context for one stock to assist DeepThink rounds."""
        code = str(stock_code or "").strip().upper()
        if not code:
            return {"available": False}
        latest: dict[str, Any] | None = None
        # Iterate reverse insertion order to find newest matching report quickly.
        for value in reversed(list(self._reports.values())):
            if not isinstance(value, dict):
                continue
            row_code = str(value.get("stock_code", "")).strip().upper()
            if row_code == code:
                latest = value
                break
        if not latest:
            return {"available": False, "stock_code": code}
        decision = dict(latest.get("final_decision", {}) or {})
        committee = dict(latest.get("committee", {}) or {})
        node_map = {
            str(row.get("node_id", "")).strip().lower(): row
            for row in list(latest.get("analysis_nodes", []) or [])
            if isinstance(row, dict) and str(row.get("node_id", "")).strip()
        }
        research_node = dict(node_map.get("research_summarizer", {}) or {})
        risk_node = dict(node_map.get("risk_arbiter", {}) or {})
        return {
            "available": True,
            "stock_code": code,
            "report_id": str(latest.get("report_id", "")),
            "signal": self._normalize_report_signal(str(decision.get("signal", "hold"))),
            "confidence": max(0.0, min(1.0, self._safe_float(decision.get("confidence", 0.5), default=0.5))),
            "rationale": str(decision.get("rationale", ""))[:400],
            "research_note": str(committee.get("research_note", ""))[:300],
            "risk_note": str(committee.get("risk_note", ""))[:300],
            "research_summary": str(research_node.get("summary", ""))[:280],
            "risk_summary": str(risk_node.get("summary", ""))[:280],
        }

    def _sanitize_report_text(self, raw: str) -> str:
        """Normalize known mojibake fragments in report-facing Chinese text."""
        text = str(raw or "")
        replacements = {
            # Common mojibake fragments observed in report fields.
            "缂佹捁顔戦幗妯款洣": "Executive Summary",
            "鐠囦焦宓佸〒鍛礋": "Evidence List",
            "妞嬪酣娅撴稉搴″冀鐠?": "Risks And Counterpoints",
            "閹垮秳缍斿楦款唴": "Action Plan",
            "闁轰胶澧楀畵浣烘嫻閵娾晛娅ら梻鍌樺妿": "Data Quality Gate",
            "闁圭瑳鍡╂斀闁硅姤顭堥々?LLM)": "LLM Summary",
            "闁哄秶顭堢缓楣冩焻閺勫繒甯?LLM)": "Core Logic (LLM)",
            "濡炲閰ｅ▍鎾绘儗閳哄懏鈻?LLM)": "Risk Matrix (LLM)",
            "闁诡垰鎳忓▍娆撳箳閵婏妇宸?LLM)": "Scenario Analysis (LLM)",
            "閺嶇绺鹃柅鏄忕帆(LLM)": "Core Logic (LLM)",
            "閹躲儱鎲℃稉顓炵妇": "Report Center",
            "閸掑棙鐎介幎銉ユ啞": "Analysis Report",
            "妞嬪酣娅撴稉搴″冀鐠囦箺n": "Risks And Counterpoints\n",
            # Also cover visible mojibake strings from historical payloads.
            "缁撹鎽樿": "Executive Summary",
            "璇佹嵁娓呭崟": "Evidence List",
            "椋庨櫓涓庡弽璇?": "Risks And Counterpoints",
            "鎿嶄綔寤鸿": "Action Plan",
            "閺佺増宓佺拹銊╁櫤闂傘劎": "Data Quality Gate",
            "閹笛嗩攽閹芥顩?LLM)": "LLM Summary",
            "妞嬪酣娅撻惌鈺呮█(LLM)": "Risk Matrix (LLM)",
            "閹懏娅欓幒銊︾川(LLM)": "Scenario Analysis (LLM)",
        }
        for bad, good in replacements.items():
            text = text.replace(bad, good)
        return text

    def _sanitize_report_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Ensure report payload is readable and contains degrade metadata."""
        body = dict(payload)
        body["markdown"] = self._sanitize_report_text(str(body.get("markdown", "")))

        sections: list[dict[str, Any]] = []
        for row in list(body.get("report_sections", []) or []):
            if not isinstance(row, dict):
                continue
            section = dict(row)
            section["title"] = self._sanitize_report_text(str(section.get("title", "")))
            section["content"] = self._sanitize_report_text(str(section.get("content", "")))
            sections.append(section)
        if sections:
            body["report_sections"] = sections

        module_rows: list[dict[str, Any]] = []
        for row in list(body.get("report_modules", []) or []):
            if not isinstance(row, dict):
                continue
            item = dict(row)
            item["title"] = self._sanitize_report_text(str(item.get("title", "")))
            item["content"] = self._sanitize_report_text(str(item.get("content", "")))
            module_rows.append(item)
        if module_rows:
            body["report_modules"] = self._normalize_report_modules(module_rows)
        body["evidence_refs"] = self._normalize_report_evidence_refs(
            [dict(row) for row in list(body.get("evidence_refs", []) or []) if isinstance(row, dict)]
        )

        node_rows: list[dict[str, Any]] = []
        for row in list(body.get("analysis_nodes", []) or []):
            if isinstance(row, dict):
                node_rows.append(dict(row))
        body["analysis_nodes"] = self._normalize_report_analysis_nodes(node_rows)

        role_opinions: list[dict[str, Any]] = []
        for row in list(body.get("role_opinions", []) or []):
            if not isinstance(row, dict):
                continue
            role_opinions.append(
                self._normalize_deep_opinion(
                    agent=str(row.get("agent", "unknown")),
                    signal=str(row.get("signal", "hold")),
                    confidence=self._safe_float(row.get("confidence", 0.5), default=0.5),
                    reason=self._sanitize_report_text(str(row.get("reason", "")).strip()),
                    evidence_ids=[str(x).strip() for x in list(row.get("evidence_ids", []) or []) if str(x).strip()],
                    risk_tags=[str(x).strip() for x in list(row.get("risk_tags", []) or []) if str(x).strip()],
                )
            )
        if role_opinions:
            body["role_opinions"] = role_opinions

        multi_role = dict(body.get("multi_role_decision", {}) or {})
        if multi_role:
            multi_role["enabled"] = bool(multi_role.get("enabled", bool(role_opinions)))
            multi_role["trace_id"] = str(multi_role.get("trace_id", body.get("multi_role_trace_id", ""))).strip()
            multi_role["debate_mode"] = str(multi_role.get("debate_mode", "rule_fallback")).strip()
            multi_role["opinions"] = role_opinions or [
                self._normalize_deep_opinion(
                    agent=str(row.get("agent", "unknown")),
                    signal=str(row.get("signal", "hold")),
                    confidence=self._safe_float(row.get("confidence", 0.5), default=0.5),
                    reason=self._sanitize_report_text(str(row.get("reason", "")).strip()),
                    evidence_ids=[str(x).strip() for x in list(row.get("evidence_ids", []) or []) if str(x).strip()],
                    risk_tags=[str(x).strip() for x in list(row.get("risk_tags", []) or []) if str(x).strip()],
                )
                for row in list(multi_role.get("opinions", []) or [])
                if isinstance(row, dict)
            ]
            multi_role["consensus_signal"] = self._normalize_report_signal(
                str(multi_role.get("consensus_signal", body.get("consensus_signal", "hold")))
            )
            multi_role["consensus_confidence"] = round(
                max(0.0, min(1.0, self._safe_float(multi_role.get("consensus_confidence", body.get("consensus_confidence", 0.5)), default=0.5))),
                4,
            )
            multi_role["disagreement_score"] = round(
                max(0.0, min(1.0, self._safe_float(multi_role.get("disagreement_score", 0.0), default=0.0))),
                4,
            )
            multi_role["conflict_sources"] = [
                str(x).strip() for x in list(multi_role.get("conflict_sources", body.get("conflict_sources", [])) or []) if str(x).strip()
            ]
            multi_role["counter_view"] = self._sanitize_report_text(str(multi_role.get("counter_view", "")).strip())[:320]
            multi_role["judge_summary"] = self._sanitize_report_text(
                str(multi_role.get("judge_summary", body.get("judge_summary", ""))).strip()
            )[:320]
            body["multi_role_decision"] = multi_role
            # Keep top-level compatibility fields so existing frontends can consume report output directly.
            body["multi_role_enabled"] = bool(multi_role.get("enabled", False))
            body["multi_role_trace_id"] = str(multi_role.get("trace_id", ""))
            body["role_opinions"] = list(multi_role.get("opinions", []) or [])
            body["judge_summary"] = str(multi_role.get("judge_summary", ""))
            body["conflict_sources"] = list(multi_role.get("conflict_sources", []) or [])
            body["consensus_signal"] = str(multi_role.get("consensus_signal", "hold"))
            body["consensus_confidence"] = float(multi_role.get("consensus_confidence", 0.5) or 0.5)

        committee = dict(body.get("committee", {}) or {})
        if committee:
            body["committee"] = {
                "research_note": self._sanitize_report_text(str(committee.get("research_note", "")).strip()),
                "risk_note": self._sanitize_report_text(str(committee.get("risk_note", "")).strip()),
            }

        final_decision = dict(body.get("final_decision", {}) or {})
        if final_decision:
            final_decision["signal"] = self._normalize_report_signal(str(final_decision.get("signal", "hold")))
            final_decision["confidence"] = round(
                max(0.0, min(1.0, self._safe_float(final_decision.get("confidence", 0.5), default=0.5))),
                4,
            )
            final_decision["rationale"] = self._sanitize_report_text(str(final_decision.get("rationale", "")).strip())
            target_price = dict(final_decision.get("target_price", {}) or {})
            final_decision["target_price"] = {
                "low": round(self._safe_float(target_price.get("low", 0.0), default=0.0), 3),
                "base": round(self._safe_float(target_price.get("base", 0.0), default=0.0), 3),
                "high": round(self._safe_float(target_price.get("high", 0.0), default=0.0), 3),
            }
            final_decision["risk_score"] = round(
                max(0.0, min(100.0, self._safe_float(final_decision.get("risk_score", 0.0), default=0.0))),
                2,
            )
            final_decision["reward_risk_ratio"] = round(
                max(0.0, min(10.0, self._safe_float(final_decision.get("reward_risk_ratio", 0.0), default=0.0))),
                3,
            )
            final_decision["position_sizing_hint"] = self._sanitize_report_text(
                str(final_decision.get("position_sizing_hint", "")).strip()
            )[:80]
            if isinstance(final_decision.get("execution_plan"), list):
                final_decision["execution_plan"] = [
                    self._sanitize_report_text(str(x).strip())
                    for x in list(final_decision.get("execution_plan", []))
                    if str(x).strip()
                ]
            if isinstance(final_decision.get("invalidation_conditions"), list):
                final_decision["invalidation_conditions"] = [
                    self._sanitize_report_text(str(x).strip())
                    for x in list(final_decision.get("invalidation_conditions", []))
                    if str(x).strip()
                ]
            body["final_decision"] = final_decision

        degrade = body.get("degrade")
        if not isinstance(degrade, dict):
            reasons = list((body.get("quality_gate", {}) or {}).get("reasons", []) or [])
            quality_status = str((body.get("quality_gate", {}) or {}).get("status", "pass")).strip().lower() or "pass"
            degrade_code = ""
            if quality_status == "degraded":
                degrade_code = "quality_degraded"
            elif quality_status == "watch":
                degrade_code = "quality_watch"
            quality_explain = self._build_quality_explain(
                reasons=[str(x) for x in reasons],
                quality_status=quality_status,
                context="report",
            )
            degrade = {
                "active": bool(reasons),
                "code": degrade_code if reasons else "",
                "severity": "high" if quality_status == "degraded" else "low" if quality_status == "watch" else "low",
                "reasons": reasons,
                "missing_data": reasons,
                "confidence_penalty": round(min(0.7, 0.15 * float(len(reasons))), 4),
                "user_message": str(quality_explain.get("summary", "")),
                "explain": quality_explain,
            }
        else:
            quality_status = str((body.get("quality_gate", {}) or {}).get("status", degrade.get("severity", "pass"))).strip().lower() or "pass"
            reason_rows = [str(x) for x in list(degrade.get("reasons", []) or []) if str(x).strip()]
            quality_explain = degrade.get("explain")
            if not isinstance(quality_explain, dict):
                quality_explain = self._build_quality_explain(
                    reasons=reason_rows,
                    quality_status=quality_status,
                    context="report",
                )
            degrade["explain"] = quality_explain
            degrade["user_message"] = str(degrade.get("user_message", "")).strip() or str(quality_explain.get("summary", ""))
        body["degrade"] = degrade
        body["quality_dashboard"] = self._build_report_quality_dashboard(
            report_modules=list(body.get("report_modules", []) or []),
            quality_gate=dict(body.get("quality_gate", {}) or {}),
            final_decision=dict(body.get("final_decision", {}) or {}),
            analysis_nodes=list(body.get("analysis_nodes", []) or []),
            evidence_refs=list(body.get("evidence_refs", []) or []),
        )
        body["schema_version"] = str(body.get("schema_version", self._report_bundle_schema_version) or self._report_bundle_schema_version)
        if "result_level" not in body:
            body["result_level"] = "full"
        return body

    def _build_report_task_partial_result(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Build a minimal usable report result for async task partial_ready state."""
        req = ReportRequest(**payload)
        code = str(req.stock_code).strip().upper()
        partial_pack = self._build_llm_input_pack(code, question=f"report_partial:{code}", scenario="report")
        query_result = self.query(
            {
                "user_id": req.user_id,
                "question": f"请先给出 {req.stock_code} {req.period} 的核心结论与证据摘要",
                "stock_codes": [code],
            }
        )
        citations = list(query_result.get("citations", []) or [])
        citation_lines = "\n".join(
            f"- {str(item.get('source_id', ''))}: {str(item.get('excerpt', ''))}"
            for item in citations[:8]
        )
        markdown = (
            f"# {str(req.stock_code).strip().upper()} Analysis Report (Partial)\n\n"
            f"## Summary\n{str(query_result.get('answer', '')).strip() or 'No summary available'}\n\n"
            "## Evidence\n"
            + (citation_lines or "- No evidence")
            + "\n\n## Note\nThis is a minimum viable report. Full report is still running."
        )
        result = {
            "report_id": f"partial-{uuid.uuid4().hex[:10]}",
            "schema_version": self._report_bundle_schema_version,
            "trace_id": str(query_result.get("trace_id", "")),
            "stock_code": code,
            "report_type": str(req.report_type),
            "markdown": markdown,
            "citations": citations,
            "evidence_refs": [self._build_report_evidence_ref(dict(item)) for item in citations[:10]],
            "report_modules": [
                {
                    "module_id": "executive_summary",
                    "title": "执行摘要",
                    "content": str(query_result.get("answer", "")).strip() or "暂无摘要",
                    "evidence_refs": [str(item.get("source_id", "")) for item in citations[:6] if str(item.get("source_id", "")).strip()],
                    "coverage": {"status": "partial", "data_points": int(len(citations))},
                    "confidence": 0.52,
                    "degrade_reason": ["partial_result"],
                },
                {
                    "module_id": "risk_matrix",
                    "title": "风险矩阵",
                    "content": json.dumps(
                        [
                            {
                                "risk": "partial_result",
                                "signal": "warning",
                                "detail": "完整报告尚未完成，当前仅返回最小可用结果。",
                            }
                        ],
                        ensure_ascii=False,
                    ),
                    "evidence_refs": [],
                    "coverage": {"status": "partial", "data_points": 1},
                    "confidence": 0.45,
                    "degrade_reason": ["partial_result"],
                },
                {
                    "module_id": "execution_plan",
                    "title": "执行策略",
                    "content": "- 等待 full 报告完成后再做仓位动作。\n- 先查看数据缺口与质量门控说明。",
                    "evidence_refs": [],
                    "coverage": {"status": "partial", "data_points": 1},
                    "confidence": 0.42,
                    "degrade_reason": ["partial_result"],
                },
            ],
            "committee": {
                "research_note": "研究汇总：当前为 partial 结果，结论仅供预览。",
                "risk_note": "风险仲裁：完整风险矩阵尚未生成，暂不建议执行重仓动作。",
            },
            "final_decision": {
                "signal": "hold",
                "confidence": 0.45,
                "rationale": "当前任务处于 partial 阶段，证据和模块尚未完整，暂保持观望。",
                "invalidation_conditions": ["full_report_ready"],
                "execution_plan": ["等待完整报告结果", "优先核验数据缺口后再决策"],
            },
            "analysis_nodes": [
                {
                    "node_id": "research_summarizer",
                    "title": "研究汇总器",
                    "status": "degraded",
                    "signal": "hold",
                    "confidence": 0.45,
                    "summary": "研究汇总器：当前仅有 partial 证据，结论仅供预览。",
                    "highlights": ["等待 full 报告补齐核心证据后再更新结论。"],
                    "evidence_refs": [str(item.get("source_id", "")) for item in citations[:4] if str(item.get("source_id", "")).strip()],
                    "coverage": {"citation_count": int(len(citations))},
                    "degrade_reason": ["partial_result"],
                    "guardrails": [],
                    "veto": False,
                },
                {
                    "node_id": "risk_arbiter",
                    "title": "风险仲裁器",
                    "status": "ready",
                    "signal": "warning",
                    "confidence": 0.52,
                    "summary": "风险仲裁器：partial 阶段默认保持风控优先，不建议执行重仓动作。",
                    "highlights": ["完整风险矩阵未生成前，维持观望仓位。"],
                    "evidence_refs": [],
                    "coverage": {"risk_watch_count": 0, "quality_status": "degraded"},
                    "degrade_reason": ["partial_result"],
                    "guardrails": ["等待 full 报告后再执行仓位动作。"],
                    "veto": True,
                },
            ],
            "metric_snapshot": {
                "history_sample_size": 0,
                "news_count": 0,
                "research_count": 0,
                "macro_count": 0,
                "quality_score": 0.55,
                "citation_count": int(len(citations)),
                "predict_quality": "pending",
            },
            "quality_gate": {
                "status": "degraded",
                "score": 0.55,
                "reasons": ["partial_result"],
            },
            "quality_dashboard": {
                "status": "degraded",
                "overall_score": 0.48,
                "module_count": 3,
                "avg_module_quality": 0.43,
                "min_module_quality": 0.38,
                "coverage_ratio": 0.0,
                "evidence_ref_count": int(len(citations)),
                "evidence_density": round(float(len(citations)) / 3.0, 4) if citations else 0.0,
                "consistency_score": 0.72,
                "low_quality_modules": ["executive_summary", "risk_matrix", "execution_plan"],
                "reasons": ["partial_result"],
                "node_veto": True,
            },
            "report_data_pack_summary": {
                "as_of": datetime.now(timezone.utc).isoformat(),
                "history_sample_size": 0,
                "predict_quality": "pending",
                "predict_degrade_reasons": ["pending_full_pipeline"],
                "intel_signal": "pending",
                "intel_confidence": 0.0,
                "news_count": 0,
                "research_count": 0,
                "macro_count": 0,
                "missing_data": list(partial_pack.get("missing_data", []) or []),
            },
            "generation_mode": "partial",
            "generation_error": "",
            "degrade": {
                "active": True,
                "code": "partial_result",
                "reasons": ["partial_result"],
                "missing_data": ["full_report_pending"],
                "confidence_penalty": 0.45,
                "user_message": "Minimum viable report returned. Full report is still being generated.",
                "explain": self._build_quality_explain(
                    reasons=["partial_result"],
                    quality_status="degraded",
                    context="report_task",
                ),
            },
            "result_level": "partial",
            "stage_progress": {"stage": "partial_ready", "progress": 0.45},
        }
        return self._sanitize_report_payload(result)

    def _build_report_task_failed_result(self, payload: dict[str, Any], *, error_code: str, error_message: str) -> dict[str, Any]:
        """Build a lightweight fallback result when async task terminates before full report."""
        stock_code = str(payload.get("stock_code", "")).strip().upper() or "UNKNOWN"
        report_type = str(payload.get("report_type", "research")).strip() or "research"
        reason_code = str(error_code or "report_task_failed").strip() or "report_task_failed"
        quality_explain = self._build_quality_explain(
            reasons=[reason_code],
            quality_status="degraded",
            context="report_task",
        )
        next_actions = list(quality_explain.get("actions", []) or [])
        markdown = (
            f"# {stock_code} Analysis Report (Task Failed)\n\n"
            "## Status\n"
            f"- error_code: {reason_code}\n"
            f"- message: {str(error_message or 'report task failed')[:260]}\n\n"
            "## Recovery Actions\n"
            + ("\n".join(f"- {line}" for line in next_actions[:5]) if next_actions else "- Retry task after checking backend health.")
        )
        result = {
            "report_id": f"failed-{uuid.uuid4().hex[:10]}",
            "schema_version": self._report_bundle_schema_version,
            "trace_id": "",
            "stock_code": stock_code,
            "report_type": report_type,
            "markdown": markdown,
            "citations": [],
            "evidence_refs": [],
            "report_modules": [
                {
                    "module_id": "executive_summary",
                    "title": "执行摘要（失败降级）",
                    "content": f"任务执行失败，已返回降级结果。错误码：{reason_code}。",
                    "evidence_refs": [],
                    "coverage": {"status": "partial", "data_points": 0},
                    "confidence": 0.25,
                    "degrade_reason": [reason_code],
                },
                {
                    "module_id": "execution_plan",
                    "title": "补救策略",
                    "content": "\n".join(f"- {line}" for line in next_actions[:6]) or "- 检查后端状态并重试任务。",
                    "evidence_refs": [],
                    "coverage": {"status": "partial", "data_points": int(len(next_actions))},
                    "confidence": 0.22,
                    "degrade_reason": [reason_code],
                },
            ],
            "committee": {
                "research_note": "研究汇总：任务失败，当前为降级结果，仅用于快速定位问题。",
                "risk_note": "风险仲裁：不建议依据该结果直接执行仓位动作。",
            },
            "final_decision": {
                "signal": "hold",
                "confidence": 0.2,
                "rationale": f"任务失败（{reason_code}），完整报告未生成。",
                "invalidation_conditions": ["task_failed", "full_report_missing"],
                "execution_plan": [
                    "检查任务错误与后端状态",
                    "缩小输入范围后重试",
                    "等待完整报告后再决策",
                ],
            },
            "analysis_nodes": [
                {
                    "node_id": "task_guard",
                    "title": "任务守卫",
                    "status": "degraded",
                    "signal": "hold",
                    "confidence": 0.25,
                    "summary": f"任务因 {reason_code} 终止，返回降级结果。",
                    "highlights": [str(error_message or "")[:180]],
                    "evidence_refs": [],
                    "coverage": {"status": "partial"},
                    "degrade_reason": [reason_code],
                    "guardrails": ["仅作排障参考，不用于执行交易。"],
                    "veto": True,
                }
            ],
            "metric_snapshot": {
                "history_sample_size": 0,
                "news_count": 0,
                "research_count": 0,
                "macro_count": 0,
                "quality_score": 0.2,
                "citation_count": 0,
                "predict_quality": "failed",
            },
            "quality_gate": {
                "status": "degraded",
                "score": 0.2,
                "reasons": [reason_code],
            },
            "quality_dashboard": {
                "status": "degraded",
                "overall_score": 0.2,
                "module_count": 2,
                "avg_module_quality": 0.22,
                "min_module_quality": 0.2,
                "coverage_ratio": 0.0,
                "evidence_ref_count": 0,
                "evidence_density": 0.0,
                "consistency_score": 0.5,
                "low_quality_modules": ["executive_summary", "execution_plan"],
                "reasons": [reason_code],
                "node_veto": True,
            },
            "report_data_pack_summary": {
                "as_of": datetime.now(timezone.utc).isoformat(),
                "history_sample_size": 0,
                "predict_quality": "failed",
                "predict_degrade_reasons": [reason_code],
                "intel_signal": "pending",
                "intel_confidence": 0.0,
                "news_count": 0,
                "research_count": 0,
                "macro_count": 0,
                "missing_data": [reason_code],
                "quality_explain_summary": str(quality_explain.get("summary", "")),
            },
            "generation_mode": "failed_fallback",
            "generation_error": str(error_message or "")[:260],
            "degrade": {
                "active": True,
                "code": reason_code,
                "severity": "high",
                "reasons": [reason_code],
                "missing_data": [reason_code],
                "confidence_penalty": 0.7,
                "user_message": str(quality_explain.get("summary", "")),
                "explain": quality_explain,
            },
            "result_level": "partial",
            "stage_progress": {"stage": "failed_fallback", "progress": 1.0},
        }
        return self._sanitize_report_payload(result)

    def _report_task_mark_failed(self, task: dict[str, Any], *, error_code: str, error_message: str) -> None:
        """Mark one report task as failed with a normalized error payload."""
        if not isinstance(task.get("result_partial"), dict):
            payload = dict(task.get("payload", {}) or {})
            task["result_partial"] = self._build_report_task_failed_result(
                payload,
                error_code=error_code,
                error_message=error_message,
            )
        now_iso = datetime.now(timezone.utc).isoformat()
        task["status"] = "failed"
        task["current_stage"] = "failed"
        task["stage_message"] = "Report task failed"
        task["error_code"] = str(error_code or "report_task_failed")
        task["error_message"] = str(error_message or "report_task_failed")[:260]
        task["updated_at"] = now_iso
        task["heartbeat_at"] = now_iso
        task["completed_at"] = now_iso

    def _report_task_apply_runtime_guard(self, task: dict[str, Any]) -> None:
        """Apply timeout/stall guard in-place before task snapshot/result is returned."""
        status = str(task.get("status", "")).strip().lower()
        if status in {"completed", "failed", "cancelled"}:
            return
        now = datetime.now(timezone.utc)

        deadline_raw = str(task.get("deadline_at", "")).strip()
        if deadline_raw:
            deadline = self._parse_time(deadline_raw)
            if now > deadline:
                self._report_task_mark_failed(
                    task,
                    error_code="report_task_timeout",
                    error_message="Report task exceeded runtime limit; please retry with fewer inputs.",
                )
                return

        # If worker heartbeat disappears for too long, fail fast so frontend does not spin forever.
        heartbeat_raw = str(task.get("heartbeat_at", "")).strip() or str(task.get("updated_at", "")).strip()
        if heartbeat_raw and status in {"running", "partial_ready"}:
            heartbeat_at = self._parse_time(heartbeat_raw)
            if (now - heartbeat_at).total_seconds() > float(self._report_task_stall_seconds):
                self._report_task_mark_failed(
                    task,
                    error_code="report_task_stalled",
                    error_message="Report task heartbeat stalled; please retry.",
                )

    def _report_task_tick_heartbeat(
        self,
        task: dict[str, Any],
        *,
        progress_floor: float = 0.0,
        stage_message: str | None = None,
    ) -> None:
        """Touch heartbeat and keep progress moving while long-running stage is executing."""
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        task["updated_at"] = now_iso
        task["heartbeat_at"] = now_iso
        if stage_message is not None:
            task["stage_message"] = stage_message
        if progress_floor > 0:
            task["progress"] = max(float(task.get("progress", 0.0) or 0.0), float(progress_floor))

    def _report_task_snapshot(self, task: dict[str, Any]) -> dict[str, Any]:
        """Project internal task state to API-safe payload."""
        self._report_task_apply_runtime_guard(task)
        has_full = bool(task.get("result_full"))
        has_partial = bool(task.get("result_partial"))
        task_status = str(task.get("status", "queued"))
        level = "full" if has_full else "partial" if has_partial else "none"
        # Failed task with fallback partial result should still be display-ready.
        display_ready = bool(has_full or (task_status == "failed" and has_partial))
        partial_reason = ""
        if has_partial and not has_full:
            partial_reason = "failed_fallback" if task_status == "failed" else "warming_up"
        best_result = task.get("result_full") if has_full else task.get("result_partial") if has_partial else {}
        best_result = best_result if isinstance(best_result, dict) else {}
        pack_summary = dict(best_result.get("report_data_pack_summary", {}) or {})
        missing_data = [str(x) for x in list(pack_summary.get("missing_data", []) or []) if str(x).strip()]
        data_pack_status = "ready"
        if not best_result:
            data_pack_status = "failed" if str(task.get("status", "")) == "failed" else "partial"
        elif missing_data:
            data_pack_status = "partial"
        stage_started_at = str(task.get("stage_started_at", "")).strip()
        heartbeat_at = str(task.get("heartbeat_at", "")).strip()
        stage_elapsed_seconds = 0
        heartbeat_age_seconds = 0
        now = datetime.now(timezone.utc)
        if stage_started_at:
            stage_elapsed_seconds = max(0, int((now - self._parse_time(stage_started_at)).total_seconds()))
        if heartbeat_at:
            heartbeat_age_seconds = max(0, int((now - self._parse_time(heartbeat_at)).total_seconds()))
        return {
            "task_id": str(task.get("task_id", "")),
            "status": task_status,
            "progress": round(max(0.0, min(1.0, float(task.get("progress", 0.0) or 0.0))), 4),
            "current_stage": str(task.get("current_stage", "")),
            "stage_message": str(task.get("stage_message", "")),
            "created_at": str(task.get("created_at", "")),
            "updated_at": str(task.get("updated_at", "")),
            "started_at": str(task.get("started_at", "")),
            "completed_at": str(task.get("completed_at", "")),
            "deadline_at": str(task.get("deadline_at", "")),
            "heartbeat_at": heartbeat_at,
            "stage_started_at": stage_started_at,
            "stage_elapsed_seconds": int(stage_elapsed_seconds),
            "heartbeat_age_seconds": int(heartbeat_age_seconds),
            "result_level": level,
            "has_partial_result": has_partial,
            "has_full_result": has_full,
            "display_ready": display_ready,
            "partial_reason": partial_reason,
            "error_code": str(task.get("error_code", "")),
            "error_message": str(task.get("error_message", "")),
            "data_pack_status": data_pack_status,
            "data_pack_missing": missing_data,
            "quality_gate_detail": dict(best_result.get("quality_gate", {}) or {}),
            "report_quality_dashboard": dict(best_result.get("quality_dashboard", {}) or {}),
        }

    def _run_report_task(self, task_id: str) -> None:
        """Background worker for async report tasks."""
        with self._report_task_lock:
            task = self._report_tasks.get(task_id)
            if not task:
                return
            if bool(task.get("cancel_requested", False)):
                now_iso = datetime.now(timezone.utc).isoformat()
                task["status"] = "cancelled"
                task["updated_at"] = now_iso
                task["heartbeat_at"] = now_iso
                task["completed_at"] = now_iso
                task["stage_message"] = "Task was cancelled before start"
                return
            now_iso = datetime.now(timezone.utc).isoformat()
            task["status"] = "running"
            task["started_at"] = now_iso
            task["updated_at"] = now_iso
            task["heartbeat_at"] = now_iso
            task["stage_started_at"] = now_iso
            task["current_stage"] = "partial"
            task["stage_message"] = "Generating minimum viable result"
            task["progress"] = max(float(task.get("progress", 0.0) or 0.0), 0.1)

        try:
            with self._report_task_lock:
                task_payload = dict(self._report_tasks.get(task_id, {}).get("payload", {}))

            partial = self._build_report_task_partial_result(task_payload)
            with self._report_task_lock:
                current = self._report_tasks.get(task_id)
                if not current:
                    return
                self._report_task_apply_runtime_guard(current)
                if str(current.get("status", "")) == "failed":
                    return
                if bool(current.get("cancel_requested", False)):
                    now = datetime.now(timezone.utc).isoformat()
                    current["status"] = "cancelled"
                    current["updated_at"] = now
                    current["heartbeat_at"] = now
                    current["completed_at"] = now
                    current["stage_message"] = "Task cancelled"
                    return
                current["status"] = "partial_ready"
                current["current_stage"] = "full_report"
                current["stage_message"] = "Partial result ready; generating full report"
                current["progress"] = max(float(current.get("progress", 0.0) or 0.0), 0.45)
                current["result_partial"] = partial
                now_iso = datetime.now(timezone.utc).isoformat()
                current["updated_at"] = now_iso
                current["heartbeat_at"] = now_iso
                current["stage_started_at"] = now_iso

            keepalive_stop = threading.Event()

            def _keepalive_loop() -> None:
                while not keepalive_stop.wait(self._report_task_heartbeat_interval_seconds):
                    with self._report_task_lock:
                        running = self._report_tasks.get(task_id)
                        if not running:
                            return
                        self._report_task_apply_runtime_guard(running)
                        if str(running.get("status", "")) in {"completed", "failed", "cancelled"}:
                            return
                        if bool(running.get("cancel_requested", False)):
                            return
                        stage_started = str(running.get("stage_started_at", "")).strip()
                        elapsed = 0
                        if stage_started:
                            elapsed = max(0, int((datetime.now(timezone.utc) - self._parse_time(stage_started)).total_seconds()))
                        # Keep users informed that full generation is still progressing.
                        dynamic_progress = min(0.92, 0.45 + min(0.42, float(elapsed) / 150.0))
                        self._report_task_tick_heartbeat(
                            running,
                            progress_floor=dynamic_progress,
                            stage_message=f"Generating full report ({elapsed}s)",
                        )

            keepalive_thread = threading.Thread(target=_keepalive_loop, daemon=True)
            keepalive_thread.start()
            try:
                full = self.report_generate(task_payload)
            finally:
                keepalive_stop.set()
                keepalive_thread.join(timeout=1.0)

            with self._report_task_lock:
                current = self._report_tasks.get(task_id)
                if not current:
                    return
                self._report_task_apply_runtime_guard(current)
                if str(current.get("status", "")) == "failed":
                    return
                if bool(current.get("cancel_requested", False)):
                    now = datetime.now(timezone.utc).isoformat()
                    current["status"] = "cancelled"
                    current["updated_at"] = now
                    current["heartbeat_at"] = now
                    current["completed_at"] = now
                    current["stage_message"] = "Task cancelled"
                    return
                current["status"] = "completed"
                current["current_stage"] = "done"
                current["stage_message"] = "Report generation completed"
                current["progress"] = 1.0
                current["result_full"] = self._sanitize_report_payload(full if isinstance(full, dict) else {})
                if not current.get("result_partial"):
                    current["result_partial"] = current["result_full"]
                now = datetime.now(timezone.utc).isoformat()
                current["updated_at"] = now
                current["heartbeat_at"] = now
                current["stage_started_at"] = now
                current["completed_at"] = now
        except Exception as ex:  # noqa: BLE001
            with self._report_task_lock:
                current = self._report_tasks.get(task_id)
                if not current:
                    return
                self._report_task_mark_failed(
                    current,
                    error_code="report_task_failed",
                    error_message=str(ex),
                )

    def report_task_create(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create async report task and return status immediately."""
        req = ReportRequest(**payload)
        task_id = f"rpt-{uuid.uuid4().hex[:12]}"
        now_iso = datetime.now(timezone.utc).isoformat()
        deadline_iso = (datetime.now(timezone.utc) + timedelta(seconds=max(60, int(self._report_task_timeout_seconds)))).isoformat()
        task_payload = {
            "user_id": str(req.user_id),
            "stock_code": str(req.stock_code).strip().upper(),
            "period": str(req.period),
            "report_type": str(req.report_type),
            "run_id": str(payload.get("run_id", "")).strip(),
            "pool_snapshot_id": str(payload.get("pool_snapshot_id", "")).strip(),
            "template_id": str(payload.get("template_id", "default")).strip() or "default",
        }
        token = str(payload.get("token", "")).strip()
        if token:
            task_payload["token"] = token

        with self._report_task_lock:
            self._report_tasks[task_id] = {
                "task_id": task_id,
                "status": "queued",
                "progress": 0.0,
                "current_stage": "queued",
                "stage_message": "Task queued",
                "created_at": now_iso,
                "updated_at": now_iso,
                "heartbeat_at": now_iso,
                "deadline_at": deadline_iso,
                "stage_started_at": now_iso,
                "started_at": "",
                "completed_at": "",
                "error_code": "",
                "error_message": "",
                "cancel_requested": False,
                "payload": task_payload,
                "result_partial": None,
                "result_full": None,
            }
        self._report_task_executor.submit(self._run_report_task, task_id)
        return self.report_task_get(task_id)

    def report_task_get(self, task_id: str) -> dict[str, Any]:
        """Get async report task status."""
        key = str(task_id or "").strip()
        if not key:
            return {"error": "not_found", "task_id": key}
        with self._report_task_lock:
            task = self._report_tasks.get(key)
            if not task:
                return {"error": "not_found", "task_id": key}
            return self._report_task_snapshot(task)

    def report_task_result(self, task_id: str) -> dict[str, Any]:
        """Get best available report result for one task."""
        key = str(task_id or "").strip()
        if not key:
            return {"error": "not_found", "task_id": key}
        with self._report_task_lock:
            task = self._report_tasks.get(key)
            if not task:
                return {"error": "not_found", "task_id": key}
            self._report_task_apply_runtime_guard(task)
            full = task.get("result_full")
            partial = task.get("result_partial")
            if isinstance(full, dict):
                return {
                    "task_id": key,
                    "status": str(task.get("status", "")),
                    "result_level": "full",
                    "display_ready": True,
                    "partial_reason": "",
                    "deadline_at": str(task.get("deadline_at", "")),
                    "heartbeat_at": str(task.get("heartbeat_at", "")),
                    "result": full,
                }
            if isinstance(partial, dict):
                failed_fallback = str(task.get("status", "")) == "failed"
                return {
                    "task_id": key,
                    "status": str(task.get("status", "")),
                    "result_level": "partial",
                    "display_ready": bool(failed_fallback),
                    "partial_reason": "failed_fallback" if failed_fallback else "warming_up",
                    "deadline_at": str(task.get("deadline_at", "")),
                    "heartbeat_at": str(task.get("heartbeat_at", "")),
                    "result": partial,
                }
            return {
                "task_id": key,
                "status": str(task.get("status", "")),
                "result_level": "none",
                "display_ready": False,
                "partial_reason": "",
                "deadline_at": str(task.get("deadline_at", "")),
                "heartbeat_at": str(task.get("heartbeat_at", "")),
                "result": None,
            }

    def report_task_cancel(self, task_id: str) -> dict[str, Any]:
        """Cancel async report task."""
        key = str(task_id or "").strip()
        if not key:
            return {"error": "not_found", "task_id": key}
        with self._report_task_lock:
            task = self._report_tasks.get(key)
            if not task:
                return {"error": "not_found", "task_id": key}
            status = str(task.get("status", ""))
            if status in {"completed", "failed", "cancelled"}:
                return self._report_task_snapshot(task)
            task["cancel_requested"] = True
            now_iso = datetime.now(timezone.utc).isoformat()
            task["updated_at"] = now_iso
            task["heartbeat_at"] = now_iso
            if status == "queued":
                task["status"] = "cancelled"
                task["current_stage"] = "cancelled"
                task["stage_message"] = "Task cancelled"
                task["completed_at"] = now_iso
            else:
                task["status"] = "cancelling"
                task["stage_message"] = "Cancel requested; waiting for safe stop"
            return self._report_task_snapshot(task)

    def reports_list(self, token: str) -> list[dict[str, Any]]:
        return self.web.report_list(token)

    def _load_report_payload_from_version_row(self, report_id: str, row: dict[str, Any]) -> dict[str, Any]:
        """Load one persisted report version row into sanitized payload."""
        parsed: dict[str, Any] = {}
        raw_payload = str(row.get("payload_json", "")).strip()
        if raw_payload:
            try:
                data = json.loads(raw_payload)
                if isinstance(data, dict):
                    parsed = dict(data)
            except Exception:
                parsed = {}
        if not parsed:
            parsed = {
                "report_id": report_id,
                "markdown": self._sanitize_report_text(str(row.get("markdown", ""))),
                "report_modules": [],
                "analysis_nodes": [],
                "final_decision": {"signal": "hold", "confidence": 0.5, "rationale": ""},
                "committee": {"research_note": "", "risk_note": ""},
                "quality_gate": {"status": "degraded", "score": 0.5, "reasons": ["legacy_version_without_payload"]},
                "quality_dashboard": {},
            }
        parsed["report_id"] = report_id
        parsed["version"] = int(row.get("version", 0) or 0)
        parsed["created_at"] = str(row.get("created_at", ""))
        if not str(parsed.get("markdown", "")).strip():
            parsed["markdown"] = self._sanitize_report_text(str(row.get("markdown", "")))
        return self._sanitize_report_payload(parsed)

    def _build_report_version_diff_payload(self, *, base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
        """Compute report diff across decision, quality dashboard, modules and analysis nodes."""
        base_decision = dict(base.get("final_decision", {}) or {})
        cand_decision = dict(candidate.get("final_decision", {}) or {})
        base_quality = dict(base.get("quality_dashboard", {}) or {})
        cand_quality = dict(candidate.get("quality_dashboard", {}) or {})

        module_map_base = {
            str(row.get("module_id", "")).strip(): row
            for row in list(base.get("report_modules", []) or [])
            if isinstance(row, dict) and str(row.get("module_id", "")).strip()
        }
        module_map_cand = {
            str(row.get("module_id", "")).strip(): row
            for row in list(candidate.get("report_modules", []) or [])
            if isinstance(row, dict) and str(row.get("module_id", "")).strip()
        }
        module_ids = sorted(set(module_map_base.keys()) | set(module_map_cand.keys()))
        module_deltas: list[dict[str, Any]] = []
        for module_id in module_ids:
            b = dict(module_map_base.get(module_id, {}) or {})
            c = dict(module_map_cand.get(module_id, {}) or {})
            module_deltas.append(
                {
                    "module_id": module_id,
                    "exists_in_base": bool(b),
                    "exists_in_candidate": bool(c),
                    "quality_delta": round(
                        self._safe_float(c.get("module_quality_score", 0.0), default=0.0)
                        - self._safe_float(b.get("module_quality_score", 0.0), default=0.0),
                        4,
                    ),
                    "confidence_delta": round(
                        self._safe_float(c.get("confidence", 0.0), default=0.0)
                        - self._safe_float(b.get("confidence", 0.0), default=0.0),
                        4,
                    ),
                    "coverage_changed": str((b.get("coverage", {}) or {}).get("status", "")) != str((c.get("coverage", {}) or {}).get("status", "")),
                    "degrade_changed": list(b.get("degrade_reason", []) or []) != list(c.get("degrade_reason", []) or []),
                }
            )

        node_map_base = {
            str(row.get("node_id", "")).strip(): row
            for row in list(base.get("analysis_nodes", []) or [])
            if isinstance(row, dict) and str(row.get("node_id", "")).strip()
        }
        node_map_cand = {
            str(row.get("node_id", "")).strip(): row
            for row in list(candidate.get("analysis_nodes", []) or [])
            if isinstance(row, dict) and str(row.get("node_id", "")).strip()
        }
        node_ids = sorted(set(node_map_base.keys()) | set(node_map_cand.keys()))
        node_deltas: list[dict[str, Any]] = []
        for node_id in node_ids:
            b = dict(node_map_base.get(node_id, {}) or {})
            c = dict(node_map_cand.get(node_id, {}) or {})
            node_deltas.append(
                {
                    "node_id": node_id,
                    "exists_in_base": bool(b),
                    "exists_in_candidate": bool(c),
                    "signal_from": str(b.get("signal", "")),
                    "signal_to": str(c.get("signal", "")),
                    "signal_changed": str(b.get("signal", "")) != str(c.get("signal", "")),
                    "confidence_delta": round(
                        self._safe_float(c.get("confidence", 0.0), default=0.0)
                        - self._safe_float(b.get("confidence", 0.0), default=0.0),
                        4,
                    ),
                    "veto_from": bool(b.get("veto", False)),
                    "veto_to": bool(c.get("veto", False)),
                }
            )

        quality_delta = {
            "status_from": str(base_quality.get("status", "")),
            "status_to": str(cand_quality.get("status", "")),
            "status_changed": str(base_quality.get("status", "")) != str(cand_quality.get("status", "")),
            "overall_score_delta": round(
                self._safe_float(cand_quality.get("overall_score", 0.0), default=0.0)
                - self._safe_float(base_quality.get("overall_score", 0.0), default=0.0),
                4,
            ),
            "coverage_ratio_delta": round(
                self._safe_float(cand_quality.get("coverage_ratio", 0.0), default=0.0)
                - self._safe_float(base_quality.get("coverage_ratio", 0.0), default=0.0),
                4,
            ),
            "consistency_score_delta": round(
                self._safe_float(cand_quality.get("consistency_score", 0.0), default=0.0)
                - self._safe_float(base_quality.get("consistency_score", 0.0), default=0.0),
                4,
            ),
            "freshness_score_delta": round(
                self._safe_float(cand_quality.get("evidence_freshness_score", 0.0), default=0.0)
                - self._safe_float(base_quality.get("evidence_freshness_score", 0.0), default=0.0),
                4,
            ),
        }
        decision_delta = {
            "signal_from": str(base_decision.get("signal", "hold")),
            "signal_to": str(cand_decision.get("signal", "hold")),
            "signal_changed": str(base_decision.get("signal", "hold")) != str(cand_decision.get("signal", "hold")),
            "confidence_delta": round(
                self._safe_float(cand_decision.get("confidence", 0.0), default=0.0)
                - self._safe_float(base_decision.get("confidence", 0.0), default=0.0),
                4,
            ),
        }
        summary: list[str] = []
        if quality_delta["status_changed"]:
            summary.append(f"质量状态变化：{quality_delta['status_from']} -> {quality_delta['status_to']}")
        if decision_delta["signal_changed"]:
            summary.append(f"决策信号变化：{decision_delta['signal_from']} -> {decision_delta['signal_to']}")
        top_module_shift = sorted(module_deltas, key=lambda x: abs(self._safe_float(x.get("quality_delta", 0.0), default=0.0)), reverse=True)[:3]
        for row in top_module_shift:
            if abs(self._safe_float(row.get("quality_delta", 0.0), default=0.0)) >= 0.01:
                summary.append(f"模块 {row['module_id']} 质量变化 {self._safe_float(row['quality_delta'], default=0.0):+.2f}")
        if not summary:
            summary.append("版本间结构化差异较小。")
        return {
            "base_version": int(base.get("version", 0) or 0),
            "candidate_version": int(candidate.get("version", 0) or 0),
            "schema_version": str(candidate.get("schema_version", self._report_bundle_schema_version) or self._report_bundle_schema_version),
            "quality_delta": quality_delta,
            "decision_delta": decision_delta,
            "module_deltas": module_deltas,
            "node_deltas": node_deltas,
            "summary": summary[:8],
        }

    def report_versions(self, token: str, report_id: str) -> list[dict[str, Any]]:
        rows = self.web.report_version_rows(token, report_id, limit=50)
        items: list[dict[str, Any]] = []
        for row in rows:
            payload = self._load_report_payload_from_version_row(report_id, row)
            quality = dict(payload.get("quality_dashboard", {}) or {})
            decision = dict(payload.get("final_decision", {}) or {})
            items.append(
                {
                    "version": int(payload.get("version", 0) or 0),
                    "created_at": str(payload.get("created_at", "")),
                    "schema_version": str(payload.get("schema_version", self._report_bundle_schema_version)),
                    "signal": str(decision.get("signal", "hold")),
                    "confidence": float(decision.get("confidence", 0.0) or 0.0),
                    "quality_status": str(quality.get("status", "unknown")),
                    "quality_score": float(quality.get("overall_score", 0.0) or 0.0),
                    "module_count": len(list(payload.get("report_modules", []) or [])),
                    "analysis_node_count": len(list(payload.get("analysis_nodes", []) or [])),
                    "evidence_freshness_score": float(quality.get("evidence_freshness_score", 0.0) or 0.0),
                }
            )
        for idx in range(len(items)):
            if idx + 1 >= len(items):
                items[idx]["delta_vs_prev"] = {}
                continue
            curr = dict(items[idx])
            prev = dict(items[idx + 1])
            items[idx]["delta_vs_prev"] = {
                "quality_score_delta": round(self._safe_float(curr.get("quality_score", 0.0), default=0.0) - self._safe_float(prev.get("quality_score", 0.0), default=0.0), 4),
                "confidence_delta": round(self._safe_float(curr.get("confidence", 0.0), default=0.0) - self._safe_float(prev.get("confidence", 0.0), default=0.0), 4),
                "signal_changed": str(curr.get("signal", "")) != str(prev.get("signal", "")),
                "freshness_score_delta": round(
                    self._safe_float(curr.get("evidence_freshness_score", 0.0), default=0.0)
                    - self._safe_float(prev.get("evidence_freshness_score", 0.0), default=0.0),
                    4,
                ),
            }
        return items

    def report_versions_diff(
        self,
        token: str,
        report_id: str,
        *,
        base_version: int | None = None,
        candidate_version: int | None = None,
    ) -> dict[str, Any]:
        rows = self.web.report_version_rows(token, report_id, limit=80)
        if not rows:
            return {"error": "not_found", "report_id": report_id}
        payload_by_version = {
            int(row.get("version", 0) or 0): self._load_report_payload_from_version_row(report_id, row)
            for row in rows
        }
        versions = sorted(payload_by_version.keys(), reverse=True)
        candidate_v = int(candidate_version or versions[0])
        if candidate_v not in payload_by_version:
            return {"error": "candidate_version_not_found", "report_id": report_id, "candidate_version": candidate_v}
        if base_version is not None:
            base_v = int(base_version)
        else:
            lower = [v for v in versions if v < candidate_v]
            base_v = int(lower[0]) if lower else int(candidate_v)
        if base_v not in payload_by_version:
            return {"error": "base_version_not_found", "report_id": report_id, "base_version": base_v}
        base_payload = payload_by_version[base_v]
        candidate_payload = payload_by_version[candidate_v]
        diff_payload = self._build_report_version_diff_payload(base=base_payload, candidate=candidate_payload)
        return {
            "report_id": report_id,
            "available_versions": versions,
            "diff": diff_payload,
            "base_snapshot": {
                "version": int(base_payload.get("version", 0) or 0),
                "signal": str((base_payload.get("final_decision", {}) or {}).get("signal", "hold")),
                "quality_status": str((base_payload.get("quality_dashboard", {}) or {}).get("status", "unknown")),
            },
            "candidate_snapshot": {
                "version": int(candidate_payload.get("version", 0) or 0),
                "signal": str((candidate_payload.get("final_decision", {}) or {}).get("signal", "hold")),
                "quality_status": str((candidate_payload.get("quality_dashboard", {}) or {}).get("status", "unknown")),
            },
        }

    def report_export(self, token: str, report_id: str, *, format: str = "markdown") -> dict[str, Any]:
        payload = self.web.report_export(token, report_id)
        if "error" in payload:
            return payload
        export_format = str(format or "markdown").strip().lower() or "markdown"
        if export_format not in {"markdown", "module_markdown", "json_bundle"}:
            raise ValueError("format must be one of: markdown, module_markdown, json_bundle")

        # Prefer in-memory enriched payload so export can include modules/quality board.
        cached = self._reports.get(report_id)
        persisted_payload: dict[str, Any] = {}
        raw_payload_json = str(payload.get("payload_json", "")).strip()
        if raw_payload_json:
            try:
                parsed = json.loads(raw_payload_json)
                if isinstance(parsed, dict):
                    persisted_payload = parsed
            except Exception:
                persisted_payload = {}
        report_payload = self._sanitize_report_payload(dict(cached)) if isinstance(cached, dict) else {}
        if not report_payload and persisted_payload:
            report_payload = self._sanitize_report_payload(dict(persisted_payload))
        report_payload["report_id"] = report_id
        report_payload["version"] = int(payload.get("version", 0) or 0)
        if not str(report_payload.get("markdown", "")).strip():
            report_payload["markdown"] = self._sanitize_report_text(str(payload.get("markdown", "")))
        report_payload["schema_version"] = str(
            report_payload.get("schema_version")
            or persisted_payload.get("schema_version")
            or self._report_bundle_schema_version
        )

        module_markdown = self._render_report_module_markdown(report_payload)
        json_bundle = {
            "schema_version": str(report_payload.get("schema_version", self._report_bundle_schema_version)),
            "report_id": report_id,
            "version": report_payload.get("version", 0),
            "stock_code": str(report_payload.get("stock_code", "")),
            "report_type": str(report_payload.get("report_type", "")),
            "final_decision": dict(report_payload.get("final_decision", {}) or {}),
            "committee": dict(report_payload.get("committee", {}) or {}),
            "report_modules": list(report_payload.get("report_modules", []) or []),
            "analysis_nodes": list(report_payload.get("analysis_nodes", []) or []),
            "quality_dashboard": dict(report_payload.get("quality_dashboard", {}) or {}),
            "metric_snapshot": dict(report_payload.get("metric_snapshot", {}) or {}),
            "quality_gate": dict(report_payload.get("quality_gate", {}) or {}),
            "evidence_refs": list(report_payload.get("evidence_refs", []) or []),
        }
        if export_format == "module_markdown":
            content = module_markdown
            media_type = "text/markdown; charset=utf-8"
        elif export_format == "json_bundle":
            content = json.dumps(json_bundle, ensure_ascii=False, indent=2)
            media_type = "application/json; charset=utf-8"
        else:
            content = self._sanitize_report_text(str(report_payload.get("markdown", "")))
            media_type = "text/markdown; charset=utf-8"

        return {
            "report_id": report_id,
            "version": report_payload.get("version", 0),
            "schema_version": str(report_payload.get("schema_version", self._report_bundle_schema_version)),
            "format": export_format,
            "media_type": media_type,
            "content": content,
            # Keep legacy field for older clients that still read markdown directly.
            "markdown": content if export_format != "json_bundle" else self._sanitize_report_text(str(report_payload.get("markdown", ""))),
            "module_markdown": module_markdown,
            "json_bundle": json_bundle,
            "quality_dashboard": dict(report_payload.get("quality_dashboard", {}) or {}),
        }

