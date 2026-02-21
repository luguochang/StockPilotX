from __future__ import annotations

from .shared import *

class RuntimeCoreMixin:
    def _select_runtime(self, preference: str | None = None):
        """Select runtime by request preference: langgraph/direct/auto."""
        pref = (preference or "").strip().lower()
        if pref in ("", "auto", "default"):
            return self.workflow_runtime
        if pref == "direct":
            return build_workflow_runtime(self.workflow, prefer_langgraph=False)
        if pref == "langgraph":
            return build_workflow_runtime(self.workflow, prefer_langgraph=True)
        return self.workflow_runtime

    def _register_default_jobs(self) -> None:
        """Register default scheduler jobs."""

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
        self.scheduler.register(
            JobConfig(
                name="memory_ttl_cleanup",
                cadence="daily",
                # Keep cleanup lightweight and observable for long-running dev/prod runtimes.
                fn=lambda: {"status": "ok", "task": "memory_ttl_cleanup", "deleted": self.memory.cleanup_expired()},
            )
        )

    def _register_default_agent_cards(self) -> None:
        """Register built-in agent cards for A2A capability discovery."""
        cards = [
            ("supervisor_agent", "Supervisor Agent", "Coordinate rounds and output arbitration conclusions.", ["plan", "route", "arbitrate"]),
            ("pm_agent", "PM Agent", "Evaluate theme logic and narrative consistency.", ["theme_analysis", "narrative"]),
            ("quant_agent", "Quant Agent", "Evaluate valuation, expectancy and probabilistic signals.", ["factor_analysis", "probability"]),
            ("risk_agent", "Risk Agent", "Evaluate drawdown, volatility and downside risk.", ["risk_scoring", "drawdown_check"]),
            ("critic_agent", "Critic Agent", "Check evidence completeness and logical consistency.", ["consistency_check", "counter_view"]),
            ("macro_agent", "Macro Agent", "Evaluate macro-policy and global shocks.", ["macro_event", "policy_watch"]),
            ("execution_agent", "Execution Agent", "Evaluate position sizing, timing and execution constraints.", ["execution_plan", "position_sizing"]),
            ("compliance_agent", "Compliance Agent", "Check compliance boundaries and expression risk.", ["compliance_check", "policy_block"]),
        ]
        for agent_id, display_name, description, capabilities in cards:
            self.web.register_agent_card(
                agent_id=agent_id,
                display_name=display_name,
                description=description,
                capabilities=capabilities,
            )

    def _normalize_deep_opinion(
        self,
        *,
        agent: str,
        signal: str,
        confidence: float,
        reason: str,
        evidence_ids: list[str] | None = None,
        risk_tags: list[str] | None = None,
    ) -> dict[str, Any]:
        mapped = signal.strip().lower()
        if mapped not in {"buy", "hold", "reduce"}:
            mapped = "hold"
        return {
            "agent": agent,
            "signal": mapped,
            "confidence": max(0.0, min(1.0, float(confidence))),
            "reason": reason[:360],
            "evidence_ids": evidence_ids or [],
            "risk_tags": risk_tags or [],
        }

    def _arbitrate_opinions(self, opinions: list[dict[str, Any]]) -> dict[str, Any]:
        bucket = {"buy": 0, "hold": 0, "reduce": 0}
        for opinion in opinions:
            bucket[str(opinion.get("signal", "hold"))] += 1

        ranked = sorted(bucket.items(), key=lambda item: item[1], reverse=True)
        consensus_signal = ranked[0][0]
        disagreement_score = round(1.0 - (ranked[0][1] / max(1, len(opinions))), 4)
        unique_signals = {str(opinion.get("signal", "hold")) for opinion in opinions}

        conflict_sources: list[str] = []
        if len(unique_signals) > 1:
            conflict_sources.append("signal_divergence")
        if consensus_signal == "buy" and any(
            str(opinion.get("agent")) == "risk_agent" and str(opinion.get("signal")) == "reduce" for opinion in opinions
        ):
            conflict_sources.append("risk_veto")
        if consensus_signal == "buy" and any(
            str(opinion.get("agent")) == "compliance_agent" and str(opinion.get("signal")) == "reduce" for opinion in opinions
        ):
            conflict_sources.append("compliance_veto")

        confidence_avg = sum(float(opinion.get("confidence", 0.0)) for opinion in opinions) / max(1, len(opinions))
        if confidence_avg < 0.6:
            conflict_sources.append("low_confidence")

        counter_candidates = [
            opinion
            for opinion in opinions
            if str(opinion.get("signal", "hold")) != consensus_signal and str(opinion.get("reason", "")).strip()
        ]
        counter_candidates.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
        counter_view = str(counter_candidates[0]["reason"]) if counter_candidates else "no significant counter view"

        return {
            "consensus_signal": consensus_signal,
            "disagreement_score": disagreement_score,
            "conflict_sources": conflict_sources,
            "counter_view": counter_view,
        }

    def _quality_score_for_group_card(
        self, *, disagreement_score: float, avg_confidence: float, evidence_count: int
    ) -> float:
        return round((1.0 - disagreement_score) * 0.6 + min(1.0, evidence_count / 4.0) * 0.2 + avg_confidence * 0.2, 4)

    def _build_fallback_final_decision(
        self,
        *,
        code: str,
        report_type: str,
        intel: dict[str, Any],
        quality_gate: dict[str, Any],
        quality_reasons: list[str],
    ) -> dict[str, Any]:
        """Build deterministic final decision before optional LLM overlay."""
        signal = self._normalize_report_signal(str(intel.get("overall_signal", "hold")))
        base_conf = self._safe_float(intel.get("confidence", 0.55), default=0.55)
        quality_cap = self._safe_float(quality_gate.get("score", 0.55), default=0.55)
        confidence = max(0.25, min(0.92, min(base_conf, quality_cap)))
        quality_status = str(quality_gate.get("status", "pass")).strip().lower() or "pass"

        key_catalysts = list(intel.get("key_catalysts", []) or [])
        risk_watch = list(intel.get("risk_watch", []) or [])
        catalyst_note = str(key_catalysts[0].get("summary", key_catalysts[0].get("title", ""))).strip() if key_catalysts else ""
        risk_note = str(risk_watch[0].get("summary", risk_watch[0].get("title", ""))).strip() if risk_watch else ""
        rationale_parts = [
            f"{code} 当前综合信号为 `{signal}`，结论受数据质量门控约束。",
            f"报告类型 `{report_type}` 下优先考虑可验证证据与风险触发条件。",
        ]
        if catalyst_note:
            rationale_parts.append(f"主要催化：{catalyst_note[:160]}")
        if risk_note:
            rationale_parts.append(f"主要风险：{risk_note[:160]}")

        invalidation_conditions = [
            "关键风险事件落地偏负面",
            "分歧持续扩大且缺口未补齐",
            "预算或风控阈值触发降级",
        ]
        if quality_reasons:
            invalidation_conditions.append(f"当前数据质量门控处于 {quality_status}，需要先补数再放大仓位")

        if signal == "buy":
            execution_plan = [
                "先小仓位试探，等待二次确认后再加仓。",
                "若 2-3 个交易日内出现放量回撤，暂停加仓。",
            ]
        elif signal == "reduce":
            execution_plan = [
                "优先降低高波动仓位，保留少量观察仓。",
                "若负面事件继续发酵，继续降低风险敞口。",
            ]
        else:
            execution_plan = [
                "维持观察仓，等待新增证据再决策。",
                "优先补齐缺失数据，避免因样本不足误判。",
            ]

        market_snapshot = dict(intel.get("market_snapshot", {}) or {})
        price_now = self._safe_float(market_snapshot.get("price", 0.0), default=0.0)
        risk_level = str(intel.get("risk_level", "medium")).strip().lower() or "medium"
        risk_base = {"low": 34.0, "medium": 55.0, "high": 74.0}.get(risk_level, 55.0)
        # Keep risk score bounded and interpretable (0-100) for UI and export consumers.
        risk_score = max(5.0, min(100.0, risk_base + min(24.0, 6.0 * float(len(quality_reasons)))))
        if signal == "buy":
            risk_score = max(0.0, risk_score - 4.0)
        elif signal == "reduce":
            risk_score = min(100.0, risk_score + 3.0)

        if price_now > 0:
            if signal == "buy":
                target_price = {
                    "low": round(price_now * 1.04, 3),
                    "base": round(price_now * 1.12, 3),
                    "high": round(price_now * 1.18, 3),
                }
            elif signal == "reduce":
                target_price = {
                    "low": round(price_now * 0.88, 3),
                    "base": round(price_now * 0.94, 3),
                    "high": round(price_now * 0.98, 3),
                }
            else:
                target_price = {
                    "low": round(price_now * 0.96, 3),
                    "base": round(price_now * 1.03, 3),
                    "high": round(price_now * 1.08, 3),
                }
        else:
            target_price = {"low": 0.0, "base": 0.0, "high": 0.0}

        upside = max(0.0, float(target_price.get("high", 0.0) - price_now))
        downside = max(0.01, float(price_now - target_price.get("low", 0.0)))
        reward_risk_ratio = round(max(0.1, min(6.0, upside / downside)), 3)
        position_sizing_hint = str(intel.get("position_hint", "")).strip() or (
            "35-60%" if signal == "buy" else "5-15%" if signal == "reduce" else "20-35%"
        )

        return {
            "signal": signal,
            "confidence": round(confidence, 4),
            "rationale": " ".join(rationale_parts),
            "invalidation_conditions": invalidation_conditions,
            "execution_plan": execution_plan,
            "target_price": target_price,
            "risk_score": round(risk_score, 2),
            "reward_risk_ratio": reward_risk_ratio,
            "position_sizing_hint": position_sizing_hint,
        }

    def _build_quality_explain(
        self,
        *,
        reasons: list[str],
        quality_status: str,
        context: str = "report",
    ) -> dict[str, Any]:
        """Build reusable quality explanation cards for report/deep-think degraded paths."""
        profiles: dict[str, dict[str, str]] = {
            "quote_missing": {"title": "实时行情缺失", "impact": "价格与波动判断不完整", "action": "刷新行情源并校验最近交易时刻。"},
            "history_insufficient": {"title": "历史日线不足", "impact": "趋势与回撤判断不稳定", "action": "补齐至少252交易日日线样本。"},
            "history_sample_insufficient": {"title": "历史样本偏少", "impact": "统计稳定性下降", "action": "补齐连续交易日序列后重算指标。"},
            "history_30d_insufficient": {"title": "近30日样本不足", "impact": "短期节奏判断偏弱", "action": "补齐近30日连续日线。"},
            "financial_missing": {"title": "财报快照缺失", "impact": "估值与盈利归因不足", "action": "补齐最新季度财务指标（营收、利润、ROE）。"},
            "announcement_missing": {"title": "公告事件不足", "impact": "事件冲击解释不完整", "action": "补拉最近公告与关键事项。"},
            "news_insufficient": {"title": "新闻样本不足", "impact": "情绪与催化识别偏弱", "action": "补齐近30天有效新闻样本。"},
            "research_insufficient": {"title": "研报样本不足", "impact": "机构观点覆盖不足", "action": "补齐近季度研报摘要与评级变化。"},
            "macro_insufficient": {"title": "宏观指标不足", "impact": "政策/宏观冲击分析受限", "action": "补齐GDP/CPI/PMI等核心指标。"},
            "fund_missing": {"title": "资金面样本不足", "impact": "资金行为判断不完整", "action": "补齐基金/资金流向快照。"},
            "predict_degraded": {"title": "预测模块降级", "impact": "预测信号可信度下降", "action": "补数后重新触发预测模块。"},
            "citations_insufficient": {"title": "证据引用不足", "impact": "结论可核验性下降", "action": "增加可追溯引用并复核来源时效。"},
            "auto_refresh_failed": {"title": "自动补数失败", "impact": "数据缺口未被自动修复", "action": "检查数据源健康并重试补数。"},
            "partial_result": {"title": "仅返回临时结果", "impact": "报告模块尚未完整", "action": "等待 full 报告完成后再决策。"},
            "report_task_timeout": {"title": "任务执行超时", "impact": "完整报告未生成", "action": "缩小输入范围后重试（减少标的/缩短问题）。"},
            "report_task_stalled": {"title": "任务心跳停滞", "impact": "后台执行中断", "action": "重试任务并检查后端运行状态。"},
            "report_task_failed": {"title": "任务执行失败", "impact": "报告流程提前结束", "action": "查看错误信息并重试。"},
        }
        normalized = [str(x).strip() for x in reasons if str(x).strip()]
        dedup = list(dict.fromkeys(normalized))
        items: list[dict[str, str]] = []
        actions: list[str] = []
        for reason in dedup[:10]:
            profile = profiles.get(reason) or {
                "title": "未知质量缺口",
                "impact": "可核验性下降",
                "action": "补齐相关数据后重试。",
            }
            items.append(
                {
                    "reason": reason,
                    "title": str(profile.get("title", "")),
                    "impact": str(profile.get("impact", "")),
                    "action": str(profile.get("action", "")),
                }
            )
            action = str(profile.get("action", "")).strip()
            if action and action not in actions:
                actions.append(action)

        status = str(quality_status or "pass").strip().lower() or "pass"
        if not dedup:
            summary = "质量门控正常，证据覆盖满足当前输出要求。"
        elif status == "watch":
            summary = "存在可恢复的数据缺口，当前结论可用但应在补数后复核。"
        else:
            summary = "存在关键数据缺口，当前结果已降级，建议先补数再执行交易决策。"
        if context == "report_task" and dedup:
            summary = "任务未完成完整报告，已返回降级结果供快速参考；请按建议补数后重试。"

        return {
            "status": status,
            "summary": summary,
            "items": items,
            "actions": actions[:6],
        }

    def _index_text_rows_to_rag(
        self,
        *,
        rows: list[dict[str, Any]],
        namespace: str,
        source_field: str,
        time_field: str,
        content_fields: tuple[str, ...],
        filename_ext: str = "txt",
    ) -> None:
        """Best-effort helper: map structured rows to local RAG docs without blocking ingestion."""

        for row in rows:
            # Stable id dedups repeated ingest calls for the same upstream record.
            signature = (
                str(row.get("source_url", ""))
                + "|"
                + str(row.get("title", ""))
                + "|"
                + str(row.get(time_field, ""))
            )
            doc_id = f"{namespace}-{hashlib.md5(signature.encode('utf-8')).hexdigest()[:16]}"
            if doc_id in self.ingestion.store.docs:
                continue
            content = ""
            for field in content_fields:
                value = str(row.get(field, "")).strip()
                if value:
                    content = value
                    break
            if not content:
                continue
            try:
                self.docs_upload(doc_id, f"{doc_id}.{filename_ext}", content, source=str(row.get(source_field, namespace)))
                self.docs_index(doc_id)
            except Exception:
                # Do not fail ingestion when one row cannot be indexed.
                continue

    @staticmethod
    def _infer_graph_entity_type(entity_id: str) -> str:
        token = str(entity_id or "").strip().upper()
        if token.startswith(("SH", "SZ", "BJ")) and len(token) >= 8:
            return "stock"
        return "concept"

    def _validate_rag_parsed_text(
        self,
        *,
        extracted_text: str,
        parse_note: str,
        parse_quality: dict[str, Any],
    ) -> str | None:
        note = str(parse_note or "").lower()
        if "pdf_binary_stream_detected" in note:
            return "PDF parsing failed: binary stream detected, not human-readable text"
        if "pdf_parser_unavailable" in note:
            return "PDF parsing failed: parser unavailable in runtime, install PDF parser and retry"
        normalized = re.sub(r"\s+", " ", str(extracted_text or "")).strip()
        if not normalized:
            return "No readable text extracted from file"
        if "pdf_ascii_fallback" in note and self._looks_like_pdf_binary_stream_text(normalized):
            return "PDF parsing failed: fallback extracted PDF internals instead of document text"
        quality_score = float(parse_quality.get("quality_score", 0.0) or 0.0)
        if quality_score <= 0.01:
            return "Extracted text quality is too low for indexing"
        return None

    @staticmethod
    def _rag_preview_trim(text: str, max_len: int = 140) -> str:
        """Normalize and truncate preview text to keep payload readable."""
        compact = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(compact) <= max(1, int(max_len)):
            return compact
        return compact[: max(1, int(max_len))].rstrip() + "..."

    def _build_rag_preview_queries(
        self,
        *,
        chunks: list[dict[str, Any]],
        stock_codes: list[str],
        tags: list[str],
        max_queries: int,
    ) -> list[str]:
        """Generate deterministic sample queries for retrieval verification."""
        candidates: list[str] = []
        snippet_queries: list[str] = []

        for row in chunks[:12]:
            text = str(row.get("chunk_text_redacted") or row.get("chunk_text") or "")
            normalized = re.sub(r"\s+", " ", text).strip()
            if not normalized:
                continue
            # Use sentence-like fragments so users can quickly understand preview intent.
            for part in re.split(r"[。！？!?\n]", normalized):
                piece = self._rag_preview_trim(part, max_len=42)
                if len(piece) < 12:
                    continue
                snippet_queries.append(piece)
                break
            if len(snippet_queries) >= max_queries * 2:
                break

        # Put current-document snippets first to maximize "this doc is retrievable" signal.
        candidates.extend(snippet_queries)
        for code in stock_codes[:2]:
            normalized = str(code or "").strip().upper()
            if not normalized:
                continue
            candidates.append(f"{normalized} 鍏抽敭缁撹")
            candidates.append(f"{normalized} 椋庨櫓鎻愮ず")
        for tag in tags[:2]:
            normalized = str(tag or "").strip()
            if not normalized:
                continue
            candidates.append(f"{normalized} 鏍稿績淇℃伅")

        deduped: list[str] = []
        seen: set[str] = set()
        for query in candidates:
            key = query.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(query.strip())
            if len(deduped) >= max_queries:
                break
        return deduped or ["鏂囨。鏍稿績缁撹"]

    @staticmethod
    def _decode_text_bytes(raw_bytes: bytes) -> str:
        for enc in ("utf-8", "gbk", "utf-16", "latin1"):
            try:
                return raw_bytes.decode(enc, errors="ignore")
            except Exception:  # noqa: BLE001
                continue
        return ""

    def _extract_text_from_upload_bytes(self, *, filename: str, raw_bytes: bytes, content_type: str = "") -> tuple[str, str]:
        """Extract text from uploaded bytes; use lightweight fallbacks when third-party parsers are unavailable."""
        ext = ""
        idx = str(filename).rfind(".")
        if idx >= 0:
            ext = str(filename)[idx:].lower()
        note_parts: list[str] = []
        if ext in {".txt", ".md", ".csv", ".json", ".log", ".html", ".htm", ".ts", ".js", ".py"}:
            note_parts.append("plain_text_decode")
            return self._decode_text_bytes(raw_bytes), ",".join(note_parts)
        if ext == ".docx":
            try:
                with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                    xml_bytes = zf.read("word/document.xml")
                root = ET.fromstring(xml_bytes)
                text = " ".join(x.strip() for x in root.itertext() if str(x).strip())
                note_parts.append("docx_xml_extract")
                return text, ",".join(note_parts)
            except Exception:  # noqa: BLE001
                note_parts.append("docx_extract_failed_fallback_decode")
                return self._decode_text_bytes(raw_bytes), ",".join(note_parts)
        if ext in {".xlsx", ".xlsm"} and load_workbook is not None:
            try:
                wb = load_workbook(filename=io.BytesIO(raw_bytes), read_only=True, data_only=True)
                ws = wb.active
                lines: list[str] = []
                max_rows = 1500
                max_cols = 32
                for row_idx, row in enumerate(ws.iter_rows(values_only=True), start=1):
                    if row_idx > max_rows:
                        break
                    cells = [str(c).strip() for c in row[:max_cols] if c is not None and str(c).strip()]
                    if cells:
                        lines.append(" | ".join(cells))
                note_parts.append("xlsx_extract")
                return "\n".join(lines), ",".join(note_parts)
            except Exception:  # noqa: BLE001
                note_parts.append("xlsx_extract_failed_fallback_decode")
                return self._decode_text_bytes(raw_bytes), ",".join(note_parts)
        if ext == ".pdf":
            try:
                import pypdf  # type: ignore
            except Exception:  # noqa: BLE001
                note_parts.append("pdf_parser_unavailable")
            else:
                try:
                    reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
                    pages = [str(page.extract_text() or "") for page in reader.pages]
                    text = "\n".join(x for x in pages if x.strip())
                    if text.strip():
                        note_parts.append("pdf_pypdf_extract")
                        return text, ",".join(note_parts)
                    note_parts.append("pdf_parse_failed")
                except Exception:  # noqa: BLE001
                    # Keep parser availability and parse quality diagnostics separated.
                    note_parts.append("pdf_parse_failed")
            # 鏃?pdf 瑙ｆ瀽搴撴椂鐨勫厹搴曪細灏濊瘯鎶藉彇鍙 ASCII 涓诧紝鑷冲皯淇濈暀閮ㄥ垎鍙绱㈠唴瀹广€?
            ascii_chunks = re.findall(rb"[A-Za-z0-9][A-Za-z0-9 ,.;:%()\-_/]{16,}", raw_bytes)
            decoded = " ".join(x.decode("latin1", errors="ignore") for x in ascii_chunks)
            note_parts.append("pdf_ascii_fallback")
            return decoded, ",".join(note_parts)

        note_parts.append(f"generic_decode:{content_type or 'unknown'}")
        return self._decode_text_bytes(raw_bytes), ",".join(note_parts)

    def _persist_doc_chunks_to_rag(self, doc_id: str, doc: dict[str, Any]) -> None:
        """Persist in-memory document chunks into the web-layer RAG storage."""
        chunks = [str(x) for x in doc.get("chunks", []) if str(x).strip()]
        if not chunks:
            return
        source = str(doc.get("source", "user_upload")).strip().lower() or "user_upload"
        source_url = f"local://docs/{doc_id}"
        # Product requirement: all uploaded/indexed docs become searchable immediately.
        # Source policy is kept for governance metadata, but not used as an activation gate.
        effective_status = "active"
        quality_score = float(doc.get("parse_confidence", 0.0))
        stock_codes = self._extract_stock_codes_from_text(
            f"{doc.get('filename', '')}\n{doc.get('cleaned_text', '')}\n{doc.get('content', '')}"
        )
        payload_chunks: list[dict[str, Any]] = []
        for idx, chunk in enumerate(chunks, start=1):
            payload_chunks.append(
                {
                    "chunk_id": f"{doc_id}-c{idx}",
                    "chunk_no": idx,
                    "chunk_text": chunk,
                    # 鍙岃建涓殑鈥滄憳瑕?鑴辨晱杞ㄢ€濓細鍦ㄧ嚎妫€绱㈤粯璁や娇鐢?redacted 鏂囨湰銆?
                    "chunk_text_redacted": self._redact_text(chunk),
                    "stock_codes": stock_codes,
                    "industry_tags": [],
                }
            )
        self.web.rag_doc_chunk_replace(
            doc_id=doc_id,
            source=source,
            source_url=source_url,
            effective_status=effective_status,
            quality_score=quality_score,
            chunks=payload_chunks,
        )

    @staticmethod
    def _extract_stock_codes_from_text(text: str) -> list[str]:
        """Extract SH/SZ stock codes from free text for retrieval filtering."""
        items = re.findall(r"\b(?:SH|SZ)\d{6}\b", str(text or "").upper())
        return list(dict.fromkeys(items))

    @staticmethod
    def _redact_text(text: str) -> str:
        """Lightweight redaction to remove email/phone/ID patterns from shared text."""
        value = str(text or "")
        value = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]", value)
        value = re.sub(r"\b1\d{10}\b", "[REDACTED_PHONE]", value)
        value = re.sub(r"\b\d{15,18}[0-9Xx]?\b", "[REDACTED_ID]", value)
        return value

    def evals_run(self, samples: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Run evals and return gate results."""
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
        """Get eval run status."""
        return {"eval_run_id": eval_run_id, "status": "not_persisted_in_mvp"}

    def _sql_agent_poc_allowed_schema(self) -> tuple[set[str], set[str]]:
        """Build SQL PoC allowlist from curated tables that exist in local web DB."""
        curated_tables = {
            "query_history",
            "watchlist_items",
            "reports",
            "rag_eval_cases",
            "rag_retrieval_trace",
        }
        existing_rows = self.web.store.query_all(  # type: ignore[attr-defined]
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        existing = {str(x.get("name", "")).strip().lower() for x in existing_rows}
        allowed_tables = {t for t in curated_tables if t in existing}
        allowed_columns: set[str] = set()
        for table in sorted(allowed_tables):
            for row in self.web.store.query_all(f"PRAGMA table_info({table})"):  # type: ignore[attr-defined]
                col = str(row.get("name", "")).strip().lower()
                if col:
                    allowed_columns.add(col)
        return allowed_tables, allowed_columns

    def sql_agent_poc_query(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute SQL Agent PoC query with strict read-only validator and audit trace."""
        sql = str(payload.get("sql", "")).strip()
        requested_max_rows = max(1, min(1000, int(payload.get("max_rows", 200) or 200)))
        trace_id = self.traces.new_trace()
        allowed_tables, allowed_columns = self._sql_agent_poc_allowed_schema()
        if not allowed_tables:
            result = {"ok": False, "error": "sql_agent_unavailable", "reason": "no_allowed_table_available", "rows": []}
            self.traces.emit(trace_id, "sql_agent_poc", {"ok": False, "reason": result["reason"]})
            result["trace_id"] = trace_id
            return result
        validation = SQLSafetyValidator.validate_select_sql(
            sql,
            allowed_tables=allowed_tables,
            allowed_columns=allowed_columns,
            max_limit=requested_max_rows,
        )
        if not bool(validation.get("ok", False)):
            result = {
                "ok": False,
                "error": "sql_not_allowed",
                "reason": str(validation.get("reason", "validation_failed")),
                "validation": validation,
                "rows": [],
                "trace_id": trace_id,
            }
            self.traces.emit(trace_id, "sql_agent_poc", {"ok": False, "reason": result["reason"], "sql": sql[:200]})
            return result
        try:
            rows = self.web.store.query_all(sql)  # type: ignore[attr-defined]
        except Exception as ex:  # noqa: BLE001
            result = {
                "ok": False,
                "error": "sql_execution_error",
                "reason": str(ex)[:220],
                "validation": validation,
                "rows": [],
                "trace_id": trace_id,
            }
            self.traces.emit(trace_id, "sql_agent_poc", {"ok": False, "reason": "execution_error", "sql": sql[:200]})
            return result
        safe_rows = rows[:requested_max_rows]
        result = {
            "ok": True,
            "error": "",
            "reason": "ok",
            "validation": validation,
            "rows": safe_rows,
            "row_count": len(safe_rows),
            "trace_id": trace_id,
        }
        self.traces.emit(
            trace_id,
            "sql_agent_poc",
            {"ok": True, "row_count": len(safe_rows), "tables": list(validation.get("tables", []))[:5]},
        )
        return result

    def _parse_deep_archive_timestamp(self, value: str | None, field: str) -> str | None:
        clean = str(value or "").strip() or None
        if not clean:
            return None
        try:
            parsed = datetime.strptime(clean, self._deep_archive_ts_format)
        except ValueError as ex:
            raise ValueError(f"{field} must use format YYYY-MM-DD HH:MM:SS") from ex
        return parsed.strftime(self._deep_archive_ts_format)

    def _build_deep_archive_query_options(
        self,
        *,
        round_id: str | None = None,
        limit: int = 200,
        event_name: str | None = None,
        cursor: int | None = None,
        created_from: str | None = None,
        created_to: str | None = None,
    ) -> dict[str, Any]:
        try:
            safe_limit = max(1, min(2000, int(limit)))
        except Exception:  # noqa: BLE001
            safe_limit = 200
        safe_cursor = None
        if cursor is not None:
            try:
                safe_cursor = max(0, int(cursor))
            except Exception:  # noqa: BLE001
                safe_cursor = None
        round_id_clean = str(round_id or "").strip() or None
        event_name_clean = str(event_name or "").strip() or None
        created_from_clean = self._parse_deep_archive_timestamp(created_from, "created_from")
        created_to_clean = self._parse_deep_archive_timestamp(created_to, "created_to")
        if created_from_clean and created_to_clean:
            if created_from_clean > created_to_clean:
                raise ValueError("created_from must be <= created_to")
        return {
            "round_id": round_id_clean,
            "event_name": event_name_clean,
            "limit": safe_limit,
            "cursor": safe_cursor,
            "created_from": created_from_clean,
            "created_to": created_to_clean,
        }

    def _resolve_deep_archive_retention_policy(
        self,
        *,
        session: dict[str, Any],
        requested_max_events: Any | None = None,
    ) -> dict[str, Any]:
        env = str(self.settings.env or "dev").strip().lower() or "dev"
        env_defaults = {
            "dev": int(self.settings.deep_archive_max_events_dev),
            "staging": int(self.settings.deep_archive_max_events_staging),
            "prod": int(self.settings.deep_archive_max_events_prod),
        }
        hard_cap = max(100, int(self.settings.deep_archive_max_events_hard_cap))
        base = int(env_defaults.get(env, int(self.settings.deep_archive_max_events_default)))
        policy_source = f"env:{env}"
        user_id = str(session.get("user_id", "")).strip()
        tenant_policy = self._deep_archive_tenant_policy()
        if user_id and user_id in tenant_policy:
            base = int(tenant_policy[user_id])
            policy_source = f"user:{user_id}"
        base = max(1, min(hard_cap, base))

        requested = None
        if requested_max_events is not None:
            try:
                requested = int(requested_max_events)
            except Exception:  # noqa: BLE001
                requested = None
        resolved = base if requested is None else max(1, min(hard_cap, requested))
        return {
            "max_events": resolved,
            "base_max_events": base,
            "policy_source": policy_source,
            "hard_cap": hard_cap,
            "requested_max_events": requested,
        }

    def _emit_deep_archive_audit(
        self,
        *,
        session_id: str,
        action: str,
        status: str,
        started_at: float,
        result_count: int = 0,
        export_bytes: int = 0,
        detail: dict[str, Any] | None = None,
    ) -> None:
        duration_ms = int(round((time.perf_counter() - started_at) * 1000))
        try:
            self.web.deep_think_archive_audit_log(
                session_id=session_id,
                action=action,
                status=status,
                duration_ms=duration_ms,
                result_count=result_count,
                export_bytes=export_bytes,
                detail=detail or {},
            )
        except Exception:  # noqa: BLE001
            return

    def _build_deep_think_round_events(self, session_id: str, latest: dict[str, Any]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = [
            {
                "event": "round_started",
                "data": {
                    "session_id": session_id,
                    "round_id": latest.get("round_id"),
                    "round_no": latest.get("round_no"),
                    "task_graph": latest.get("task_graph", []),
                },
            }
        ]
        budget_usage = latest.get("budget_usage", {})
        if bool(budget_usage.get("warn")):
            events.append({"event": "budget_warning", "data": budget_usage})

        for opinion in latest.get("opinions", []):
            reason = str(opinion.get("reason", ""))
            pivot = max(1, min(len(reason), len(reason) // 2))
            events.append(
                {
                    "event": "agent_opinion_delta",
                    "data": {"agent": opinion.get("agent_id"), "delta": reason[:pivot]},
                }
            )
            events.append({"event": "agent_opinion_final", "data": opinion})
            if str(opinion.get("agent_id")) == "critic_agent":
                events.append(
                    {
                        "event": "critic_feedback",
                        "data": {"reason": reason, "round_id": latest.get("round_id")},
                    }
                )

        events.append(
            {
                "event": "arbitration_final",
                "data": {
                    "consensus_signal": latest.get("consensus_signal"),
                    "disagreement_score": latest.get("disagreement_score"),
                    "conflict_sources": latest.get("conflict_sources", []),
                    "counter_view": latest.get("counter_view", ""),
                },
            }
        )
        if bool(latest.get("replan_triggered", False)):
            events.append(
                {
                    "event": "replan_triggered",
                    "data": {
                        "round_id": latest.get("round_id"),
                        "task_graph": latest.get("task_graph", []),
                        "reason": "disagreement_above_threshold",
                    },
                }
            )
        round_id = str(latest.get("round_id", "")).strip()
        if round_id:
            try:
                related_research_id = self._deep_journal_related_research_id(session_id, round_id)
                linked = self.web.journal_find_by_related_research("", related_research_id=related_research_id)
                linked_id = int(linked.get("journal_id", 0) or 0)
                if linked_id > 0:
                    events.append(
                        {
                            "event": "journal_linked",
                            "data": {
                                "ok": True,
                                "enabled": True,
                                "action": "reused",
                                "journal_id": linked_id,
                                "related_research_id": related_research_id,
                                "round_id": round_id,
                                "round_no": latest.get("round_no", 0),
                            },
                        }
                    )
            except Exception:
                pass
        events.append({"event": "done", "data": {"ok": True, "session_id": session_id}})
        return events

    def _run_deep_archive_export_task(
        self,
        task_id: str,
        session_id: str,
        export_format: str,
        filters: dict[str, Any],
    ) -> None:
        base_backoff_seconds = max(0.0, float(self.settings.deep_archive_export_retry_backoff_seconds))
        while True:
            task_snapshot = self.web.deep_think_export_task_try_claim(task_id, session_id=session_id)
            if not task_snapshot:
                return
            attempt_count = max(1, int(task_snapshot.get("attempt_count", 1) or 1))
            max_attempts = max(1, int(task_snapshot.get("max_attempts", 1) or 1))
            started_at = time.perf_counter()
            try:
                exported = self.deep_think_export_events(
                    session_id,
                    round_id=filters.get("round_id"),
                    limit=int(filters.get("limit", 200)),
                    event_name=filters.get("event_name"),
                    cursor=filters.get("cursor"),
                    created_from=filters.get("created_from"),
                    created_to=filters.get("created_to"),
                    format=export_format,
                    audit_action="archive_export_task_run",
                    emit_audit=False,
                )
                if "error" in exported:
                    raise RuntimeError(str(exported.get("error", "failed")))
                content = str(exported.get("content", ""))
                self.web.deep_think_export_task_update(
                    task_id=task_id,
                    status="completed",
                    filename=str(exported.get("filename", "")),
                    media_type=str(exported.get("media_type", "text/plain; charset=utf-8")),
                    content_text=content,
                    row_count=int(exported.get("count", 0) or 0),
                    error="",
                )
                self._emit_deep_archive_audit(
                    session_id=session_id,
                    action="archive_export_task_complete",
                    status="ok",
                    started_at=started_at,
                    result_count=int(exported.get("count", 0) or 0),
                    export_bytes=len(content.encode("utf-8")),
                    detail={
                        "task_id": task_id,
                        "format": export_format,
                        "attempt_count": attempt_count,
                        "max_attempts": max_attempts,
                    },
                )
                return
            except Exception as ex:  # noqa: BLE001
                message = str(ex)
                if attempt_count < max_attempts:
                    self.web.deep_think_export_task_requeue(task_id=task_id, error=message)
                    self._emit_deep_archive_audit(
                        session_id=session_id,
                        action="archive_export_task_complete",
                        status="retrying",
                        started_at=started_at,
                        detail={
                            "task_id": task_id,
                            "attempt_count": attempt_count,
                            "max_attempts": max_attempts,
                            "error": message,
                        },
                    )
                    if base_backoff_seconds > 0:
                        backoff = min(3.0, base_backoff_seconds * (2 ** max(0, attempt_count - 1)))
                        if backoff > 0:
                            time.sleep(backoff)
                    continue
                self.web.deep_think_export_task_update(task_id=task_id, status="failed", error=message)
                self._emit_deep_archive_audit(
                    session_id=session_id,
                    action="archive_export_task_complete",
                    status="error",
                    started_at=started_at,
                    detail={
                        "task_id": task_id,
                        "attempt_count": attempt_count,
                        "max_attempts": max_attempts,
                        "error": message,
                    },
                )
                return

    def a2a_agent_cards(self) -> list[dict[str, Any]]:
        return self.web.list_agent_cards()

    def a2a_create_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        agent_id = str(payload.get("agent_id", "")).strip()
        if not agent_id:
            raise ValueError("agent_id is required")
        cards = self.web.list_agent_cards()
        if not any(str(card.get("agent_id")) == agent_id for card in cards):
            raise ValueError("agent_id not found in card registry")

        task_id = str(payload.get("task_id", "")).strip() or f"a2a-{uuid.uuid4().hex[:12]}"
        session_id = str(payload.get("session_id", "")).strip() or None
        trace_ref = self.traces.new_trace()
        self.web.a2a_create_task(
            task_id=task_id,
            session_id=session_id,
            agent_id=agent_id,
            status="created",
            payload=payload,
            result={"history": ["created"]},
            trace_ref=trace_ref,
        )

        self.web.a2a_update_task(task_id=task_id, status="accepted", result={"history": ["created", "accepted"]})
        self.web.a2a_update_task(task_id=task_id, status="in_progress", result={"history": ["created", "accepted", "in_progress"]})

        result: dict[str, Any] = {"task_id": task_id, "agent_id": agent_id, "status": "completed"}
        if session_id and str(payload.get("task_type", "")) == "deep_round":
            snapshot = self.deep_think_run_round(session_id, {"question": payload.get("question", "")})
            result["deep_think_snapshot"] = snapshot
        self.web.a2a_update_task(
            task_id=task_id,
            status="completed",
            result={"history": ["created", "accepted", "in_progress", "completed"], "payload_result": result},
        )
        self.traces.emit(trace_ref, "a2a_task_completed", {"task_id": task_id, "agent_id": agent_id})
        return self.web.a2a_get_task(task_id)

    def a2a_get_task(self, task_id: str) -> dict[str, Any]:
        task = self.web.a2a_get_task(task_id)
        if not task:
            return {"error": "not_found", "task_id": task_id}
        return task

    @staticmethod
    def _percentile_from_sorted(values: list[int], ratio: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return float(values[0])
        idx = int(round((len(values) - 1) * max(0.0, min(1.0, float(ratio)))))
        idx = max(0, min(len(values) - 1, idx))
        return float(values[idx])

    def _build_datasource_catalog(self) -> list[dict[str, Any]]:
        """Build datasource metadata snapshot from currently wired adapters."""

        rows: list[dict[str, Any]] = []
        adapter_groups = [
            ("quote", getattr(self.ingestion.quote_service, "adapters", [])),
            ("announcement", getattr(self.ingestion.announcement_service, "adapters", [])),
            ("financial", getattr(self.ingestion.financial_service, "adapters", []) if self.ingestion.financial_service else []),
            ("news", getattr(self.ingestion.news_service, "adapters", []) if self.ingestion.news_service else []),
            ("research", getattr(self.ingestion.research_service, "adapters", []) if self.ingestion.research_service else []),
            ("macro", getattr(self.ingestion.macro_service, "adapters", []) if self.ingestion.macro_service else []),
            ("fund", getattr(self.ingestion.fund_service, "adapters", []) if self.ingestion.fund_service else []),
        ]
        for category, adapters in adapter_groups:
            for adapter in adapters:
                source_id = str(getattr(adapter, "source_id", f"{category}_unknown"))
                cfg = getattr(adapter, "config", None)
                reliability = float(
                    getattr(cfg, "reliability_score", getattr(adapter, "reliability_score", 0.0)) or 0.0
                )
                proxy_url = str(getattr(cfg, "proxy_url", "") or "")
                cookie = str(getattr(adapter, "cookie", "") or "")
                # Cookie-backed sources are marked disabled when credential is missing.
                enabled = bool(cookie.strip()) if hasattr(adapter, "cookie") else True
                rows.append(
                    {
                        "source_id": source_id,
                        "category": category,
                        "enabled": enabled,
                        "reliability_score": round(reliability, 4),
                        "source_url": str(getattr(adapter, "source_url", "") or ""),
                        "proxy_enabled": bool(proxy_url.strip()),
                        # Frontend uses this list to explain datasource business impact.
                        "used_in_ui_modules": self._datasource_ui_modules(category),
                    }
                )
        # History ingestion currently relies on HistoryService (non-adapter style), expose a synthetic row.
        rows.append(
            {
                "source_id": "eastmoney_history",
                "category": "history",
                "enabled": True,
                "reliability_score": 0.9,
                "source_url": "https://push2his.eastmoney.com/",
                "proxy_enabled": bool(str(self.settings.datasource_proxy_url or "").strip()),
                "used_in_ui_modules": self._datasource_ui_modules("history"),
            }
        )

        dedup: dict[str, dict[str, Any]] = {}
        for row in rows:
            dedup.setdefault(str(row["source_id"]), row)
        return sorted(dedup.values(), key=lambda x: (str(x.get("category", "")), str(x.get("source_id", ""))))

    @staticmethod
    def _datasource_ui_modules(category: str) -> list[str]:
        """Map datasource category to user-facing modules for observability explainability."""
        mapping = {
            "quote": ["/deep-think", "/analysis-studio", "/predict", "/watchlist"],
            "announcement": ["/deep-think", "/reports", "/analysis-studio"],
            "financial": ["/deep-think", "/analysis-studio", "/predict"],
            "news": ["/deep-think", "/analysis-studio", "/reports"],
            "research": ["/deep-think", "/analysis-studio", "/reports"],
            "macro": ["/deep-think", "/analysis-studio"],
            "fund": ["/deep-think", "/analysis-studio"],
            "history": ["/deep-think", "/analysis-studio", "/predict"],
            "scheduler": ["/ops/scheduler", "/ops/health"],
        }
        return list(mapping.get(str(category or "").strip().lower(), []))

    def _append_datasource_log(
        self,
        *,
        source_id: str,
        category: str,
        action: str,
        status: str,
        latency_ms: int,
        detail: dict[str, Any] | None = None,
        error: str = "",
    ) -> dict[str, Any]:
        self._datasource_log_seq += 1
        row = {
            "log_id": self._datasource_log_seq,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_id": source_id,
            "category": category,
            "action": action,
            "status": status,
            "latency_ms": max(0, int(latency_ms)),
            "error": error,
            "detail": detail or {},
        }
        self._datasource_logs.append(row)
        # Keep bounded memory footprint for long-running dev processes.
        if len(self._datasource_logs) > 5000:
            self._datasource_logs = self._datasource_logs[-5000:]
        return row

    def datasource_sources(self, token: str) -> dict[str, Any]:
        self.web.require_role(token, {"admin", "ops"})
        items = self._build_datasource_catalog()
        return {"count": len(items), "items": items}

    def datasource_logs(self, token: str, *, source_id: str = "", status: str = "", limit: int = 100) -> dict[str, Any]:
        self.web.require_role(token, {"admin", "ops"})
        safe_limit = max(1, min(1000, int(limit)))
        source_filter = str(source_id or "").strip()
        status_filter = str(status or "").strip().lower()
        rows = list(reversed(self._datasource_logs))
        if source_filter:
            rows = [x for x in rows if str(x.get("source_id", "")) == source_filter]
        if status_filter:
            rows = [x for x in rows if str(x.get("status", "")).lower() == status_filter]
        rows = rows[:safe_limit]
        return {"count": len(rows), "items": rows}

    def datasource_health(self, token: str, *, limit: int = 200) -> dict[str, Any]:
        self.web.require_role(token, {"admin", "ops"})
        safe_limit = max(1, min(1000, int(limit)))
        catalog = self._build_datasource_catalog()
        metrics_by_source: dict[str, dict[str, Any]] = {}
        for row in reversed(self._datasource_logs):
            source_id = str(row.get("source_id", ""))
            if not source_id:
                continue
            metrics = metrics_by_source.setdefault(
                source_id,
                {"attempts": 0, "success": 0, "failed": 0, "last_error": "", "last_latency_ms": 0, "last_seen_at": ""},
            )
            metrics["attempts"] += 1
            if str(row.get("status", "")) == "ok":
                metrics["success"] += 1
            elif str(row.get("status", "")) == "failed":
                metrics["failed"] += 1
                if not metrics["last_error"]:
                    metrics["last_error"] = str(row.get("error", ""))
            metrics["last_latency_ms"] = int(row.get("latency_ms", 0) or 0)
            metrics["last_seen_at"] = str(row.get("created_at", ""))

        items: list[dict[str, Any]] = []
        for source in catalog:
            source_id = str(source.get("source_id", ""))
            metric = metrics_by_source.get(source_id, {})
            attempts = int(metric.get("attempts", 0) or 0)
            failed = int(metric.get("failed", 0) or 0)
            success = int(metric.get("success", 0) or 0)
            success_rate = (float(success) / float(attempts)) if attempts > 0 else 1.0
            failure_rate = (float(failed) / float(attempts)) if attempts > 0 else 0.0
            # When a source has never been fetched in this process lifetime, freshness is unknown.
            last_seen_at = str(metric.get("last_seen_at", ""))
            staleness_minutes: int | None = None
            if last_seen_at:
                delta = datetime.now(timezone.utc) - self._parse_time(last_seen_at)
                staleness_minutes = max(0, int(delta.total_seconds() // 60))
            item = {
                **source,
                "attempts": attempts,
                "success_rate": round(success_rate, 4),
                "failure_rate": round(failure_rate, 4),
                "last_error": str(metric.get("last_error", "")),
                "last_latency_ms": int(metric.get("last_latency_ms", 0) or 0),
                "updated_at": last_seen_at,
                "last_used_at": last_seen_at,
                "staleness_minutes": staleness_minutes,
                "circuit_open": False,
                "used_in_ui_modules": source.get(
                    "used_in_ui_modules",
                    self._datasource_ui_modules(str(source.get("category", ""))),
                ),
            }
            self.web.source_health_upsert(
                source_id=source_id,
                success_rate=float(item["success_rate"]),
                circuit_open=False,
                last_error=item["last_error"],
            )
            # Alert when a source continuously fails within the local observation window.
            if attempts >= 3 and failure_rate >= 0.6:
                self.web.create_alert(
                    "datasource_failure_rate_high",
                    "high",
                    f"{source_id} failure_rate={failure_rate:.2f} attempts={attempts}",
                )
            items.append(item)

        scheduler_state = self.scheduler_status()
        for job_name, snapshot in scheduler_state.items():
            circuit_open = bool(snapshot.get("circuit_open_until"))
            if circuit_open:
                self.web.create_alert("scheduler_circuit_open", "high", f"{job_name} circuit open")
            scheduler_updated_at = str(snapshot.get("last_run_at", ""))
            scheduler_staleness: int | None = None
            if scheduler_updated_at:
                delta = datetime.now(timezone.utc) - self._parse_time(scheduler_updated_at)
                scheduler_staleness = max(0, int(delta.total_seconds() // 60))
            items.append(
                {
                    "source_id": str(job_name),
                    "category": "scheduler",
                    "enabled": True,
                    "reliability_score": 0.0,
                    "source_url": "",
                    "proxy_enabled": False,
                    "attempts": 0,
                    "success_rate": 1.0 if snapshot.get("last_status") in ("ok", "never") else 0.0,
                    "failure_rate": 0.0 if snapshot.get("last_status") in ("ok", "never") else 1.0,
                    "last_error": str(snapshot.get("last_error", "")),
                    "last_latency_ms": 0,
                    "updated_at": scheduler_updated_at,
                    "last_used_at": scheduler_updated_at,
                    "staleness_minutes": scheduler_staleness,
                    "circuit_open": circuit_open,
                    "used_in_ui_modules": self._datasource_ui_modules("scheduler"),
                }
            )

        items = sorted(items, key=lambda x: (str(x.get("category", "")), str(x.get("source_id", ""))))[:safe_limit]
        return {"count": len(items), "items": items}

    def business_data_health(self, *, stock_code: str = "", limit: int = 200) -> dict[str, Any]:
        """Return business-readable datasource health summary for frontend modules."""
        try:
            health = self.datasource_health("", limit=limit)
        except Exception:
            # Fallback to catalog-only snapshot when role checks block observability APIs.
            health = {"count": 0, "items": self._build_datasource_catalog()}
        rows = list(health.get("items", []))
        category_map: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            category = str(row.get("category", "")).strip() or "unknown"
            category_map.setdefault(category, []).append(row)

        # Keep the mapping explicit so the frontend can explain module impact in plain language.
        module_categories: dict[str, list[str]] = {
            "predict": ["quote", "history", "financial"],
            "reports": ["quote", "history", "financial", "news", "research", "macro", "fund", "announcement"],
            "deep-think": ["quote", "history", "financial", "news", "research", "macro", "fund", "announcement"],
            "analysis-studio": ["quote", "history", "financial", "news", "research", "macro", "fund", "announcement"],
        }
        module_health: list[dict[str, Any]] = []
        global_reasons: list[str] = []
        for module, categories in module_categories.items():
            expected = max(1, len(categories))
            healthy_count = 0
            missing_categories: list[str] = []
            stale_categories: list[str] = []
            failed_categories: list[str] = []
            for category in categories:
                candidates = category_map.get(category, [])
                if not candidates:
                    missing_categories.append(category)
                    continue
                top = sorted(
                    candidates,
                    key=lambda x: (
                        float(x.get("success_rate", 0.0) or 0.0),
                        -float(x.get("failure_rate", 1.0) or 1.0),
                    ),
                    reverse=True,
                )[0]
                failure_rate = float(top.get("failure_rate", 0.0) or 0.0)
                staleness = top.get("staleness_minutes")
                staleness_minutes = int(staleness) if isinstance(staleness, (int, float)) else None
                if failure_rate >= 0.6:
                    failed_categories.append(category)
                elif staleness_minutes is not None and staleness_minutes > 240:
                    stale_categories.append(category)
                else:
                    healthy_count += 1

            coverage = round(float(healthy_count) / float(expected), 4)
            status = "ok" if coverage >= 0.85 else "degraded" if coverage >= 0.5 else "critical"
            reasons: list[str] = []
            if missing_categories:
                reasons.append(f"missing:{','.join(missing_categories)}")
            if stale_categories:
                reasons.append(f"stale:{','.join(stale_categories)}")
            if failed_categories:
                reasons.append(f"failed:{','.join(failed_categories)}")
            module_health.append(
                {
                    "module": module,
                    "status": status,
                    "coverage": coverage,
                    "healthy_categories": healthy_count,
                    "expected_categories": expected,
                    "degrade_reasons": reasons,
                }
            )
            global_reasons.extend(reasons)

        status_rank = {"ok": 0, "degraded": 1, "critical": 2}
        overall_status = "ok"
        for row in module_health:
            if status_rank.get(str(row.get("status", "")), 0) > status_rank.get(overall_status, 0):
                overall_status = str(row.get("status", "ok"))

        stock = str(stock_code or "").strip().upper()
        stock_snapshot = {}
        if stock:
            stock_snapshot = {
                "stock_code": stock,
                "has_quote": bool(self._latest_quote(stock)),
                "history_sample_size": len(self._history_bars(stock, limit=260)),
                "has_financial": bool(self._latest_financial_snapshot(stock)),
                "news_count": len([x for x in self.ingestion_store.news_items if str(x.get("stock_code", "")).upper() == stock][-20:]),
                "research_count": len(
                    [x for x in self.ingestion_store.research_reports if str(x.get("stock_code", "")).upper() == stock][-20:]
                ),
            }

        return {
            "status": overall_status,
            "module_health": module_health,
            "degrade_reasons": list(dict.fromkeys(global_reasons)),
            "category_health_count": len(rows),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "stock_snapshot": stock_snapshot,
        }

    @staticmethod
    def _infer_datasource_category(source_id: str) -> str:
        sid = str(source_id or "").strip().lower()
        if sid.startswith(("tencent", "sina", "netease", "xueqiu")) and "news" not in sid:
            return "quote"
        if "announcement" in sid:
            return "announcement"
        if "financial" in sid or "tushare" in sid:
            return "financial"
        if "news" in sid or sid.startswith(("cls", "tradingview")):
            return "news"
        if "research" in sid:
            return "research"
        if "macro" in sid:
            return "macro"
        if "fund" in sid or "ttjj" in sid:
            return "fund"
        if "history" in sid:
            return "history"
        return ""

    def datasource_fetch(self, token: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.web.require_role(token, {"admin", "ops"})
        source_id = str(payload.get("source_id", "")).strip()
        if not source_id:
            raise ValueError("source_id is required")

        stock_codes = [str(x).strip().upper() for x in payload.get("stock_codes", []) if str(x).strip()]
        limit = max(1, min(500, int(payload.get("limit", 20) or 20)))
        catalog = {str(x.get("source_id", "")): x for x in self._build_datasource_catalog()}
        category = str(payload.get("category", "")).strip().lower() or str(catalog.get(source_id, {}).get("category", ""))
        if not category:
            category = self._infer_datasource_category(source_id)
        if not category:
            raise ValueError("unable to infer datasource category from source_id")

        started_at = time.perf_counter()
        error = ""
        result: dict[str, Any]
        try:
            if category == "quote":
                result = self.ingest_market_daily(stock_codes or ["SH600000"])
            elif category == "announcement":
                result = self.ingest_announcements(stock_codes or ["SH600000"])
            elif category == "history":
                result = self.ingestion.ingest_history_daily(stock_codes or ["SH600000"], limit=max(60, limit))
            elif category == "financial":
                result = self.ingest_financials(stock_codes or ["SH600000"])
            elif category == "news":
                result = self.ingest_news(stock_codes or ["SH600000"], limit=limit)
            elif category == "research":
                result = self.ingest_research_reports(stock_codes or ["SH600000"], limit=limit)
            elif category == "macro":
                result = self.ingest_macro_indicators(limit=limit)
            elif category == "fund":
                result = self.ingest_fund_snapshots(stock_codes or ["SH600000"])
            else:
                raise ValueError(f"unsupported datasource category: {category}")
        except Exception as ex:  # noqa: BLE001
            error = str(ex)
            result = {
                "task_name": f"{category}-fetch",
                "success_count": 0,
                "failed_count": max(1, len(stock_codes) if stock_codes else 1),
                "details": [{"status": "failed", "error": error}],
            }

        success_count = int(result.get("success_count", 0) or 0)
        failed_count = int(result.get("failed_count", 0) or 0)
        status = "ok"
        if failed_count > 0 and success_count > 0:
            status = "partial"
        elif failed_count > 0 and success_count == 0:
            status = "failed"
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        log = self._append_datasource_log(
            source_id=source_id,
            category=category,
            action="fetch",
            status=status,
            latency_ms=latency_ms,
            detail={
                "stock_codes": stock_codes,
                "limit": limit,
                "task_name": str(result.get("task_name", "")),
                "success_count": success_count,
                "failed_count": failed_count,
            },
            error=error or (str(result.get("details", [{}])[0].get("error", "")) if failed_count > 0 else ""),
        )
        if status == "failed":
            self.web.create_alert("datasource_fetch_failed", "medium", f"{source_id} fetch failed")

        return {
            "source_id": source_id,
            "category": category,
            "status": status,
            "latency_ms": latency_ms,
            "result": result,
            "log_id": int(log["log_id"]),
        }

    def _build_rule_based_debate_opinions(
        self,
        question: str,
        trend: dict[str, Any],
        quote: dict[str, Any],
        quant_20: dict[str, Any],
    ) -> list[dict[str, Any]]:
        pm_signal = "hold"
        if trend.get("ma20_slope", 0.0) > 0 and float(quote.get("pct_change", 0.0)) > 0:
            pm_signal = "buy"
        elif trend.get("ma20_slope", 0.0) < 0:
            pm_signal = "reduce"

        quant_signal = str(quant_20.get("signal", "hold")).replace("strong_", "")
        risk_signal = "hold"
        if trend.get("max_drawdown_60", 0.0) > 0.2 or trend.get("volatility_20", 0.0) > 0.025:
            risk_signal = "reduce"
        elif trend.get("volatility_20", 0.0) < 0.012 and trend.get("momentum_20", 0.0) > 0:
            risk_signal = "buy"

        return [
            {
                "agent": "pm_agent",
                "signal": pm_signal,
                "confidence": 0.66,
                "reason": f"基于趋势斜率与当日涨跌构建产品侧可解释观点，question={question or 'default'}",
            },
            {
                "agent": "quant_agent",
                "signal": quant_signal,
                "confidence": float(quant_20.get("up_probability", 0.5)),
                "reason": f"来自20日预测：{quant_20.get('rationale', '')}",
            },
            {
                "agent": "risk_agent",
                "signal": risk_signal,
                "confidence": 0.72,
                "reason": (
                    f"回撤={trend.get('max_drawdown_60', 0.0):.4f}, 波动={trend.get('volatility_20', 0.0):.4f}, "
                    f"动量={trend.get('momentum_20', 0.0):.4f}"
                ),
            },
        ]

    def _build_llm_debate_opinions(
        self,
        question: str,
        stock_code: str,
        trend: dict[str, Any],
        quote: dict[str, Any],
        quant_20: dict[str, Any],
        rule_defaults: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Call LLM to generate multi-agent debate opinions."""
        context = {
            "question": question or "请给出短中期观点",
            "stock_code": stock_code,
            "quote": {"price": quote.get("price"), "pct_change": quote.get("pct_change")},
            "trend": trend,
            "quant_20": quant_20,
        }
        prompts = [
            (
                "pm_agent",
                (
                    "You are a portfolio manager. Return JSON only with keys: signal, confidence, reason. "
                    f"context={json.dumps(context, ensure_ascii=False)}"
                ),
            ),
            (
                "quant_agent",
                (
                    "You are a quant analyst. Return JSON only with keys: signal, confidence, reason. "
                    f"context={json.dumps(context, ensure_ascii=False)}"
                ),
            ),
            (
                "risk_agent",
                (
                    "You are a risk controller. Return JSON only with keys: signal, confidence, reason. "
                    f"context={json.dumps(context, ensure_ascii=False)}"
                ),
            ),
        ]

        rule_map = {x["agent"]: x for x in rule_defaults}
        outputs: list[dict[str, Any]] = []

        def run_one(agent_name: str, prompt_text: str) -> dict[str, Any]:
            fallback = rule_map.get(agent_name, {"signal": "hold", "confidence": 0.5, "reason": "rule default"})
            state = AgentState(user_id="ops", question=question or "ops debate", stock_codes=[stock_code], trace_id=self.traces.new_trace())
            try:
                raw = self.llm_gateway.generate(state, prompt_text)
                cleaned = raw.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.strip("`")
                    if cleaned.lower().startswith("json"):
                        cleaned = cleaned[4:].strip()
                obj = json.loads(cleaned)
                signal = str(obj.get("signal", fallback["signal"])).lower()
                if signal not in {"buy", "hold", "reduce"}:
                    signal = str(fallback["signal"])
                conf = float(obj.get("confidence", fallback["confidence"]))
                conf = max(0.0, min(1.0, conf))
                reason = str(obj.get("reason", fallback["reason"]))[:320]
                return {"agent": agent_name, "signal": signal, "confidence": conf, "reason": reason}
            except Exception:
                return {
                    "agent": agent_name,
                    "signal": str(fallback["signal"]),
                    "confidence": float(fallback["confidence"]),
                    "reason": str(fallback["reason"]) + " | llm_fallback",
                }

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(run_one, n, p) for n, p in prompts]
            for f in futures:
                outputs.append(f.result())
        return outputs

    def alerts_list(self, token: str) -> list[dict[str, Any]]:
        return self.web.alerts_list(token)

    def alerts_ack(self, token: str, alert_id: int) -> dict[str, Any]:
        return self.web.alerts_ack(token, alert_id)

    def alert_rule_create(self, token: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.web.alert_rule_create(token, payload)

    def alert_rule_list(self, token: str) -> list[dict[str, Any]]:
        return self.web.alert_rule_list(token)

    def alert_rule_delete(self, token: str, rule_id: int) -> dict[str, Any]:
        return self.web.alert_rule_delete(token, rule_id)

    def alert_trigger_logs(self, token: str, *, limit: int = 100) -> list[dict[str, Any]]:
        return self.web.alert_trigger_log_list(token, limit=limit)

    def alert_rule_check(self, token: str) -> dict[str, Any]:
        rules = [r for r in self.web.alert_rule_list(token) if bool(r.get("is_active", False))]
        stock_codes = list({str(r.get("stock_code", "")).upper() for r in rules if str(r.get("stock_code", "")).strip()})
        if stock_codes:
            try:
                self.ingest_market_daily(stock_codes)
            except Exception:
                pass
            try:
                self.ingest_announcements(stock_codes)
            except Exception:
                pass
        quote_map = {str(x.get("stock_code", "")).upper(): x for x in self.ingestion_store.quotes[-500:]}
        ann_by_code: dict[str, list[dict[str, Any]]] = {}
        for event in self.ingestion_store.announcements[-800:]:
            code = str(event.get("stock_code", "")).upper()
            if not code:
                continue
            ann_by_code.setdefault(code, []).append(event)

        triggered: list[dict[str, Any]] = []
        for rule in rules:
            rule_id = int(rule.get("rule_id", 0) or 0)
            code = str(rule.get("stock_code", "")).upper()
            rule_type = str(rule.get("rule_type", "")).lower()
            hit = False
            trigger_data: dict[str, Any] = {}
            if rule_type == "price":
                quote = quote_map.get(code, {})
                current = float(quote.get("price", 0.0) or 0.0)
                op = str(rule.get("operator", ""))
                target = float(rule.get("target_value", 0.0) or 0.0)
                if op == ">" and current > target:
                    hit = True
                elif op == "<" and current < target:
                    hit = True
                elif op == ">=" and current >= target:
                    hit = True
                elif op == "<=" and current <= target:
                    hit = True
                trigger_data = {"current_price": current, "target_value": target, "operator": op}
            elif rule_type == "event":
                event_type = str(rule.get("event_type", "")).strip().lower()
                events = ann_by_code.get(code, [])
                for event in events[-20:]:
                    title = str(event.get("title", "")).lower()
                    etype = str(event.get("event_type", "")).lower()
                    if not event_type or event_type in etype or event_type in title:
                        hit = True
                        trigger_data = {
                            "event_type": etype,
                            "title": str(event.get("title", "")),
                            "event_time": str(event.get("event_time", "")),
                        }
                        break

            if not hit:
                continue
            message = f"rule[{rule_id}] triggered for {code}"
            alert_id = self.web.create_alert("rule_trigger", "medium", message)
            self.web.alert_trigger_log_add(
                rule_id=rule_id,
                stock_code=code,
                trigger_message=message,
                trigger_data=trigger_data,
            )
            triggered.append({"alert_id": alert_id, "rule_id": rule_id, "stock_code": code, "message": message, "data": trigger_data})
        return {"checked_rules": len(rules), "triggered_count": len(triggered), "items": triggered}

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
        industry_l2: str = "",
        industry_l3: str = "",
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        return self.web.stock_universe_search(
            token,
            keyword=keyword,
            exchange=exchange,
            market_tier=market_tier,
            listing_board=listing_board,
            industry_l1=industry_l1,
            industry_l2=industry_l2,
            industry_l3=industry_l3,
            limit=limit,
        )

    def stock_universe_filters(self, token: str) -> dict[str, list[str]]:
        return self.web.stock_universe_filters(token)

    def close(self) -> None:
        """Release database connections and other service resources."""
        self.memory.close()
        self.prompts.close()
        self.web.close()






