from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass, field
from typing import Any

from backend.app.docs.pipeline import DocumentPipeline
from backend.app.data.sources import AnnouncementService, HistoryService, QuoteService


@dataclass(slots=True)
class IngestionStore:
    """摄取数据的本地存储容器（MVP 内存版）。"""

    quotes: list[dict[str, Any]] = field(default_factory=list)
    history_bars: list[dict[str, Any]] = field(default_factory=list)
    announcements: list[dict[str, Any]] = field(default_factory=list)
    docs: dict[str, dict[str, Any]] = field(default_factory=dict)
    review_queue: list[dict[str, Any]] = field(default_factory=list)


class IngestionService:
    """数据摄取服务。

    可类比 Java 中的数据同步 Job Service。
    """

    def __init__(
        self,
        quote_service: QuoteService,
        announcement_service: AnnouncementService,
        store: IngestionStore,
        history_service: HistoryService | None = None,
    ):
        """注入行情服务、公告服务和存储对象。"""
        self.quote_service = quote_service
        self.announcement_service = announcement_service
        self.history_service = history_service or HistoryService()
        self.store = store
        self.doc_pipeline = DocumentPipeline()

    def ingest_market_daily(self, stock_codes: list[str]) -> dict[str, Any]:
        """抓取日级行情并写入本地存储。"""
        success, failed, details = 0, 0, []
        for code in stock_codes:
            try:
                normalized_code = _normalize_stock_code(code)
                quote = self.quote_service.get_quote(normalized_code)
                payload = _normalize_quote_payload(asdict(quote))
                payload["conflict_flag"] = _detect_quote_conflict(payload, self.store.quotes)
                payload["quality_flags"] = _validate_quote_payload(payload)
                self.store.quotes.append(payload)
                success += 1
                details.append(
                    {
                        "stock_code": normalized_code,
                        "status": "ok",
                        "source": quote.source_id,
                        "conflict_flag": payload["conflict_flag"],
                        "quality_flags": payload["quality_flags"],
                    }
                )
            except RuntimeError as ex:
                failed += 1
                details.append({"stock_code": code, "status": "failed", "error": str(ex)})
        return {"task_name": "market-daily", "success_count": success, "failed_count": failed, "details": details}

    def ingest_announcements(self, stock_codes: list[str]) -> dict[str, Any]:
        """抓取公告并写入本地存储。"""
        details = []
        for code in stock_codes:
            normalized_code = _normalize_stock_code(code)
            events = self.announcement_service.fetch_announcements(normalized_code)
            normalized_events: list[dict[str, Any]] = []
            for event in events:
                item = _normalize_announcement_payload(event, normalized_code)
                item["conflict_flag"] = _detect_announcement_conflict(item, self.store.announcements)
                item["quality_flags"] = _validate_announcement_payload(item)
                normalized_events.append(item)
            self.store.announcements.extend(normalized_events)
            details.append({"stock_code": normalized_code, "count": len(normalized_events), "status": "ok"})
        return {
            "task_name": "announcements",
            "success_count": len(stock_codes),
            "failed_count": 0,
            "details": details,
        }

    def ingest_history_daily(self, stock_codes: list[str], limit: int = 240) -> dict[str, Any]:
        """抓取历史日线并写入本地存储。"""
        success, failed, details = 0, 0, []
        for code in stock_codes:
            try:
                normalized_code = _normalize_stock_code(code)
                bars = self.history_service.fetch_daily_bars(normalized_code, limit=limit)
                # 先删除该标的旧历史，再写入最新窗口。
                self.store.history_bars = [x for x in self.store.history_bars if x.get("stock_code") != normalized_code]
                self.store.history_bars.extend(bars)
                success += 1
                details.append(
                    {
                        "stock_code": normalized_code,
                        "status": "ok",
                        "bar_count": len(bars),
                        "source": bars[-1].get("source_id", "eastmoney_history"),
                    }
                )
            except RuntimeError as ex:
                failed += 1
                details.append({"stock_code": code, "status": "failed", "error": str(ex)})
        return {"task_name": "history-daily", "success_count": success, "failed_count": failed, "details": details}

    def upload_doc(self, doc_id: str, filename: str, content: str, source: str) -> dict[str, Any]:
        """上传文档原文到文档存储。"""
        processed = self.doc_pipeline.process(doc_id=doc_id, filename=filename, content=content, source=source)
        self.store.docs[doc_id] = {
            "doc_id": doc_id,
            "filename": filename,
            "content": content,
            "source": source,
            "indexed": False,
            "doc_hash": processed["doc_hash"],
            "parse_confidence": processed["parse_confidence"],
            "version": processed["version"],
        }
        if processed["needs_review"]:
            self.store.review_queue.append(
                {
                    "doc_id": doc_id,
                    "reason": "low_parse_confidence",
                    "parse_confidence": processed["parse_confidence"],
                    "filename": filename,
                }
            )
        return {"doc_id": doc_id, "status": "uploaded"}

    def index_doc(self, doc_id: str) -> dict[str, Any]:
        """执行文档切块并标记索引状态。"""
        doc = self.store.docs.get(doc_id)
        if not doc:
            return {"doc_id": doc_id, "status": "not_found"}
        processed = self.doc_pipeline.process(
            doc_id=doc["doc_id"],
            filename=doc["filename"],
            content=doc["content"],
            source=doc["source"],
        )
        doc["indexed"] = True
        doc["chunks"] = processed["chunks"]
        doc["tables"] = processed["tables"]
        doc["cleaned_text"] = processed["cleaned_text"]
        doc["parse_confidence"] = processed["parse_confidence"]
        return {
            "doc_id": doc_id,
            "status": "indexed",
            "chunk_count": len(doc["chunks"]),
            "table_count": len(doc["tables"]),
            "parse_confidence": doc["parse_confidence"],
        }


def _split_chunks(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    """按固定窗口+重叠策略切分文本。"""
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def _normalize_stock_code(stock_code: str) -> str:
    code = (stock_code or "").upper().replace(".", "")
    if code.startswith(("SH", "SZ")):
        return code
    if code.startswith("6"):
        return "SH" + code
    return "SZ" + code


def _normalize_quote_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload["stock_code"] = _normalize_stock_code(str(payload.get("stock_code", "")))
    payload["ts"] = str(payload.get("ts", ""))
    payload["price"] = float(payload.get("price", 0.0))
    payload["pct_change"] = float(payload.get("pct_change", 0.0))
    payload["volume"] = float(payload.get("volume", 0.0))
    payload["turnover"] = float(payload.get("turnover", 0.0))
    payload["source_id"] = str(payload.get("source_id", "unknown"))
    payload["source_url"] = str(payload.get("source_url", ""))
    payload["reliability_score"] = float(payload.get("reliability_score", 0.0))
    return payload


def _normalize_announcement_payload(payload: dict[str, Any], stock_code: str) -> dict[str, Any]:
    return {
        "stock_code": stock_code,
        "event_type": str(payload.get("event_type", "announcement")),
        "title": str(payload.get("title", "")),
        "content": str(payload.get("content", "")),
        "event_time": str(payload.get("event_time", "")),
        "source_id": str(payload.get("source_id", "unknown")),
        "source_url": str(payload.get("source_url", "")),
        "reliability_score": float(payload.get("reliability_score", 0.0)),
    }


def _validate_quote_payload(payload: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    required = ("stock_code", "ts", "source_id", "source_url")
    for key in required:
        if not payload.get(key):
            flags.append(f"missing_{key}")
    if payload.get("price", 0.0) <= 0:
        flags.append("invalid_price")
    if payload.get("reliability_score", 0.0) < 0.5:
        flags.append("low_reliability")
    return flags


def _validate_announcement_payload(payload: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    required = ("stock_code", "title", "event_time", "source_id", "source_url")
    for key in required:
        if not payload.get(key):
            flags.append(f"missing_{key}")
    if payload.get("reliability_score", 0.0) < 0.5:
        flags.append("low_reliability")
    return flags


def _detect_quote_conflict(current: dict[str, Any], history: list[dict[str, Any]]) -> bool:
    for item in reversed(history):
        if item.get("stock_code") != current.get("stock_code"):
            continue
        if item.get("source_id") == current.get("source_id"):
            continue
        old_price = float(item.get("price", 0.0))
        new_price = float(current.get("price", 0.0))
        if old_price > 0 and abs(new_price - old_price) / old_price > 0.03:
            return True
        break
    return False


def _detect_announcement_conflict(current: dict[str, Any], history: list[dict[str, Any]]) -> bool:
    for item in reversed(history):
        if item.get("stock_code") != current.get("stock_code"):
            continue
        if item.get("title") != current.get("title"):
            continue
        if item.get("content") != current.get("content"):
            return True
        break
    return False
