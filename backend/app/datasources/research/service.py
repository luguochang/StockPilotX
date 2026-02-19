from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol

from backend.app.datasources.base.adapter import DataSourceConfig
from backend.app.datasources.base.utils import normalize_stock_code
from backend.app.datasources.research.eastmoney_research import EastmoneyResearchAdapter


class ResearchAdapter(Protocol):
    source_id: str

    def fetch_reports(self, stock_code: str, limit: int = 20) -> list[dict]:
        ...


class MockResearchAdapter:
    def __init__(self, source_id: str = "research_mock", reliability_score: float = 0.60) -> None:
        self.source_id = source_id
        self.reliability_score = reliability_score

    def fetch_reports(self, stock_code: str, limit: int = 20) -> list[dict]:
        code = normalize_stock_code(stock_code)
        now_iso = datetime.now(timezone.utc).isoformat()
        return [
            {
                "stock_code": code,
                # Keep schema stable so downstream RAG indexing can still run.
                "title": f"{code} 研究报告摘要（mock）",
                "content": "外部研报源不可用，当前为本地兜底摘要。",
                "published_at": now_iso,
                "author": "system",
                "org_name": "mock",
                "source_id": self.source_id,
                "source_url": f"https://{self.source_id}.example.com/{code}",
                "reliability_score": self.reliability_score,
            }
        ][: max(1, limit)]


class ResearchService:
    def __init__(self, adapters: list[ResearchAdapter]) -> None:
        self.adapters = adapters

    def fetch_reports(self, stock_code: str, limit: int = 20) -> list[dict]:
        errors: list[str] = []
        for adapter in self.adapters:
            try:
                rows = adapter.fetch_reports(stock_code, limit=limit)
                if rows:
                    return rows
            except Exception as ex:  # noqa: BLE001
                errors.append(f"{getattr(adapter, 'source_id', 'unknown')}: {ex}")
        raise RuntimeError("all research sources failed: " + "; ".join(errors))

    @classmethod
    def build_default(
        cls,
        *,
        timeout_seconds: float = 2.0,
        retry_count: int = 2,
        retry_backoff_seconds: float = 0.3,
        proxy_url: str = "",
    ) -> "ResearchService":
        eastmoney_cfg = DataSourceConfig(
            source_id="eastmoney_research",
            reliability_score=0.82,
            timeout_seconds=timeout_seconds,
            retry_count=retry_count,
            retry_backoff_seconds=retry_backoff_seconds,
            proxy_url=proxy_url,
        )
        adapters: list[ResearchAdapter] = [
            EastmoneyResearchAdapter(eastmoney_cfg),
            MockResearchAdapter(),
        ]
        return cls(adapters)
