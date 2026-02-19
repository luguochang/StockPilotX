from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol

from backend.app.datasources.base.adapter import DataSourceConfig
from backend.app.datasources.base.utils import normalize_stock_code
from backend.app.datasources.news.cls import CLSNewsAdapter
from backend.app.datasources.news.tradingview import TradingViewNewsAdapter
from backend.app.datasources.news.xueqiu_news import XueqiuNewsAdapter


class NewsAdapter(Protocol):
    source_id: str

    def fetch_news(self, stock_code: str, limit: int = 20) -> list[dict]:
        ...


class MockNewsAdapter:
    def __init__(self, source_id: str = "news_mock", reliability_score: float = 0.58) -> None:
        self.source_id = source_id
        self.reliability_score = reliability_score

    def fetch_news(self, stock_code: str, limit: int = 20) -> list[dict]:
        code = normalize_stock_code(stock_code)
        now_iso = datetime.now(timezone.utc).isoformat()
        return [
            {
                "stock_code": code,
                # Mock payload keeps retrieval/ranking paths runnable in offline mode.
                "title": f"{code} 行业新闻跟踪（mock）",
                "content": "外部新闻源暂不可用，当前为本地兜底新闻摘要。",
                "event_time": now_iso,
                "source_id": self.source_id,
                "source_url": f"https://{self.source_id}.example.com/{code}",
                "reliability_score": self.reliability_score,
            }
        ][: max(1, limit)]


class NewsService:
    def __init__(self, adapters: list[NewsAdapter]) -> None:
        self.adapters = adapters

    def fetch_news(self, stock_code: str, limit: int = 20) -> list[dict]:
        errors: list[str] = []
        for adapter in self.adapters:
            try:
                rows = adapter.fetch_news(stock_code, limit=limit)
                if rows:
                    return rows
            except Exception as ex:  # noqa: BLE001
                errors.append(f"{getattr(adapter, 'source_id', 'unknown')}: {ex}")
        raise RuntimeError("all news sources failed: " + "; ".join(errors))

    @classmethod
    def build_default(
        cls,
        *,
        xueqiu_cookie: str = "",
        timeout_seconds: float = 2.0,
        retry_count: int = 2,
        retry_backoff_seconds: float = 0.3,
        proxy_url: str = "",
        tradingview_proxy_url: str = "",
    ) -> "NewsService":
        cls_cfg = DataSourceConfig(
            source_id="cls_news",
            reliability_score=0.78,
            timeout_seconds=timeout_seconds,
            retry_count=retry_count,
            retry_backoff_seconds=retry_backoff_seconds,
            proxy_url=proxy_url,
        )
        trading_cfg = DataSourceConfig(
            source_id="tradingview_news",
            reliability_score=0.72,
            timeout_seconds=timeout_seconds,
            retry_count=retry_count,
            retry_backoff_seconds=retry_backoff_seconds,
            proxy_url=tradingview_proxy_url or proxy_url,
        )
        xq_cfg = DataSourceConfig(
            source_id="xueqiu_news",
            reliability_score=0.70,
            timeout_seconds=timeout_seconds,
            retry_count=retry_count,
            retry_backoff_seconds=retry_backoff_seconds,
            proxy_url=proxy_url,
        )
        adapters: list[NewsAdapter] = [
            CLSNewsAdapter(cls_cfg),
            TradingViewNewsAdapter(trading_cfg),
            XueqiuNewsAdapter(xq_cfg, cookie=xueqiu_cookie),
            MockNewsAdapter(),
        ]
        return cls(adapters)
