from __future__ import annotations

import json
from datetime import datetime, timezone
from urllib.parse import quote

from backend.app.datasources.base.adapter import DataSourceConfig
from backend.app.datasources.base.http_client import HttpClient
from backend.app.datasources.base.utils import normalize_stock_code


class TradingViewNewsAdapter:
    source_id = "tradingview_news"

    def __init__(self, config: DataSourceConfig, client: HttpClient | None = None) -> None:
        self.config = config
        self.client = client or HttpClient(
            timeout_seconds=config.timeout_seconds,
            retry_count=config.retry_count,
            retry_backoff_seconds=config.retry_backoff_seconds,
            proxy_url=config.proxy_url,
        )

    def fetch_news(self, stock_code: str, limit: int = 20) -> list[dict]:
        code = normalize_stock_code(stock_code)
        symbol = f"SSE:{code[2:]}" if code.startswith("SH") else f"SZSE:{code[2:]}"
        url = (
            "https://news-headlines.tradingview.com/headlines/"
            f"?category=stock&lang=zh&symbol={quote(symbol)}"
        )
        body = self.client.get_bytes(url)
        parsed = json.loads(body.decode("utf-8", errors="ignore"))
        rows = list(parsed.get("items") or parsed.get("data") or [])
        now_iso = datetime.now(timezone.utc).isoformat()
        items: list[dict] = []
        for row in rows[: max(1, limit)]:
            title = str(row.get("title", "")).strip()
            content = str(row.get("storyPath", "") or row.get("description", "")).strip()
            if not title:
                continue
            source_url = str(row.get("url", "") or row.get("storyPath", "") or url)
            items.append(
                {
                    "stock_code": code,
                    "title": title[:200],
                    "content": content[:2000],
                    "event_time": now_iso,
                    "source_id": self.source_id,
                    "source_url": source_url,
                    "reliability_score": self.config.reliability_score,
                }
            )
        if not items:
            raise RuntimeError("tradingview news empty data")
        return items

