from __future__ import annotations

import json
from datetime import datetime, timezone
from urllib.parse import quote

from backend.app.datasources.base.adapter import DataSourceConfig
from backend.app.datasources.base.http_client import HttpClient
from backend.app.datasources.base.utils import normalize_stock_code


class XueqiuNewsAdapter:
    source_id = "xueqiu_news"

    def __init__(
        self,
        config: DataSourceConfig,
        *,
        cookie: str = "",
        client: HttpClient | None = None,
    ) -> None:
        self.config = config
        self.cookie = cookie
        self.client = client or HttpClient(
            timeout_seconds=config.timeout_seconds,
            retry_count=config.retry_count,
            retry_backoff_seconds=config.retry_backoff_seconds,
            proxy_url=config.proxy_url,
        )

    def fetch_news(self, stock_code: str, limit: int = 20) -> list[dict]:
        if not self.cookie.strip():
            raise RuntimeError("xueqiu news requires cookie")
        code = normalize_stock_code(stock_code)
        url = (
            "https://stock.xueqiu.com/v5/stock/realtime/news.json"
            f"?symbol={quote(code)}&count={max(1, limit)}&page=1"
        )
        body = self.client.get_bytes(
            url,
            headers={
                "Cookie": self.cookie,
                "Referer": "https://xueqiu.com/",
                "User-Agent": "StockPilotX/1.0",
            },
        )
        parsed = json.loads(body.decode("utf-8", errors="ignore"))
        rows = list((parsed.get("data") or {}).get("items") or [])
        now_iso = datetime.now(timezone.utc).isoformat()
        items: list[dict] = []
        for row in rows[: max(1, limit)]:
            title = str(row.get("title", "")).strip()
            content = str(row.get("description", "")).strip()
            if not title:
                continue
            items.append(
                {
                    "stock_code": code,
                    "title": title[:200],
                    "content": content[:2000],
                    "event_time": now_iso,
                    "source_id": self.source_id,
                    "source_url": str(row.get("target", "") or url),
                    "reliability_score": self.config.reliability_score,
                }
            )
        if not items:
            raise RuntimeError("xueqiu news empty data")
        return items

