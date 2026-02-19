from __future__ import annotations

import json
from datetime import datetime, timezone

from backend.app.datasources.base.adapter import DataSourceConfig
from backend.app.datasources.base.http_client import HttpClient
from backend.app.datasources.base.utils import normalize_stock_code


class CLSNewsAdapter:
    source_id = "cls_news"
    source_url = "https://www.cls.cn/nodeapi/telegraphList"

    def __init__(self, config: DataSourceConfig, client: HttpClient | None = None) -> None:
        self.config = config
        self.client = client or HttpClient(
            timeout_seconds=config.timeout_seconds,
            retry_count=config.retry_count,
            retry_backoff_seconds=config.retry_backoff_seconds,
            proxy_url=config.proxy_url,
        )

    def fetch_news(self, stock_code: str, limit: int = 20) -> list[dict]:
        body = self.client.get_bytes(
            self.source_url,
            headers={
                "Referer": "https://www.cls.cn/",
                "User-Agent": "StockPilotX/1.0",
            },
        )
        parsed = json.loads(body.decode("utf-8", errors="ignore"))
        roll = list(((parsed.get("data") or {}).get("roll_data") or []))
        now_iso = datetime.now(timezone.utc).isoformat()
        code = normalize_stock_code(stock_code)
        items: list[dict] = []
        for row in roll[: max(1, limit)]:
            title = str(row.get("title", "")).strip()
            content = str(row.get("content", "")).strip()
            if not title and not content:
                continue
            items.append(
                {
                    "stock_code": code,
                    "title": title[:200],
                    "content": content[:2000],
                    "event_time": now_iso,
                    "source_id": self.source_id,
                    "source_url": str(row.get("shareurl", "") or self.source_url),
                    "reliability_score": self.config.reliability_score,
                }
            )
        if not items:
            raise RuntimeError("cls news empty data")
        return items

