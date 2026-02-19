from __future__ import annotations

import json
from urllib.parse import quote

from backend.app.datasources.base.adapter import DataSourceConfig
from backend.app.datasources.base.http_client import HttpClient
from backend.app.datasources.base.utils import decode_response
from backend.app.datasources.quote.common import build_quote, to_float, to_xueqiu_symbol
from backend.app.datasources.quote.models import Quote


class XueqiuQuoteAdapter:
    source_id = "xueqiu"

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

    def fetch_quote(self, stock_code: str) -> Quote:
        if not self.cookie.strip():
            raise RuntimeError("xueqiu adapter disabled: missing cookie")
        symbol = to_xueqiu_symbol(stock_code)
        url = f"https://stock.xueqiu.com/v5/stock/realtime/quotec.json?symbol={quote(symbol)}"
        payload = self.client.get_bytes(
            url,
            headers={
                "Cookie": self.cookie,
                "Referer": "https://xueqiu.com/",
                "User-Agent": "StockPilotX/1.0",
            },
        )
        data = json.loads(decode_response(payload))
        items = data.get("data", [])
        if not items:
            raise RuntimeError("xueqiu parse failed: empty data")
        item = items[0]
        return build_quote(
            stock_code=stock_code,
            price=to_float(item.get("current")),
            pct_change=to_float(item.get("percent")),
            volume=to_float(item.get("volume")),
            turnover=to_float(item.get("amount")),
            source_id=self.source_id,
            source_url=url,
            reliability_score=self.config.reliability_score,
        )

