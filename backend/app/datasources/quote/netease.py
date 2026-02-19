from __future__ import annotations

import json
from urllib.parse import quote

from backend.app.datasources.base.adapter import DataSourceConfig
from backend.app.datasources.base.http_client import HttpClient
from backend.app.datasources.base.utils import decode_response
from backend.app.datasources.quote.common import build_quote, to_float, to_netease_code
from backend.app.datasources.quote.models import Quote


class NeteaseQuoteAdapter:
    source_id = "netease"

    def __init__(self, config: DataSourceConfig, client: HttpClient | None = None) -> None:
        self.config = config
        self.client = client or HttpClient(
            timeout_seconds=config.timeout_seconds,
            retry_count=config.retry_count,
            retry_backoff_seconds=config.retry_backoff_seconds,
            proxy_url=config.proxy_url,
        )

    def fetch_quote(self, stock_code: str) -> Quote:
        api_code = to_netease_code(stock_code)
        url = f"https://api.money.126.net/data/feed/{quote(api_code)},money.api?callback=_ntes_quote_callback"
        payload = self.client.get_bytes(url)
        text = decode_response(payload)
        start = text.find("(")
        end = text.rfind(")")
        if start < 0 or end <= start:
            raise RuntimeError("netease parse failed: invalid callback json")
        item = json.loads(text[start + 1 : end]).get(api_code)
        if not item:
            raise RuntimeError("netease parse failed: symbol not found")
        return build_quote(
            stock_code=stock_code,
            price=to_float(item.get("price")),
            pct_change=to_float(item.get("percent")),
            volume=to_float(item.get("volume")),
            turnover=to_float(item.get("turnover")),
            source_id=self.source_id,
            source_url=url,
            reliability_score=self.config.reliability_score,
        )

