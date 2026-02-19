from __future__ import annotations

import re

from backend.app.datasources.base.adapter import DataSourceConfig
from backend.app.datasources.base.http_client import HttpClient
from backend.app.datasources.base.utils import decode_response
from backend.app.datasources.quote.common import build_quote, safe_get, to_api_code, to_float
from backend.app.datasources.quote.models import Quote


class SinaQuoteAdapter:
    source_id = "sina"

    def __init__(self, config: DataSourceConfig, client: HttpClient | None = None) -> None:
        self.config = config
        self.client = client or HttpClient(
            timeout_seconds=config.timeout_seconds,
            retry_count=config.retry_count,
            retry_backoff_seconds=config.retry_backoff_seconds,
            proxy_url=config.proxy_url,
        )

    def fetch_quote(self, stock_code: str) -> Quote:
        api_code = to_api_code(stock_code)
        url = f"https://hq.sinajs.cn/list={api_code}"
        payload = self.client.get_bytes(
            url,
            headers={
                "Referer": "https://finance.sina.com.cn",
                "User-Agent": "StockPilotX/1.0",
            },
        )
        text = decode_response(payload)
        matched = re.search(r'"([^"]+)"', text)
        if not matched:
            raise RuntimeError("sina parse failed: empty payload")
        fields = matched.group(1).split(",")
        if len(fields) < 10:
            raise RuntimeError("sina parse failed: field too short")
        prev_close = to_float(safe_get(fields, 2))
        price = to_float(safe_get(fields, 3))
        pct_change = round(((price - prev_close) / prev_close) * 100, 4) if prev_close else 0.0
        return build_quote(
            stock_code=stock_code,
            price=price,
            pct_change=pct_change,
            volume=to_float(safe_get(fields, 8)),
            turnover=to_float(safe_get(fields, 9)),
            source_id=self.source_id,
            source_url=url,
            reliability_score=self.config.reliability_score,
        )

