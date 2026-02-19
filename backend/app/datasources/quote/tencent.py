from __future__ import annotations

import re

from backend.app.datasources.base.adapter import DataSourceConfig
from backend.app.datasources.base.http_client import HttpClient
from backend.app.datasources.base.utils import decode_response
from backend.app.datasources.quote.common import build_quote, safe_get, to_api_code, to_float
from backend.app.datasources.quote.models import Quote


class TencentQuoteAdapter:
    source_id = "tencent"

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
        url = f"https://qt.gtimg.cn/q={api_code}"
        payload = self.client.get_bytes(url)
        text = decode_response(payload)
        matched = re.search(r'"([^"]+)"', text)
        if not matched:
            raise RuntimeError("tencent parse failed: empty payload")
        fields = matched.group(1).split("~")
        if len(fields) < 4:
            raise RuntimeError("tencent parse failed: field too short")
        return build_quote(
            stock_code=stock_code,
            price=to_float(safe_get(fields, 3)),
            pct_change=to_float(safe_get(fields, 32)),
            volume=to_float(safe_get(fields, 36)),
            turnover=to_float(safe_get(fields, 37)),
            source_id=self.source_id,
            source_url=url,
            reliability_score=self.config.reliability_score,
        )

