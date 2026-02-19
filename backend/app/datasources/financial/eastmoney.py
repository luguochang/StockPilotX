from __future__ import annotations

import json
from datetime import datetime, timezone

from backend.app.datasources.base.adapter import DataSourceConfig
from backend.app.datasources.base.http_client import HttpClient
from backend.app.datasources.base.utils import normalize_stock_code
from backend.app.datasources.financial.common import to_eastmoney_secid, to_float


class EastmoneyFinancialAdapter:
    source_id = "eastmoney_financial"

    def __init__(self, config: DataSourceConfig, client: HttpClient | None = None) -> None:
        self.config = config
        self.client = client or HttpClient(
            timeout_seconds=config.timeout_seconds,
            retry_count=config.retry_count,
            retry_backoff_seconds=config.retry_backoff_seconds,
            proxy_url=config.proxy_url,
        )

    def fetch_financial_snapshot(self, stock_code: str) -> dict:
        secid = to_eastmoney_secid(stock_code)
        url = (
            "https://push2.eastmoney.com/api/qt/stock/get"
            "?invt=2&fltt=2"
            "&fields=f57,f58,f162,f167,f116,f117,f127,f128,f129,f130,f152"
            f"&secid={secid}"
        )
        body = self.client.get_bytes(url)
        parsed = json.loads(body.decode("utf-8", errors="ignore"))
        data = parsed.get("data") or {}
        if not data:
            raise RuntimeError("eastmoney financial empty data")

        now_iso = datetime.now(timezone.utc).isoformat()
        return {
            "stock_code": normalize_stock_code(stock_code),
            "report_period": "",
            "roe": 0.0,
            "gross_margin": 0.0,
            "revenue_yoy": to_float(data.get("f127")),
            "net_profit_yoy": to_float(data.get("f128")),
            "asset_liability_ratio": to_float(data.get("f130")),
            "pe_ttm": to_float(data.get("f162")),
            "pb_mrq": to_float(data.get("f167")),
            "ts": now_iso,
            "source_id": self.source_id,
            "source_url": url,
            "reliability_score": self.config.reliability_score,
            "source_note": "push2_stock_get",
        }

