from __future__ import annotations

import json
from datetime import datetime, timezone

from backend.app.datasources.base.adapter import DataSourceConfig
from backend.app.datasources.base.http_client import HttpClient
from backend.app.datasources.base.utils import normalize_stock_code
from backend.app.datasources.financial.common import to_eastmoney_secid


class TTJJFundAdapter:
    source_id = "ttjj_fund"

    def __init__(self, config: DataSourceConfig, client: HttpClient | None = None) -> None:
        self.config = config
        self.client = client or HttpClient(
            timeout_seconds=config.timeout_seconds,
            retry_count=config.retry_count,
            retry_backoff_seconds=config.retry_backoff_seconds,
            proxy_url=config.proxy_url,
        )

    def fetch_fund_snapshot(self, stock_code: str) -> dict:
        code = normalize_stock_code(stock_code)
        secid = to_eastmoney_secid(code)
        url = (
            "https://push2.eastmoney.com/api/qt/stock/fflow/kline/get"
            "?lmt=1&klt=101&fields1=f1,f2,f3,f7"
            "&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63"
            f"&secid={secid}"
        )
        body = self.client.get_bytes(url)
        parsed = json.loads(body.decode("utf-8", errors="ignore"))
        klines = list((((parsed or {}).get("data") or {}).get("klines")) or [])
        if not klines:
            raise RuntimeError("ttjj fund flow empty data")
        last = str(klines[-1]).split(",")
        # Most endpoints return:
        # date, main_inflow, small_inflow, middle_inflow, large_inflow, ...
        now_iso = datetime.now(timezone.utc).isoformat()
        return {
            "stock_code": code,
            "trade_date": last[0] if len(last) > 0 else "",
            "main_inflow": float(last[1]) if len(last) > 1 else 0.0,
            "small_inflow": float(last[2]) if len(last) > 2 else 0.0,
            "middle_inflow": float(last[3]) if len(last) > 3 else 0.0,
            "large_inflow": float(last[4]) if len(last) > 4 else 0.0,
            "ts": now_iso,
            "source_id": self.source_id,
            "source_url": url,
            "reliability_score": self.config.reliability_score,
        }

