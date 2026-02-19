from __future__ import annotations

import json
from datetime import datetime, timezone

from backend.app.datasources.base.adapter import DataSourceConfig
from backend.app.datasources.base.http_client import HttpClient
from backend.app.datasources.base.utils import normalize_stock_code
from backend.app.datasources.financial.common import to_float, to_tushare_code


class TushareFinancialAdapter:
    source_id = "tushare_financial"
    source_url = "http://api.tushare.pro"

    def __init__(
        self,
        config: DataSourceConfig,
        *,
        token: str = "",
        client: HttpClient | None = None,
    ) -> None:
        self.config = config
        self.token = token
        self.client = client or HttpClient(
            timeout_seconds=config.timeout_seconds,
            retry_count=config.retry_count,
            retry_backoff_seconds=config.retry_backoff_seconds,
            proxy_url=config.proxy_url,
        )

    def fetch_financial_snapshot(self, stock_code: str) -> dict:
        if not self.token.strip():
            raise RuntimeError("tushare token is required")

        payload = {
            "api_name": "fina_indicator",
            "token": self.token,
            "params": {"ts_code": to_tushare_code(stock_code), "limit": 1},
            "fields": (
                "ts_code,end_date,roe,grossprofit_margin,tr_yoy,netprofit_yoy,"
                "debt_to_assets,current_ratio,quick_ratio"
            ),
        }
        body = self.client.post_json_bytes(self.source_url, payload)
        parsed = json.loads(body.decode("utf-8", errors="ignore"))
        response_code = int(parsed.get("code", -1))
        if response_code != 0:
            raise RuntimeError(f"tushare response error: {parsed.get('msg', 'unknown')}")

        data = parsed.get("data") or {}
        fields = list(data.get("fields") or [])
        items = list(data.get("items") or [])
        if not fields or not items:
            raise RuntimeError("tushare empty data")

        row = dict(zip(fields, items[0]))
        now_iso = datetime.now(timezone.utc).isoformat()
        return {
            "stock_code": normalize_stock_code(stock_code),
            "report_period": str(row.get("end_date", "")),
            "roe": to_float(row.get("roe")),
            "gross_margin": to_float(row.get("grossprofit_margin")),
            "revenue_yoy": to_float(row.get("tr_yoy")),
            "net_profit_yoy": to_float(row.get("netprofit_yoy")),
            "asset_liability_ratio": to_float(row.get("debt_to_assets")),
            "pe_ttm": 0.0,
            "pb_mrq": 0.0,
            "ts": now_iso,
            "source_id": self.source_id,
            "source_url": self.source_url,
            "reliability_score": self.config.reliability_score,
            "source_note": "fina_indicator",
        }
