from __future__ import annotations

import json
from datetime import datetime, timezone
from urllib.parse import quote

from backend.app.datasources.base.adapter import DataSourceConfig
from backend.app.datasources.base.http_client import HttpClient


class EastmoneyMacroAdapter:
    source_id = "eastmoney_macro"
    base_url = "https://datacenter-web.eastmoney.com/api/data/v1/get"

    def __init__(self, config: DataSourceConfig, client: HttpClient | None = None) -> None:
        self.config = config
        self.client = client or HttpClient(
            timeout_seconds=config.timeout_seconds,
            retry_count=config.retry_count,
            retry_backoff_seconds=config.retry_backoff_seconds,
            proxy_url=config.proxy_url,
        )

    def fetch_macro_indicators(self, limit: int = 20) -> list[dict]:
        # Keep endpoint list short and stable for initial rollout.
        tasks = [
            ("RPT_ECONOMY_CPI", "CPI", "REPORT_DATE,VALUE"),
            ("RPT_ECONOMY_PPI", "PPI", "REPORT_DATE,VALUE"),
            ("RPT_ECONOMY_PMI", "PMI", "REPORT_DATE,VALUE"),
        ]
        rows: list[dict] = []
        now_iso = datetime.now(timezone.utc).isoformat()
        for report_name, metric_name, columns in tasks:
            url = (
                f"{self.base_url}?reportName={report_name}"
                f"&columns={quote(columns)}&pageSize={max(1, limit)}&pageNumber=1"
            )
            body = self.client.get_bytes(url)
            parsed = json.loads(body.decode("utf-8", errors="ignore"))
            items = list((parsed.get("result") or {}).get("data") or [])
            for item in items[: max(1, limit)]:
                value = item.get("VALUE")
                report_date = str(item.get("REPORT_DATE", "")).strip()
                rows.append(
                    {
                        "metric_name": metric_name,
                        "metric_value": str(value) if value is not None else "",
                        "report_date": report_date,
                        "event_time": report_date or now_iso,
                        "source_id": self.source_id,
                        "source_url": url,
                        "reliability_score": self.config.reliability_score,
                    }
                )
        if not rows:
            raise RuntimeError("eastmoney macro empty data")
        return rows

