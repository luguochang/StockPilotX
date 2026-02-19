from __future__ import annotations

import json
from datetime import datetime, timezone
from urllib.parse import quote

from backend.app.datasources.base.adapter import DataSourceConfig
from backend.app.datasources.base.http_client import HttpClient
from backend.app.datasources.base.utils import normalize_stock_code


class EastmoneyResearchAdapter:
    source_id = "eastmoney_research"

    def __init__(self, config: DataSourceConfig, client: HttpClient | None = None) -> None:
        self.config = config
        self.client = client or HttpClient(
            timeout_seconds=config.timeout_seconds,
            retry_count=config.retry_count,
            retry_backoff_seconds=config.retry_backoff_seconds,
            proxy_url=config.proxy_url,
        )

    def fetch_reports(self, stock_code: str, limit: int = 20) -> list[dict]:
        code = normalize_stock_code(stock_code)
        code_plain = code[2:]
        url = (
            "https://datacenter-web.eastmoney.com/api/data/v1/get"
            "?reportName=RPT_RESEARCH_REPORT_NEW"
            "&columns=SECURITY_CODE,SECURITY_NAME_ABBR,TITLE,AUTHOR_NAME,ORG_NAME,PUBLISH_DATE,REPORT_URL"
            f"&filter={quote(f'(SECURITY_CODE=\"{code_plain}\")')}"
            f"&pageSize={max(1, limit)}&pageNumber=1"
        )
        body = self.client.get_bytes(url)
        parsed = json.loads(body.decode("utf-8", errors="ignore"))
        rows = list((parsed.get("result") or {}).get("data") or [])
        now_iso = datetime.now(timezone.utc).isoformat()
        reports: list[dict] = []
        for row in rows[: max(1, limit)]:
            title = str(row.get("TITLE", "")).strip()
            if not title:
                continue
            reports.append(
                {
                    "stock_code": code,
                    "title": title[:200],
                    "content": title[:200],
                    "published_at": str(row.get("PUBLISH_DATE", "")) or now_iso,
                    "author": str(row.get("AUTHOR_NAME", "")),
                    "org_name": str(row.get("ORG_NAME", "")),
                    "source_id": self.source_id,
                    "source_url": str(row.get("REPORT_URL", "") or url),
                    "reliability_score": self.config.reliability_score,
                }
            )
        if not reports:
            raise RuntimeError("eastmoney research empty data")
        return reports

