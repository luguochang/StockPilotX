from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol

from backend.app.datasources.base.adapter import DataSourceConfig
from backend.app.datasources.base.utils import normalize_stock_code
from backend.app.datasources.financial.eastmoney import EastmoneyFinancialAdapter
from backend.app.datasources.financial.tushare import TushareFinancialAdapter


class FinancialAdapter(Protocol):
    source_id: str

    def fetch_financial_snapshot(self, stock_code: str) -> dict:
        ...


class MockFinancialAdapter:
    def __init__(self, source_id: str = "financial_mock", reliability_score: float = 0.60) -> None:
        self.source_id = source_id
        self.reliability_score = reliability_score

    def fetch_financial_snapshot(self, stock_code: str) -> dict:
        code = normalize_stock_code(stock_code)
        seed = sum(ord(ch) for ch in code)
        return {
            "stock_code": code,
            "report_period": "",
            "roe": round((seed % 2200) / 100, 4),
            "gross_margin": round((seed % 5500) / 100, 4),
            "revenue_yoy": round(((seed % 3000) - 1500) / 100, 4),
            "net_profit_yoy": round(((seed % 2600) - 1300) / 100, 4),
            "asset_liability_ratio": round((seed % 9000) / 100, 4),
            "pe_ttm": round((seed % 6000) / 100, 4),
            "pb_mrq": round((seed % 1200) / 100, 4),
            "ts": datetime.now(timezone.utc).isoformat(),
            "source_id": self.source_id,
            "source_url": f"https://{self.source_id}.example.com/{code}",
            "reliability_score": self.reliability_score,
            "source_note": "deterministic_mock",
        }


class FinancialService:
    def __init__(self, adapters: list[FinancialAdapter]) -> None:
        self.adapters = adapters

    def get_financial_snapshot(self, stock_code: str) -> dict:
        errors: list[str] = []
        for adapter in self.adapters:
            try:
                return adapter.fetch_financial_snapshot(stock_code)
            except Exception as ex:  # noqa: BLE001
                errors.append(f"{getattr(adapter, 'source_id', 'unknown')}: {ex}")
        raise RuntimeError("all financial sources failed: " + "; ".join(errors))

    @classmethod
    def build_default(
        cls,
        *,
        tushare_token: str = "",
        timeout_seconds: float = 2.0,
        retry_count: int = 2,
        retry_backoff_seconds: float = 0.3,
        proxy_url: str = "",
    ) -> "FinancialService":
        tushare_cfg = DataSourceConfig(
            source_id="tushare_financial",
            reliability_score=0.86,
            timeout_seconds=timeout_seconds,
            retry_count=retry_count,
            retry_backoff_seconds=retry_backoff_seconds,
            proxy_url=proxy_url,
        )
        eastmoney_cfg = DataSourceConfig(
            source_id="eastmoney_financial",
            reliability_score=0.84,
            timeout_seconds=timeout_seconds,
            retry_count=retry_count,
            retry_backoff_seconds=retry_backoff_seconds,
            proxy_url=proxy_url,
        )
        adapters: list[FinancialAdapter] = [
            TushareFinancialAdapter(tushare_cfg, token=tushare_token),
            EastmoneyFinancialAdapter(eastmoney_cfg),
            MockFinancialAdapter(),
        ]
        return cls(adapters)

