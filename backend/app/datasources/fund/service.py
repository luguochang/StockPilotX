from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol

from backend.app.datasources.base.adapter import DataSourceConfig
from backend.app.datasources.base.utils import normalize_stock_code
from backend.app.datasources.fund.ttjj import TTJJFundAdapter


class FundAdapter(Protocol):
    source_id: str

    def fetch_fund_snapshot(self, stock_code: str) -> dict:
        ...


class MockFundAdapter:
    def __init__(self, source_id: str = "fund_mock", reliability_score: float = 0.56) -> None:
        self.source_id = source_id
        self.reliability_score = reliability_score

    def fetch_fund_snapshot(self, stock_code: str) -> dict:
        code = normalize_stock_code(stock_code)
        seed = sum(ord(ch) for ch in code)
        now_iso = datetime.now(timezone.utc).isoformat()
        return {
            "stock_code": code,
            "trade_date": now_iso[:10],
            "main_inflow": float((seed % 20000) - 10000),
            "small_inflow": float((seed % 15000) - 7500),
            "middle_inflow": float((seed % 12000) - 6000),
            "large_inflow": float((seed % 25000) - 12500),
            "ts": now_iso,
            "source_id": self.source_id,
            "source_url": f"https://{self.source_id}.example.com/{code}",
            "reliability_score": self.reliability_score,
        }


class FundService:
    def __init__(self, adapters: list[FundAdapter]) -> None:
        self.adapters = adapters

    def fetch_fund_snapshot(self, stock_code: str) -> dict:
        errors: list[str] = []
        for adapter in self.adapters:
            try:
                return adapter.fetch_fund_snapshot(stock_code)
            except Exception as ex:  # noqa: BLE001
                errors.append(f"{getattr(adapter, 'source_id', 'unknown')}: {ex}")
        raise RuntimeError("all fund sources failed: " + "; ".join(errors))

    @classmethod
    def build_default(
        cls,
        *,
        timeout_seconds: float = 2.0,
        retry_count: int = 2,
        retry_backoff_seconds: float = 0.3,
        proxy_url: str = "",
    ) -> "FundService":
        cfg = DataSourceConfig(
            source_id="ttjj_fund",
            reliability_score=0.76,
            timeout_seconds=timeout_seconds,
            retry_count=retry_count,
            retry_backoff_seconds=retry_backoff_seconds,
            proxy_url=proxy_url,
        )
        adapters: list[FundAdapter] = [TTJJFundAdapter(cfg), MockFundAdapter()]
        return cls(adapters)

