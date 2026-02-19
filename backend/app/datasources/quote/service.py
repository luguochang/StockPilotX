from __future__ import annotations

from typing import Any, Protocol

from backend.app.datasources.base.adapter import DataSourceConfig
from backend.app.datasources.base.utils import normalize_stock_code
from backend.app.datasources.quote.common import build_quote
from backend.app.datasources.quote.models import Quote
from backend.app.datasources.quote.netease import NeteaseQuoteAdapter
from backend.app.datasources.quote.sina import SinaQuoteAdapter
from backend.app.datasources.quote.tencent import TencentQuoteAdapter
from backend.app.datasources.quote.xueqiu import XueqiuQuoteAdapter


class QuoteAdapter(Protocol):
    source_id: str

    def fetch_quote(self, stock_code: str) -> Quote:
        ...


class MockQuoteAdapter:
    """Deterministic fallback adapter to keep local flows available offline."""

    def __init__(self, source_id: str, reliability_score: float) -> None:
        self.source_id = source_id
        self.reliability_score = reliability_score

    def fetch_quote(self, stock_code: str) -> Quote:
        normalized = normalize_stock_code(stock_code)
        seed = sum(ord(ch) for ch in normalized)
        price = round((seed % 5000) / 100 + 5, 2)
        pct_change = round(((seed % 1000) - 500) / 100, 2)
        return build_quote(
            stock_code=normalized,
            price=price,
            pct_change=pct_change,
            volume=float(100000 + seed),
            turnover=float(1000000 + seed * 10),
            source_id=self.source_id,
            source_url=f"https://{self.source_id}.example.com/{normalized}",
            reliability_score=self.reliability_score,
        )


class QuoteService:
    """Quote service with explicit fallback chain."""

    def __init__(self, adapters: list[QuoteAdapter]) -> None:
        self.adapters = adapters

    def get_quote(self, stock_code: str) -> Quote:
        errors: list[str] = []
        for adapter in self.adapters:
            try:
                return adapter.fetch_quote(stock_code)
            except Exception as ex:  # noqa: BLE001
                errors.append(f"{getattr(adapter, 'source_id', 'unknown')}: {ex}")
        raise RuntimeError("all quote sources failed: " + "; ".join(errors))

    @classmethod
    def build_default(
        cls,
        *,
        xueqiu_cookie: str = "",
        timeout_seconds: float = 2.0,
        retry_count: int = 2,
        retry_backoff_seconds: float = 0.3,
        proxy_url: str = "",
    ) -> "QuoteService":
        tencent_cfg = DataSourceConfig(
            source_id="tencent",
            reliability_score=0.85,
            timeout_seconds=timeout_seconds,
            retry_count=retry_count,
            retry_backoff_seconds=retry_backoff_seconds,
            proxy_url=proxy_url,
        )
        netease_cfg = DataSourceConfig(
            source_id="netease",
            reliability_score=0.82,
            timeout_seconds=timeout_seconds,
            retry_count=retry_count,
            retry_backoff_seconds=retry_backoff_seconds,
            proxy_url=proxy_url,
        )
        sina_cfg = DataSourceConfig(
            source_id="sina",
            reliability_score=0.80,
            timeout_seconds=timeout_seconds,
            retry_count=retry_count,
            retry_backoff_seconds=retry_backoff_seconds,
            proxy_url=proxy_url,
        )
        xueqiu_cfg = DataSourceConfig(
            source_id="xueqiu",
            reliability_score=0.78,
            timeout_seconds=timeout_seconds,
            retry_count=retry_count,
            retry_backoff_seconds=retry_backoff_seconds,
            proxy_url=proxy_url,
        )

        adapters: list[QuoteAdapter] = [
            TencentQuoteAdapter(tencent_cfg),
            NeteaseQuoteAdapter(netease_cfg),
            SinaQuoteAdapter(sina_cfg),
            XueqiuQuoteAdapter(xueqiu_cfg, cookie=xueqiu_cookie),
            # Keep deterministic mock fallback for offline/dev mode.
            MockQuoteAdapter("tencent_mock", 0.70),
            MockQuoteAdapter("netease_mock", 0.68),
            MockQuoteAdapter("sina_mock", 0.66),
        ]
        return cls(adapters)

    def debug_snapshot(self) -> list[dict[str, Any]]:
        """Provide a lightweight, serializable snapshot for ops diagnostics."""
        return [
            {
                "source_id": str(getattr(adapter, "source_id", "")),
                "adapter": adapter.__class__.__name__,
            }
            for adapter in self.adapters
        ]

