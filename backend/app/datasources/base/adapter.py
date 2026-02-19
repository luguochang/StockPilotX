from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True)
class DataSourceConfig:
    """Shared runtime config for datasource adapters."""

    source_id: str
    reliability_score: float
    timeout_seconds: float = 2.0
    retry_count: int = 2
    retry_backoff_seconds: float = 0.3
    proxy_url: str = ""
    enabled: bool = True


class QuoteAdapterProtocol(Protocol):
    source_id: str

    def fetch_quote(self, stock_code: str) -> dict[str, Any]:
        ...


class AnnouncementAdapterProtocol(Protocol):
    source_id: str

    def fetch_announcements(self, stock_code: str) -> list[dict[str, Any]]:
        ...


class HistoryAdapterProtocol(Protocol):
    source_id: str

    def fetch_daily_bars(
        self,
        stock_code: str,
        beg: str = "20240101",
        end: str = "20500101",
        limit: int = 240,
    ) -> list[dict[str, Any]]:
        ...

