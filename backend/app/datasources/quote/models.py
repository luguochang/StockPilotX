from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class Quote:
    """Canonical quote object consumed by ingestion service."""

    stock_code: str
    price: float
    pct_change: float
    volume: float
    turnover: float
    ts: datetime
    source_id: str
    source_url: str
    reliability_score: float

