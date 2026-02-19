from __future__ import annotations

from typing import Any

from backend.app.datasources.base.utils import normalize_stock_code
from backend.app.datasources.quote.models import Quote, now_utc


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_get(values: list[str], index: int, default: str = "0") -> str:
    return values[index] if 0 <= index < len(values) else default


def to_api_code(stock_code: str) -> str:
    code = normalize_stock_code(stock_code)
    if code.startswith("SH"):
        return "sh" + code[2:]
    return "sz" + code[2:]


def to_netease_code(stock_code: str) -> str:
    code = normalize_stock_code(stock_code)
    if code.startswith("SH"):
        return "0" + code[2:]
    return "1" + code[2:]


def to_xueqiu_symbol(stock_code: str) -> str:
    return normalize_stock_code(stock_code)


def build_quote(
    *,
    stock_code: str,
    price: float,
    pct_change: float,
    volume: float,
    turnover: float,
    source_id: str,
    source_url: str,
    reliability_score: float,
) -> Quote:
    return Quote(
        stock_code=normalize_stock_code(stock_code),
        price=round(price, 4),
        pct_change=round(pct_change, 4),
        volume=round(volume, 4),
        turnover=round(turnover, 4),
        ts=now_utc(),
        source_id=source_id,
        source_url=source_url,
        reliability_score=reliability_score,
    )

