from __future__ import annotations

from typing import Any

from backend.app.datasources.base.utils import normalize_stock_code


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def to_tushare_code(stock_code: str) -> str:
    code = normalize_stock_code(stock_code)
    if code.startswith("SH"):
        return f"{code[2:]}.SH"
    return f"{code[2:]}.SZ"


def to_eastmoney_secid(stock_code: str) -> str:
    code = normalize_stock_code(stock_code)
    if code.startswith("SH"):
        return f"1.{code[2:]}"
    return f"0.{code[2:]}"

