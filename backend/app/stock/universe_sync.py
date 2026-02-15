from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from backend.app.web.store import WebStore


@dataclass(slots=True)
class UniverseSyncResult:
    total_stocks: int
    total_industry_links: int
    source: str


def _normalize_code(code: str) -> str:
    c = (code or "").strip().upper().replace(".", "")
    if c.startswith(("SH", "SZ", "BJ")):
        return c
    if c.startswith(("60", "68")):
        return f"SH{c}"
    if c.startswith(("00", "30")):
        return f"SZ{c}"
    if c.startswith(("4", "8")):
        return f"BJ{c}"
    return c


def _is_a_share_code(code: str) -> bool:
    raw = code[2:] if code.startswith(("SH", "SZ", "BJ")) else code
    if len(raw) != 6 or not raw.isdigit():
        return False
    return raw.startswith(("60", "68", "00", "30", "4", "8"))


def _exchange_from_code(code: str) -> str:
    if code.startswith("SH"):
        return "SH"
    if code.startswith("SZ"):
        return "SZ"
    if code.startswith("BJ"):
        return "BJ"
    return "UNKNOWN"


def _listing_board_from_code(code: str) -> str:
    raw = code[2:] if code.startswith(("SH", "SZ", "BJ")) else code
    if raw.startswith("68"):
        return "科创板"
    if raw.startswith("30"):
        return "创业板"
    if raw.startswith(("60", "00")):
        return "主板"
    if raw.startswith(("4", "8")):
        return "北交所"
    return "其他"


def _market_tier(exchange: str, listing_board: str) -> str:
    if listing_board == "创业板":
        return "创业板"
    if listing_board == "科创板":
        return "科创板"
    if listing_board == "北交所":
        return "北交所"
    if exchange == "SH" and listing_board == "主板":
        return "上证主板"
    if exchange == "SZ" and listing_board == "主板":
        return "深证主板"
    return f"{exchange}-{listing_board}"


class AShareUniverseSyncService:
    """A 股主数据同步（股票清单 + 行业层级）。"""

    def __init__(self, store: WebStore) -> None:
        self.store = store

    def sync_from_akshare(self) -> UniverseSyncResult:
        os.environ.setdefault("TQDM_DISABLE", "1")
        try:
            import akshare as ak  # type: ignore
        except Exception as ex:  # noqa: BLE001
            raise RuntimeError("akshare not installed. run: pip install akshare") from ex

        base_df = ak.stock_info_a_code_name()
        stocks: dict[str, dict[str, Any]] = {}
        for row in base_df.to_dict("records"):
            code = _normalize_code(str(row.get("code", "")))
            if not code:
                continue
            if not _is_a_share_code(code):
                continue
            stocks[code] = {
                "stock_code": code,
                "stock_name": str(row.get("name", "")).strip() or code,
                "exchange": _exchange_from_code(code),
                "listing_board": _listing_board_from_code(code),
                "market_tier": _market_tier(_exchange_from_code(code), _listing_board_from_code(code)),
                "industry_l1": "",
                "source": "akshare:stock_info_a_code_name",
            }

        industry_links: list[dict[str, str]] = []
        board_df = ak.stock_board_industry_name_em()
        for board in board_df.to_dict("records"):
            industry_name = str(board.get("板块名称", "")).strip()
            if not industry_name:
                continue
            try:
                cons_df = ak.stock_board_industry_cons_em(symbol=industry_name)
            except Exception:
                continue
            for item in cons_df.to_dict("records"):
                code = _normalize_code(str(item.get("代码", "")))
                if not code:
                    continue
                if not _is_a_share_code(code):
                    continue
                if code not in stocks:
                    stocks[code] = {
                        "stock_code": code,
                        "stock_name": str(item.get("名称", "")).strip() or code,
                        "exchange": _exchange_from_code(code),
                        "listing_board": _listing_board_from_code(code),
                        "market_tier": _market_tier(_exchange_from_code(code), _listing_board_from_code(code)),
                        "industry_l1": "",
                        "source": "akshare:stock_board_industry_cons_em",
                    }
                if not stocks[code]["industry_l1"]:
                    stocks[code]["industry_l1"] = industry_name
                industry_links.append(
                    {
                        "stock_code": code,
                        "industry_l1": industry_name,
                        "source": "akshare:stock_board_industry_cons_em",
                    }
                )

        self._replace_universe(list(stocks.values()), industry_links)
        return UniverseSyncResult(
            total_stocks=len(stocks),
            total_industry_links=len(industry_links),
            source="akshare",
        )

    def search(
        self,
        *,
        keyword: str = "",
        exchange: str = "",
        listing_board: str = "",
        industry_l1: str = "",
        market_tier: str = "",
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        if keyword.strip():
            where.append("(u.stock_code LIKE ? OR u.stock_name LIKE ?)")
            k = f"%{keyword.strip().upper()}%"
            params.extend([k, f"%{keyword.strip()}%"])
        if exchange.strip():
            where.append("u.exchange = ?")
            params.append(exchange.strip().upper())
        if listing_board.strip():
            where.append("u.listing_board = ?")
            params.append(listing_board.strip())
        if market_tier.strip():
            where.append("u.market_tier = ?")
            params.append(market_tier.strip())
        if industry_l1.strip():
            where.append("u.industry_l1 = ?")
            params.append(industry_l1.strip())

        sql = """
            SELECT
                u.stock_code,
                u.stock_name,
                u.exchange,
                u.market_tier,
                u.listing_board,
                u.industry_l1,
                u.updated_at
            FROM stock_universe u
        """
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY u.stock_code LIMIT ?"
        params.append(max(1, min(int(limit), 200)))
        return self.store.query_all(sql, tuple(params))

    def filters(self) -> dict[str, list[str]]:
        exchanges = self.store.query_all("SELECT DISTINCT exchange AS v FROM stock_universe ORDER BY exchange")
        boards = self.store.query_all("SELECT DISTINCT listing_board AS v FROM stock_universe ORDER BY listing_board")
        tiers = self.store.query_all("SELECT DISTINCT market_tier AS v FROM stock_universe WHERE market_tier <> '' ORDER BY market_tier")
        industries = self.store.query_all("SELECT DISTINCT industry_l1 AS v FROM stock_universe WHERE industry_l1 <> '' ORDER BY industry_l1")
        return {
            "exchange": [x["v"] for x in exchanges],
            "market_tier": [x["v"] for x in tiers],
            "listing_board": [x["v"] for x in boards],
            "industry_l1": [x["v"] for x in industries],
        }

    def _replace_universe(self, stocks: list[dict[str, Any]], industry_links: list[dict[str, str]]) -> None:
        self.store.execute("DELETE FROM stock_universe_industry_map")
        self.store.execute("DELETE FROM stock_universe")
        for s in stocks:
            self.store.execute(
                """
                INSERT INTO stock_universe (stock_code, stock_name, exchange, market_tier, listing_board, industry_l1, source, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    str(s.get("stock_code", "")),
                    str(s.get("stock_name", "")),
                    str(s.get("exchange", "")),
                    str(s.get("market_tier", "")),
                    str(s.get("listing_board", "")),
                    str(s.get("industry_l1", "")),
                    str(s.get("source", "")),
                ),
            )
        for m in industry_links:
            self.store.execute(
                """
                INSERT INTO stock_universe_industry_map (stock_code, industry_l1, source, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (m["stock_code"], m["industry_l1"], m["source"]),
            )
