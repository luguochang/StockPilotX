from __future__ import annotations

from .shared import *

class PortfolioWatchlistMixin:
    def watchlist_list(self, token: str) -> list[dict[str, Any]]:
        return self.web.watchlist_list(token)

    def watchlist_add(self, token: str, stock_code: str) -> dict[str, Any]:
        return self.web.watchlist_add(token, stock_code)

    def watchlist_delete(self, token: str, stock_code: str) -> dict[str, Any]:
        return self.web.watchlist_delete(token, stock_code)

    def watchlist_pool_list(self, token: str) -> list[dict[str, Any]]:
        return self.web.watchlist_pool_list(token)

    def watchlist_pool_create(self, token: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.web.watchlist_pool_create(
            token,
            pool_name=str(payload.get("pool_name", "")),
            description=str(payload.get("description", "")),
            is_default=bool(payload.get("is_default", False)),
        )

    def watchlist_pool_add_stock(self, token: str, pool_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.web.watchlist_pool_add_stock(
            token,
            pool_id=pool_id,
            stock_code=str(payload.get("stock_code", "")),
            source_filters=payload.get("source_filters", {}),
        )

    def watchlist_pool_stocks(self, token: str, pool_id: str) -> list[dict[str, Any]]:
        return self.web.watchlist_pool_stocks(token, pool_id)

    def watchlist_pool_delete_stock(self, token: str, pool_id: str, stock_code: str) -> dict[str, Any]:
        return self.web.watchlist_pool_delete_stock(token, pool_id, stock_code)

    def dashboard_overview(self, token: str) -> dict[str, Any]:
        return self.web.dashboard_overview(token)

    def portfolio_create(self, token: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.web.portfolio_create(
            token,
            portfolio_name=str(payload.get("portfolio_name", "")),
            initial_capital=float(payload.get("initial_capital", 0.0) or 0.0),
            description=str(payload.get("description", "")),
        )

    def portfolio_list(self, token: str) -> list[dict[str, Any]]:
        return self.web.portfolio_list(token)

    def portfolio_add_transaction(self, token: str, portfolio_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        return self.web.portfolio_add_transaction(
            token,
            portfolio_id=portfolio_id,
            stock_code=str(payload.get("stock_code", "")),
            transaction_type=str(payload.get("transaction_type", "")),
            quantity=float(payload.get("quantity", 0.0) or 0.0),
            price=float(payload.get("price", 0.0) or 0.0),
            fee=float(payload.get("fee", 0.0) or 0.0),
            transaction_date=str(payload.get("transaction_date", "")),
            notes=str(payload.get("notes", "")),
        )

    def _portfolio_price_map(self, stock_codes: list[str]) -> dict[str, float]:
        unique_codes = list(dict.fromkeys([str(x).strip().upper() for x in stock_codes if str(x).strip()]))
        if not unique_codes:
            return {}
        try:
            refresh_codes = [c for c in unique_codes if self._needs_quote_refresh(c)]
            if refresh_codes:
                self.ingest_market_daily(refresh_codes)
        except Exception:
            pass
        prices: dict[str, float] = {}
        for code in unique_codes:
            q = self._latest_quote(code) or {}
            p = float(q.get("price", 0.0) or 0.0)
            if p > 0:
                prices[code] = p
        return prices

    def portfolio_summary(self, token: str, portfolio_id: int) -> dict[str, Any]:
        positions = self.web.portfolio_positions(token, portfolio_id=portfolio_id)
        price_map = self._portfolio_price_map([str(x.get("stock_code", "")) for x in positions])
        return self.web.portfolio_summary(token, portfolio_id=portfolio_id, price_map=price_map)

    def portfolio_transactions(self, token: str, portfolio_id: int, *, limit: int = 200) -> list[dict[str, Any]]:
        return self.web.portfolio_transactions(token, portfolio_id=portfolio_id, limit=limit)

