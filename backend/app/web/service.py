from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

from backend.app.stock.universe_sync import AShareUniverseSyncService
from backend.app.web.security import create_token, decode_token, hash_password, verify_password
from backend.app.web.store import WebStore


class WebAppService:
    """完整 Web 应用领域服务。"""

    def __init__(self, store: WebStore, jwt_secret: str, jwt_expire_seconds: int, auth_bypass: bool = True) -> None:
        self.store = store
        self.jwt_secret = jwt_secret
        self.jwt_expire_seconds = jwt_expire_seconds
        self.auth_bypass = auth_bypass
        self.universe = AShareUniverseSyncService(store)
        # 默认白名单来源：用于“上传后自动生效”策略，避免初始状态全部进入 review。
        self._seed_default_rag_source_policies()

    def _seed_default_rag_source_policies(self) -> None:
        """写入默认 RAG 来源白名单策略（幂等）。"""
        defaults = [
            ("exchange_announcement", 1, 0.95, 1),
            ("cninfo", 1, 0.95, 1),
            ("eastmoney", 1, 0.85, 1),
            ("sse", 1, 0.95, 1),
            ("szse", 1, 0.95, 1),
            ("user_upload", 0, 0.70, 1),
        ]
        for source, auto_approve, trust_score, enabled in defaults:
            self.store.execute(
                """
                INSERT OR IGNORE INTO rag_doc_source_policy (source, auto_approve, trust_score, enabled)
                VALUES (?, ?, ?, ?)
                """,
                (source, auto_approve, trust_score, enabled),
            )

    # ----------------- Auth / RBAC -----------------
    def auth_register(self, username: str, password: str, tenant_name: str | None = None) -> dict[str, Any]:
        tenant_name = tenant_name or f"{username}_tenant"
        try:
            cur = self.store.execute(
                "INSERT INTO user (username, password_hash) VALUES (?, ?)",
                (username, hash_password(password)),
            )
            user_id = int(cur.lastrowid)
            t_cur = self.store.execute("INSERT OR IGNORE INTO tenant (name) VALUES (?)", (tenant_name,))
            if t_cur.lastrowid:
                tenant_id = int(t_cur.lastrowid)
            else:
                tenant_id = int(self.store.query_one("SELECT id FROM tenant WHERE name = ?", (tenant_name,))["id"])
            self.store.execute(
                "INSERT OR REPLACE INTO user_tenant_role (user_id, tenant_id, role) VALUES (?, ?, ?)",
                (user_id, tenant_id, "admin"),
            )
            self._audit(username, "register", True, "ok")
            return {"user_id": user_id, "tenant_id": tenant_id, "username": username, "tenant_name": tenant_name}
        except Exception as ex:  # noqa: BLE001
            self._audit(username, "register", False, str(ex))
            raise

    def auth_login(self, username: str, password: str) -> dict[str, Any]:
        user = self.store.query_one("SELECT id, username, password_hash FROM user WHERE username = ?", (username,))
        if not user or not verify_password(password, user["password_hash"]):
            self._audit(username, "login", False, "invalid credentials")
            raise ValueError("invalid credentials")
        role_row = self.store.query_one(
            "SELECT tenant_id, role FROM user_tenant_role WHERE user_id = ? ORDER BY tenant_id LIMIT 1",
            (user["id"],),
        )
        if not role_row:
            raise ValueError("user has no tenant role")
        token = create_token(
            {
                "user_id": user["id"],
                "username": user["username"],
                "tenant_id": role_row["tenant_id"],
                "role": role_row["role"],
            },
            secret=self.jwt_secret,
            expire_seconds=self.jwt_expire_seconds,
        )
        self._audit(username, "login", True, "ok")
        return {"access_token": token, "token_type": "bearer"}

    def auth_me(self, token: str) -> dict[str, Any]:
        if self.auth_bypass and not (token or "").strip():
            return {"user_id": 1, "username": "anonymous_dev", "tenant_id": 1, "role": "admin"}
        try:
            payload = decode_token(token, self.jwt_secret)
            return payload
        except Exception:
            if self.auth_bypass:
                return {"user_id": 1, "username": "anonymous_dev", "tenant_id": 1, "role": "admin"}
            raise

    def _audit(self, username: str, action: str, success: bool, detail: str) -> None:
        self.store.execute(
            "INSERT INTO auth_audit_log (username, action, success, detail) VALUES (?, ?, ?, ?)",
            (username, action, int(success), detail),
        )

    def _normalize_pool_name(self, value: str) -> str:
        name = str(value or "").strip()
        if not name:
            return ""
        # Some terminals/clients may downgrade non-ASCII input into '?'.
        if set(name) == {"?"}:
            return "Unnamed Pool"
        return name

    def require_role(self, token: str, allowed: set[str]) -> dict[str, Any]:
        if self.auth_bypass:
            return self.auth_me(token)
        payload = self.auth_me(token)
        if payload.get("role") not in allowed:
            raise PermissionError("forbidden")
        return payload

    # ----------------- Watchlist / Dashboard -----------------
    def watchlist_list(self, token: str) -> list[dict[str, Any]]:
        me = self.auth_me(token)
        return self.store.query_all(
            "SELECT stock_code, created_at FROM watchlist WHERE user_id = ? AND tenant_id = ? ORDER BY id DESC",
            (me["user_id"], me["tenant_id"]),
        )

    def watchlist_add(self, token: str, stock_code: str) -> dict[str, Any]:
        me = self.auth_me(token)
        self.store.execute(
            "INSERT OR IGNORE INTO watchlist (user_id, tenant_id, stock_code) VALUES (?, ?, ?)",
            (me["user_id"], me["tenant_id"], stock_code.upper()),
        )
        return {"status": "ok", "stock_code": stock_code.upper()}

    def watchlist_delete(self, token: str, stock_code: str) -> dict[str, Any]:
        me = self.auth_me(token)
        self.store.execute(
            "DELETE FROM watchlist WHERE user_id = ? AND tenant_id = ? AND stock_code = ?",
            (me["user_id"], me["tenant_id"], stock_code.upper()),
        )
        return {"status": "ok", "stock_code": stock_code.upper()}

    def watchlist_pool_list(self, token: str) -> list[dict[str, Any]]:
        me = self.auth_me(token)
        pools = self.store.query_all(
            """
            SELECT pool_id, pool_name, description, is_default, created_at, updated_at
            FROM watchlist_pool
            WHERE user_id = ? AND tenant_id = ?
            ORDER BY is_default DESC, updated_at DESC
            """,
            (me["user_id"], me["tenant_id"]),
        )
        for row in pools:
            cnt = self.store.query_one("SELECT COUNT(1) AS cnt FROM watchlist_pool_stock WHERE pool_id = ?", (row["pool_id"],))
            row["stock_count"] = int((cnt or {}).get("cnt", 0))
            row["is_default"] = bool(int(row.get("is_default", 0)))
        return pools

    def watchlist_pool_create(self, token: str, pool_name: str, description: str = "", is_default: bool = False) -> dict[str, Any]:
        me = self.auth_me(token)
        name = self._normalize_pool_name(pool_name)
        if not name:
            raise ValueError("pool_name is required")
        pool_id = str(uuid.uuid4())
        if is_default:
            self.store.execute(
                "UPDATE watchlist_pool SET is_default = 0, updated_at = CURRENT_TIMESTAMP WHERE user_id = ? AND tenant_id = ?",
                (me["user_id"], me["tenant_id"]),
            )
        self.store.execute(
            """
            INSERT INTO watchlist_pool (pool_id, user_id, tenant_id, pool_name, description, is_default)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (pool_id, me["user_id"], me["tenant_id"], name, str(description or "").strip(), int(bool(is_default))),
        )
        return {"status": "ok", "pool_id": pool_id, "pool_name": name}

    def watchlist_pool_add_stock(
        self,
        token: str,
        pool_id: str,
        stock_code: str,
        source_filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        me = self.auth_me(token)
        pool = self.store.query_one(
            "SELECT pool_id FROM watchlist_pool WHERE pool_id = ? AND user_id = ? AND tenant_id = ?",
            (pool_id, me["user_id"], me["tenant_id"]),
        )
        if not pool:
            raise ValueError("pool not found")
        normalized = str(stock_code or "").strip().upper()
        if not normalized:
            raise ValueError("stock_code is required")
        self.store.execute(
            """
            INSERT OR IGNORE INTO watchlist_pool_stock (pool_id, stock_code, source_filters_json)
            VALUES (?, ?, ?)
            """,
            (pool_id, normalized, json.dumps(source_filters or {}, ensure_ascii=False)),
        )
        self.store.execute("UPDATE watchlist_pool SET updated_at = CURRENT_TIMESTAMP WHERE pool_id = ?", (pool_id,))
        return {"status": "ok", "pool_id": pool_id, "stock_code": normalized}

    def watchlist_pool_stocks(self, token: str, pool_id: str) -> list[dict[str, Any]]:
        me = self.auth_me(token)
        pool = self.store.query_one(
            "SELECT pool_id FROM watchlist_pool WHERE pool_id = ? AND user_id = ? AND tenant_id = ?",
            (pool_id, me["user_id"], me["tenant_id"]),
        )
        if not pool:
            raise ValueError("pool not found")
        rows = self.store.query_all(
            """
            SELECT p.stock_code, p.source_filters_json, p.created_at,
                   u.stock_name, u.exchange, u.exchange_name, u.market_tier, u.listing_board, u.board_code,
                   u.industry_l1, u.industry_l2, u.industry_l3
            FROM watchlist_pool_stock p
            LEFT JOIN stock_universe u ON u.stock_code = p.stock_code
            WHERE p.pool_id = ?
            ORDER BY p.created_at DESC
            """,
            (pool_id,),
        )
        for row in rows:
            row["source_filters"] = self._json_loads_or(row.get("source_filters_json", "{}"), {})
        return rows

    def watchlist_pool_codes(self, token: str, pool_id: str) -> list[str]:
        rows = self.watchlist_pool_stocks(token, pool_id)
        return [str(row.get("stock_code", "")).upper() for row in rows if str(row.get("stock_code", "")).strip()]

    def watchlist_pool_delete_stock(self, token: str, pool_id: str, stock_code: str) -> dict[str, Any]:
        me = self.auth_me(token)
        pool = self.store.query_one(
            "SELECT pool_id FROM watchlist_pool WHERE pool_id = ? AND user_id = ? AND tenant_id = ?",
            (pool_id, me["user_id"], me["tenant_id"]),
        )
        if not pool:
            raise ValueError("pool not found")
        normalized = str(stock_code or "").strip().upper()
        self.store.execute("DELETE FROM watchlist_pool_stock WHERE pool_id = ? AND stock_code = ?", (pool_id, normalized))
        self.store.execute("UPDATE watchlist_pool SET updated_at = CURRENT_TIMESTAMP WHERE pool_id = ?", (pool_id,))
        return {"status": "ok", "pool_id": pool_id, "stock_code": normalized}

    def dashboard_overview(self, token: str) -> dict[str, Any]:
        me = self.auth_me(token)
        watchlist = self.watchlist_list(token)
        latest_reports = self.store.query_all(
            """
            SELECT report_id, stock_code, report_type, created_at
            FROM report_index
            WHERE user_id = ? AND tenant_id = ?
            ORDER BY created_at DESC
            LIMIT 5
            """,
            (me["user_id"], me["tenant_id"]),
        )
        open_alerts = self.store.query_one(
            "SELECT COUNT(1) AS cnt FROM alert_event WHERE status = 'open'",
            (),
        )["cnt"]
        return {
            "watchlist_count": len(watchlist),
            "watchlist": watchlist,
            "latest_reports": latest_reports,
            "open_alert_count": open_alerts,
        }

    # ----------------- Query Hub -----------------
    def query_history_add(
        self,
        token: str,
        *,
        question: str,
        stock_codes: list[str],
        trace_id: str,
        intent: str,
        cache_hit: bool,
        latency_ms: int,
        summary: str = "",
        error: str = "",
    ) -> dict[str, Any]:
        """Persist one Query Hub execution record for audit and user history view."""
        me = self.auth_me(token)
        self.store.execute(
            """
            INSERT INTO query_history
            (user_id, tenant_id, question, stock_codes_json, trace_id, intent, cache_hit, latency_ms, summary, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                me["user_id"],
                me["tenant_id"],
                str(question or "").strip()[:1000],
                json.dumps([str(x).strip().upper() for x in stock_codes if str(x).strip()], ensure_ascii=False),
                str(trace_id or "").strip()[:120],
                str(intent or "").strip()[:64],
                int(bool(cache_hit)),
                max(0, int(latency_ms)),
                str(summary or "").strip()[:400],
                str(error or "").strip()[:300],
            ),
        )
        return {"status": "ok"}

    def _normalize_query_history_time(self, value: str, *, end_of_day: bool) -> str:
        """Normalize query-history time filter to SQLite comparable timestamp text."""
        raw = str(value or "").strip()
        if not raw:
            return ""
        # Support YYYY-MM-DD for quick filtering in frontend.
        if len(raw) == 10:
            suffix = "23:59:59" if end_of_day else "00:00:00"
            raw = f"{raw} {suffix}"
        try:
            datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
        except ValueError as ex:
            raise ValueError("created_from/created_to must be YYYY-MM-DD or YYYY-MM-DD HH:MM:SS") from ex
        return raw

    def query_history_list(
        self,
        token: str,
        *,
        limit: int = 50,
        stock_code: str = "",
        created_from: str = "",
        created_to: str = "",
    ) -> list[dict[str, Any]]:
        """List latest query records for current user with optional stock/time filters."""
        me = self.auth_me(token)
        safe_limit = max(1, min(200, int(limit)))
        code_filter = str(stock_code or "").strip().upper()
        from_ts = self._normalize_query_history_time(created_from, end_of_day=False)
        to_ts = self._normalize_query_history_time(created_to, end_of_day=True)
        if from_ts and to_ts and from_ts > to_ts:
            raise ValueError("created_from must be earlier than or equal to created_to")

        conditions = ["user_id = ?", "tenant_id = ?"]
        params: list[Any] = [me["user_id"], me["tenant_id"]]
        if code_filter:
            conditions.append("INSTR(UPPER(stock_codes_json), ?) > 0")
            params.append(f"\"{code_filter}\"")
        if from_ts:
            conditions.append("created_at >= ?")
            params.append(from_ts)
        if to_ts:
            conditions.append("created_at <= ?")
            params.append(to_ts)
        params.append(safe_limit)
        rows = self.store.query_all(
            f"""
            SELECT id, question, stock_codes_json, trace_id, intent, cache_hit, latency_ms, summary, error, created_at
            FROM query_history
            WHERE {' AND '.join(conditions)}
            ORDER BY id DESC
            LIMIT ?
            """,
            tuple(params),
        )
        for row in rows:
            row["stock_codes"] = self._json_loads_or(row.get("stock_codes_json"), [])
            row.pop("stock_codes_json", None)
            row["cache_hit"] = bool(int(row.get("cache_hit", 0) or 0))
        return rows

    def query_history_clear(self, token: str) -> dict[str, Any]:
        """Delete current user's query history."""
        me = self.auth_me(token)
        self.store.execute(
            "DELETE FROM query_history WHERE user_id = ? AND tenant_id = ?",
            (me["user_id"], me["tenant_id"]),
        )
        return {"status": "ok"}

    # ----------------- Portfolio -----------------
    def portfolio_create(
        self,
        token: str,
        *,
        portfolio_name: str,
        initial_capital: float,
        description: str = "",
    ) -> dict[str, Any]:
        me = self.auth_me(token)
        name = str(portfolio_name or "").strip()
        if not name:
            raise ValueError("portfolio_name is required")
        capital = float(initial_capital or 0.0)
        if capital <= 0:
            raise ValueError("initial_capital must be > 0")
        cur = self.store.execute(
            """
            INSERT INTO portfolio
            (user_id, tenant_id, portfolio_name, description, initial_capital, current_value, total_profit_loss, total_profit_loss_pct)
            VALUES (?, ?, ?, ?, ?, ?, 0, 0)
            """,
            (me["user_id"], me["tenant_id"], name, str(description or "").strip(), capital, capital),
        )
        return {"status": "ok", "portfolio_id": int(cur.lastrowid), "portfolio_name": name, "initial_capital": capital}

    def portfolio_list(self, token: str) -> list[dict[str, Any]]:
        me = self.auth_me(token)
        return self.store.query_all(
            """
            SELECT id AS portfolio_id, portfolio_name, description, initial_capital, current_value,
                   total_profit_loss, total_profit_loss_pct, created_at, updated_at
            FROM portfolio
            WHERE user_id = ? AND tenant_id = ?
            ORDER BY updated_at DESC, id DESC
            """,
            (me["user_id"], me["tenant_id"]),
        )

    def _portfolio_ensure_owned(self, token: str, portfolio_id: int) -> dict[str, Any]:
        me = self.auth_me(token)
        row = self.store.query_one(
            "SELECT * FROM portfolio WHERE id = ? AND user_id = ? AND tenant_id = ?",
            (int(portfolio_id), me["user_id"], me["tenant_id"]),
        )
        if not row:
            raise ValueError("portfolio not found")
        return row

    def portfolio_add_transaction(
        self,
        token: str,
        *,
        portfolio_id: int,
        stock_code: str,
        transaction_type: str,
        quantity: float,
        price: float,
        fee: float = 0.0,
        transaction_date: str = "",
        notes: str = "",
    ) -> dict[str, Any]:
        _ = self._portfolio_ensure_owned(token, portfolio_id)
        side = str(transaction_type or "").strip().lower()
        if side not in {"buy", "sell"}:
            raise ValueError("transaction_type must be buy or sell")
        code = str(stock_code or "").strip().upper()
        if not code:
            raise ValueError("stock_code is required")
        qty = float(quantity or 0.0)
        px = float(price or 0.0)
        fee_value = max(0.0, float(fee or 0.0))
        if qty <= 0 or px <= 0:
            raise ValueError("quantity and price must be > 0")
        # amount is cash impact: buy negative, sell positive.
        gross = qty * px
        amount = -(gross + fee_value) if side == "buy" else (gross - fee_value)

        position = self.store.query_one(
            "SELECT id, quantity, avg_cost, current_price FROM portfolio_position WHERE portfolio_id = ? AND stock_code = ?",
            (int(portfolio_id), code),
        )
        if side == "buy":
            if position:
                old_qty = float(position.get("quantity", 0.0) or 0.0)
                old_cost = float(position.get("avg_cost", 0.0) or 0.0)
                new_qty = old_qty + qty
                new_cost = ((old_qty * old_cost) + (qty * px)) / new_qty
                self.store.execute(
                    """
                    UPDATE portfolio_position
                    SET quantity = ?, avg_cost = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (new_qty, new_cost, int(position["id"])),
                )
            else:
                self.store.execute(
                    """
                    INSERT INTO portfolio_position (portfolio_id, stock_code, quantity, avg_cost, current_price, market_value)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (int(portfolio_id), code, qty, px, px, qty * px),
                )
        else:
            if not position:
                raise ValueError("position not found for sell")
            old_qty = float(position.get("quantity", 0.0) or 0.0)
            if qty > old_qty + 1e-9:
                raise ValueError("sell quantity exceeds current position")
            new_qty = old_qty - qty
            if new_qty <= 1e-9:
                self.store.execute(
                    "DELETE FROM portfolio_position WHERE id = ?",
                    (int(position["id"]),),
                )
            else:
                self.store.execute(
                    """
                    UPDATE portfolio_position
                    SET quantity = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (new_qty, int(position["id"])),
                )

        tx_date = str(transaction_date or "").strip() or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cur = self.store.execute(
            """
            INSERT INTO portfolio_transaction
            (portfolio_id, stock_code, transaction_type, quantity, price, fee, amount, notes, transaction_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (int(portfolio_id), code, side, qty, px, fee_value, amount, str(notes or "").strip()[:400], tx_date),
        )
        return {"status": "ok", "transaction_id": int(cur.lastrowid), "portfolio_id": int(portfolio_id)}

    def portfolio_positions(self, token: str, *, portfolio_id: int) -> list[dict[str, Any]]:
        _ = self._portfolio_ensure_owned(token, portfolio_id)
        return self.store.query_all(
            """
            SELECT id AS position_id, stock_code, quantity, avg_cost, current_price, market_value,
                   profit_loss, profit_loss_pct, weight, updated_at
            FROM portfolio_position
            WHERE portfolio_id = ?
            ORDER BY market_value DESC, stock_code ASC
            """,
            (int(portfolio_id),),
        )

    def portfolio_transactions(self, token: str, *, portfolio_id: int, limit: int = 200) -> list[dict[str, Any]]:
        _ = self._portfolio_ensure_owned(token, portfolio_id)
        safe_limit = max(1, min(2000, int(limit)))
        return self.store.query_all(
            """
            SELECT id AS transaction_id, stock_code, transaction_type, quantity, price, fee, amount, notes, transaction_date, created_at
            FROM portfolio_transaction
            WHERE portfolio_id = ?
            ORDER BY transaction_date DESC, id DESC
            LIMIT ?
            """,
            (int(portfolio_id), safe_limit),
        )

    def portfolio_revalue(self, token: str, *, portfolio_id: int, price_map: dict[str, float] | None = None) -> dict[str, Any]:
        portfolio = self._portfolio_ensure_owned(token, portfolio_id)
        prices = {str(k).strip().upper(): float(v) for k, v in (price_map or {}).items() if str(k).strip()}
        positions = self.store.query_all(
            "SELECT id, stock_code, quantity, avg_cost, current_price FROM portfolio_position WHERE portfolio_id = ?",
            (int(portfolio_id),),
        )
        total_mv = 0.0
        total_cost = 0.0
        for row in positions:
            code = str(row.get("stock_code", "")).upper()
            qty = float(row.get("quantity", 0.0) or 0.0)
            avg_cost = float(row.get("avg_cost", 0.0) or 0.0)
            current_price = float(prices.get(code, float(row.get("current_price", 0.0) or avg_cost)))
            mv = qty * current_price
            pnl = (current_price - avg_cost) * qty
            pnl_pct = ((current_price / avg_cost - 1.0) * 100.0) if avg_cost > 0 else 0.0
            total_mv += mv
            total_cost += qty * avg_cost
            self.store.execute(
                """
                UPDATE portfolio_position
                SET current_price = ?, market_value = ?, profit_loss = ?, profit_loss_pct = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (current_price, mv, pnl, pnl_pct, int(row["id"])),
            )

        # Compute cash from signed transaction amount.
        tx_row = self.store.query_one(
            "SELECT COALESCE(SUM(amount), 0) AS cash_delta FROM portfolio_transaction WHERE portfolio_id = ?",
            (int(portfolio_id),),
        ) or {"cash_delta": 0}
        initial = float(portfolio.get("initial_capital", 0.0) or 0.0)
        cash = initial + float(tx_row.get("cash_delta", 0.0) or 0.0)
        current_value = cash + total_mv
        pnl_total = current_value - initial
        pnl_pct_total = ((current_value / initial - 1.0) * 100.0) if initial > 0 else 0.0
        self.store.execute(
            """
            UPDATE portfolio
            SET current_value = ?, total_profit_loss = ?, total_profit_loss_pct = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (current_value, pnl_total, pnl_pct_total, int(portfolio_id)),
        )
        if total_mv > 0:
            self.store.execute(
                """
                UPDATE portfolio_position
                SET weight = market_value / ?
                WHERE portfolio_id = ?
                """,
                (total_mv, int(portfolio_id)),
            )
        else:
            self.store.execute("UPDATE portfolio_position SET weight = 0 WHERE portfolio_id = ?", (int(portfolio_id),))
        updated = self._portfolio_ensure_owned(token, portfolio_id)
        return {
            "portfolio_id": int(portfolio_id),
            "initial_capital": initial,
            "cash": cash,
            "market_value": total_mv,
            "current_value": float(updated.get("current_value", 0.0) or 0.0),
            "total_profit_loss": float(updated.get("total_profit_loss", 0.0) or 0.0),
            "total_profit_loss_pct": float(updated.get("total_profit_loss_pct", 0.0) or 0.0),
        }

    def portfolio_summary(self, token: str, *, portfolio_id: int, price_map: dict[str, float] | None = None) -> dict[str, Any]:
        _ = self.portfolio_revalue(token, portfolio_id=portfolio_id, price_map=price_map)
        portfolio = self._portfolio_ensure_owned(token, portfolio_id)
        positions = self.portfolio_positions(token, portfolio_id=portfolio_id)
        return {
            "portfolio_id": int(portfolio_id),
            "portfolio_name": str(portfolio.get("portfolio_name", "")),
            "description": str(portfolio.get("description", "")),
            "initial_capital": float(portfolio.get("initial_capital", 0.0) or 0.0),
            "current_value": float(portfolio.get("current_value", 0.0) or 0.0),
            "total_profit_loss": float(portfolio.get("total_profit_loss", 0.0) or 0.0),
            "total_profit_loss_pct": float(portfolio.get("total_profit_loss_pct", 0.0) or 0.0),
            "positions": positions,
            "position_count": len(positions),
        }

    # ----------------- Investment Journal -----------------
    def _journal_ensure_owned(self, token: str, journal_id: int) -> dict[str, Any]:
        """Verify journal ownership to prevent cross-tenant data leakage."""
        me = self.auth_me(token)
        row = self.store.query_one(
            "SELECT * FROM investment_journal WHERE id = ? AND user_id = ? AND tenant_id = ?",
            (int(journal_id), me["user_id"], me["tenant_id"]),
        )
        if not row:
            raise ValueError("journal not found")
        return row

    def _journal_normalize_tags(self, tags: Any) -> list[str]:
        """Normalize tags for consistent querying and avoid unbounded payload growth."""
        if not isinstance(tags, list):
            return []
        normalized: list[str] = []
        for raw in tags:
            text = str(raw or "").strip()
            if not text:
                continue
            text = text[:32]
            if text in normalized:
                continue
            normalized.append(text)
            if len(normalized) >= 12:
                break
        return normalized

    def _journal_normalize_type(self, journal_type: str) -> str:
        kind = str(journal_type or "").strip().lower()
        if kind not in {"decision", "reflection", "learning"}:
            raise ValueError("journal_type must be one of decision/reflection/learning")
        return kind

    def _journal_serialize_row(self, row: dict[str, Any]) -> dict[str, Any]:
        if not row:
            return {}
        payload = dict(row)
        payload["journal_id"] = int(payload.pop("id"))
        payload["related_portfolio_id"] = (
            int(payload.get("related_portfolio_id"))
            if payload.get("related_portfolio_id") is not None
            else None
        )
        payload["tags"] = self._json_loads_or(payload.get("tags_json", "[]"), [])
        payload.pop("tags_json", None)
        return payload

    def _journal_ai_reflection_serialize_row(self, row: dict[str, Any]) -> dict[str, Any]:
        if not row:
            return {}
        payload = dict(row)
        payload["ai_reflection_id"] = int(payload.pop("id"))
        payload["journal_id"] = int(payload.get("journal_id", 0) or 0)
        payload["confidence"] = float(payload.get("confidence", 0.0) or 0.0)
        payload["insights"] = self._json_loads_or(payload.get("insights_json", "[]"), [])
        payload["lessons"] = self._json_loads_or(payload.get("lessons_json", "[]"), [])
        payload.pop("insights_json", None)
        payload.pop("lessons_json", None)
        return payload

    def _journal_normalize_brief_items(self, items: Any, *, max_items: int = 6, max_len: int = 200) -> list[str]:
        """Normalize generated insight/lesson list to keep payload stable for UI rendering."""
        if not isinstance(items, list):
            return []
        normalized: list[str] = []
        for raw in items:
            text = str(raw or "").strip()
            if not text:
                continue
            text = text[:max_len]
            if text in normalized:
                continue
            normalized.append(text)
            if len(normalized) >= max_items:
                break
        return normalized

    def journal_create(
        self,
        token: str,
        *,
        journal_type: str,
        title: str,
        content: str,
        stock_code: str = "",
        decision_type: str = "",
        related_research_id: str = "",
        related_portfolio_id: int | None = None,
        tags: Any = None,
        sentiment: str = "",
    ) -> dict[str, Any]:
        me = self.auth_me(token)
        kind = self._journal_normalize_type(journal_type)
        title_text = str(title or "").strip()
        content_text = str(content or "").strip()
        if not title_text:
            raise ValueError("title is required")
        if not content_text:
            raise ValueError("content is required")
        code = str(stock_code or "").strip().upper()
        normalized_tags = self._journal_normalize_tags(tags)
        related_portfolio = None if related_portfolio_id in (None, "") else int(related_portfolio_id)

        cur = self.store.execute(
            """
            INSERT INTO investment_journal
            (user_id, tenant_id, journal_type, title, content, stock_code, decision_type,
             related_research_id, related_portfolio_id, tags_json, sentiment, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                me["user_id"],
                me["tenant_id"],
                kind,
                title_text[:200],
                content_text[:8000],
                code,
                str(decision_type or "").strip()[:40],
                str(related_research_id or "").strip()[:120],
                related_portfolio,
                json.dumps(normalized_tags, ensure_ascii=False),
                str(sentiment or "").strip()[:24],
            ),
        )
        row = self.store.query_one("SELECT * FROM investment_journal WHERE id = ?", (int(cur.lastrowid),))
        return self._journal_serialize_row(row or {})

    def journal_get(self, token: str, *, journal_id: int) -> dict[str, Any]:
        row = self._journal_ensure_owned(token, journal_id)
        return self._journal_serialize_row(row)

    def journal_list(
        self,
        token: str,
        *,
        journal_type: str = "",
        stock_code: str = "",
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        me = self.auth_me(token)
        safe_limit = max(1, min(200, int(limit)))
        safe_offset = max(0, int(offset))
        conditions = ["user_id = ?", "tenant_id = ?"]
        params: list[Any] = [me["user_id"], me["tenant_id"]]

        kind_filter = str(journal_type or "").strip().lower()
        if kind_filter:
            kind_filter = self._journal_normalize_type(kind_filter)
            conditions.append("journal_type = ?")
            params.append(kind_filter)
        code_filter = str(stock_code or "").strip().upper()
        if code_filter:
            conditions.append("stock_code = ?")
            params.append(code_filter)
        params.extend([safe_limit, safe_offset])
        rows = self.store.query_all(
            f"""
            SELECT id, journal_type, title, content, stock_code, decision_type, related_research_id,
                   related_portfolio_id, tags_json, sentiment, created_at, updated_at
            FROM investment_journal
            WHERE {' AND '.join(conditions)}
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            tuple(params),
        )
        return [self._journal_serialize_row(row) for row in rows]

    def journal_reflection_add(
        self,
        token: str,
        *,
        journal_id: int,
        reflection_content: str,
        ai_insights: str = "",
        lessons_learned: str = "",
    ) -> dict[str, Any]:
        _ = self._journal_ensure_owned(token, journal_id)
        reflection = str(reflection_content or "").strip()
        if not reflection:
            raise ValueError("reflection_content is required")
        cur = self.store.execute(
            """
            INSERT INTO journal_reflection
            (journal_id, reflection_content, ai_insights, lessons_learned)
            VALUES (?, ?, ?, ?)
            """,
            (
                int(journal_id),
                reflection[:6000],
                str(ai_insights or "").strip()[:4000],
                str(lessons_learned or "").strip()[:2000],
            ),
        )
        row = self.store.query_one(
            """
            SELECT id AS reflection_id, journal_id, reflection_content, ai_insights, lessons_learned, created_at
            FROM journal_reflection
            WHERE id = ?
            """,
            (int(cur.lastrowid),),
        )
        return row or {}

    def journal_reflection_list(self, token: str, *, journal_id: int, limit: int = 50) -> list[dict[str, Any]]:
        _ = self._journal_ensure_owned(token, journal_id)
        safe_limit = max(1, min(200, int(limit)))
        return self.store.query_all(
            """
            SELECT id AS reflection_id, journal_id, reflection_content, ai_insights, lessons_learned, created_at
            FROM journal_reflection
            WHERE journal_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(journal_id), safe_limit),
        )

    def journal_find_by_related_research(self, token: str, *, related_research_id: str) -> dict[str, Any]:
        """Find one journal by external link key for idempotent auto-link scenarios."""
        me = self.auth_me(token)
        key = str(related_research_id or "").strip()
        if not key:
            raise ValueError("related_research_id is required")
        row = self.store.query_one(
            """
            SELECT id, journal_type, title, content, stock_code, decision_type, related_research_id,
                   related_portfolio_id, tags_json, sentiment, created_at, updated_at
            FROM investment_journal
            WHERE user_id = ? AND tenant_id = ? AND related_research_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (me["user_id"], me["tenant_id"], key),
        )
        return self._journal_serialize_row(row or {})

    def journal_ai_reflection_upsert(
        self,
        token: str,
        *,
        journal_id: int,
        status: str,
        summary: str,
        insights: Any = None,
        lessons: Any = None,
        confidence: float = 0.0,
        provider: str = "",
        model: str = "",
        trace_id: str = "",
        error_code: str = "",
        error_message: str = "",
    ) -> dict[str, Any]:
        _ = self._journal_ensure_owned(token, journal_id)
        safe_status = str(status or "").strip().lower()
        if safe_status not in {"ready", "fallback", "failed"}:
            raise ValueError("status must be one of ready/fallback/failed")
        normalized_insights = self._journal_normalize_brief_items(insights)
        normalized_lessons = self._journal_normalize_brief_items(lessons)
        self.store.execute(
            """
            INSERT INTO journal_ai_reflection
            (journal_id, status, summary, insights_json, lessons_json, confidence, provider, model, trace_id, error_code, error_message, generated_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(journal_id) DO UPDATE SET
                status = excluded.status,
                summary = excluded.summary,
                insights_json = excluded.insights_json,
                lessons_json = excluded.lessons_json,
                confidence = excluded.confidence,
                provider = excluded.provider,
                model = excluded.model,
                trace_id = excluded.trace_id,
                error_code = excluded.error_code,
                error_message = excluded.error_message,
                generated_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                int(journal_id),
                safe_status,
                str(summary or "").strip()[:2400],
                json.dumps(normalized_insights, ensure_ascii=False),
                json.dumps(normalized_lessons, ensure_ascii=False),
                max(0.0, min(1.0, float(confidence or 0.0))),
                str(provider or "").strip()[:80],
                str(model or "").strip()[:120],
                str(trace_id or "").strip()[:80],
                str(error_code or "").strip()[:64],
                str(error_message or "").strip()[:400],
            ),
        )
        return self.journal_ai_reflection_get(token, journal_id=journal_id)

    def journal_ai_reflection_get(self, token: str, *, journal_id: int) -> dict[str, Any]:
        _ = self._journal_ensure_owned(token, journal_id)
        row = self.store.query_one(
            """
            SELECT id, journal_id, status, summary, insights_json, lessons_json, confidence, provider, model,
                   trace_id, error_code, error_message, generated_at, updated_at
            FROM journal_ai_reflection
            WHERE journal_id = ?
            LIMIT 1
            """,
            (int(journal_id),),
        )
        return self._journal_ai_reflection_serialize_row(row or {})

    def journal_insights_rows(self, token: str, *, days: int = 90, limit: int = 400) -> list[dict[str, Any]]:
        """Fetch journals with reflection coverage marks for higher-level insight aggregation."""
        me = self.auth_me(token)
        safe_days = max(7, min(3650, int(days)))
        safe_limit = max(20, min(2000, int(limit)))
        since_expr = f"-{safe_days} day"
        rows = self.store.query_all(
            """
            SELECT
                j.id,
                j.journal_type,
                j.title,
                j.content,
                j.stock_code,
                j.decision_type,
                j.related_research_id,
                j.related_portfolio_id,
                j.tags_json,
                j.sentiment,
                j.created_at,
                j.updated_at,
                (
                    SELECT COUNT(1)
                    FROM journal_reflection r
                    WHERE r.journal_id = j.id
                ) AS reflection_count,
                CASE
                    WHEN EXISTS (
                        SELECT 1
                        FROM journal_ai_reflection ar
                        WHERE ar.journal_id = j.id
                    )
                    THEN 1
                    ELSE 0
                END AS has_ai_reflection
            FROM investment_journal j
            WHERE j.user_id = ?
              AND j.tenant_id = ?
              AND j.created_at >= datetime('now', ?)
            ORDER BY j.id DESC
            LIMIT ?
            """,
            (me["user_id"], me["tenant_id"], since_expr, safe_limit),
        )
        payload: list[dict[str, Any]] = []
        for row in rows:
            item = self._journal_serialize_row(row)
            item["reflection_count"] = int(row.get("reflection_count", 0) or 0)
            item["has_ai_reflection"] = bool(int(row.get("has_ai_reflection", 0) or 0))
            payload.append(item)
        return payload

    def journal_insights_timeline(self, token: str, *, days: int = 90) -> list[dict[str, Any]]:
        """Build per-day activity counts for journal/reflection/ai-reflection events."""
        me = self.auth_me(token)
        safe_days = max(7, min(3650, int(days)))
        since_expr = f"-{safe_days} day"
        journal_rows = self.store.query_all(
            """
            SELECT DATE(created_at) AS day, COUNT(1) AS journal_count
            FROM investment_journal
            WHERE user_id = ? AND tenant_id = ? AND created_at >= datetime('now', ?)
            GROUP BY DATE(created_at)
            """,
            (me["user_id"], me["tenant_id"], since_expr),
        )
        reflection_rows = self.store.query_all(
            """
            SELECT DATE(r.created_at) AS day, COUNT(1) AS reflection_count
            FROM journal_reflection r
            INNER JOIN investment_journal j ON j.id = r.journal_id
            WHERE j.user_id = ? AND j.tenant_id = ? AND r.created_at >= datetime('now', ?)
            GROUP BY DATE(r.created_at)
            """,
            (me["user_id"], me["tenant_id"], since_expr),
        )
        ai_rows = self.store.query_all(
            """
            SELECT DATE(ar.generated_at) AS day, COUNT(1) AS ai_reflection_count
            FROM journal_ai_reflection ar
            INNER JOIN investment_journal j ON j.id = ar.journal_id
            WHERE j.user_id = ? AND j.tenant_id = ? AND ar.generated_at >= datetime('now', ?)
            GROUP BY DATE(ar.generated_at)
            """,
            (me["user_id"], me["tenant_id"], since_expr),
        )
        timeline_map: dict[str, dict[str, Any]] = {}
        for row in journal_rows:
            day = str(row.get("day", "")).strip()
            if not day:
                continue
            timeline_map.setdefault(day, {"day": day, "journal_count": 0, "reflection_count": 0, "ai_reflection_count": 0})
            timeline_map[day]["journal_count"] = int(row.get("journal_count", 0) or 0)
        for row in reflection_rows:
            day = str(row.get("day", "")).strip()
            if not day:
                continue
            timeline_map.setdefault(day, {"day": day, "journal_count": 0, "reflection_count": 0, "ai_reflection_count": 0})
            timeline_map[day]["reflection_count"] = int(row.get("reflection_count", 0) or 0)
        for row in ai_rows:
            day = str(row.get("day", "")).strip()
            if not day:
                continue
            timeline_map.setdefault(day, {"day": day, "journal_count": 0, "reflection_count": 0, "ai_reflection_count": 0})
            timeline_map[day]["ai_reflection_count"] = int(row.get("ai_reflection_count", 0) or 0)
        return [timeline_map[key] for key in sorted(timeline_map.keys())]

    def journal_ai_generation_log_add(
        self,
        token: str,
        *,
        journal_id: int,
        status: str,
        provider: str = "",
        model: str = "",
        trace_id: str = "",
        error_code: str = "",
        error_message: str = "",
        latency_ms: int = 0,
    ) -> dict[str, Any]:
        _ = self._journal_ensure_owned(token, journal_id)
        safe_status = str(status or "").strip().lower()
        if safe_status not in {"ready", "fallback", "failed"}:
            raise ValueError("status must be one of ready/fallback/failed")
        cur = self.store.execute(
            """
            INSERT INTO journal_ai_generation_log
            (journal_id, status, provider, model, trace_id, error_code, error_message, latency_ms, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                int(journal_id),
                safe_status,
                str(provider or "").strip()[:80],
                str(model or "").strip()[:120],
                str(trace_id or "").strip()[:80],
                str(error_code or "").strip()[:64],
                str(error_message or "").strip()[:400],
                max(0, int(latency_ms or 0)),
            ),
        )
        row = self.store.query_one(
            """
            SELECT id AS log_id, journal_id, status, provider, model, trace_id, error_code, error_message,
                   latency_ms, generated_at
            FROM journal_ai_generation_log
            WHERE id = ?
            """,
            (int(cur.lastrowid),),
        )
        return row or {}

    def journal_ai_generation_log_list(self, token: str, *, window_hours: int = 168, limit: int = 400) -> list[dict[str, Any]]:
        me = self.auth_me(token)
        safe_window_hours = max(1, min(24 * 180, int(window_hours)))
        safe_limit = max(1, min(2000, int(limit)))
        since_expr = f"-{safe_window_hours} hour"
        return self.store.query_all(
            """
            SELECT l.id AS log_id, l.journal_id, l.status, l.provider, l.model, l.trace_id, l.error_code, l.error_message,
                   l.latency_ms, l.generated_at
            FROM journal_ai_generation_log l
            INNER JOIN investment_journal j ON j.id = l.journal_id
            WHERE j.user_id = ? AND j.tenant_id = ? AND l.generated_at >= datetime('now', ?)
            ORDER BY l.id DESC
            LIMIT ?
            """,
            (me["user_id"], me["tenant_id"], since_expr, safe_limit),
        )

    def journal_ai_coverage_counts(self, token: str) -> dict[str, int]:
        me = self.auth_me(token)
        total_row = self.store.query_one(
            "SELECT COUNT(1) AS total_journals FROM investment_journal WHERE user_id = ? AND tenant_id = ?",
            (me["user_id"], me["tenant_id"]),
        )
        ai_row = self.store.query_one(
            """
            SELECT COUNT(1) AS journals_with_ai
            FROM journal_ai_reflection ar
            INNER JOIN investment_journal j ON j.id = ar.journal_id
            WHERE j.user_id = ? AND j.tenant_id = ?
            """,
            (me["user_id"], me["tenant_id"]),
        )
        return {
            "total_journals": int((total_row or {}).get("total_journals", 0) or 0),
            "journals_with_ai": int((ai_row or {}).get("journals_with_ai", 0) or 0),
        }

    # ----------------- Reports -----------------
    def save_report_index(
        self,
        *,
        report_id: str,
        user_id: int | None,
        tenant_id: int | None,
        stock_code: str,
        report_type: str,
        markdown: str,
        run_id: str = "",
        pool_snapshot_id: str = "",
        template_id: str = "",
    ) -> None:
        self.store.execute(
            """
            INSERT OR REPLACE INTO report_index (report_id, user_id, tenant_id, stock_code, report_type, run_id, pool_snapshot_id, template_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (report_id, user_id, tenant_id, stock_code, report_type, run_id, pool_snapshot_id, template_id),
        )
        existing = self.store.query_one("SELECT MAX(version) AS v FROM report_version WHERE report_id = ?", (report_id,))
        next_version = int(existing["v"] or 0) + 1
        self.store.execute(
            "INSERT INTO report_version (report_id, version, markdown) VALUES (?, ?, ?)",
            (report_id, next_version, markdown),
        )

    def report_list(self, token: str) -> list[dict[str, Any]]:
        me = self.auth_me(token)
        return self.store.query_all(
            """
            SELECT report_id, stock_code, report_type, run_id, pool_snapshot_id, template_id, created_at
            FROM report_index
            WHERE user_id = ? AND tenant_id = ?
            ORDER BY created_at DESC
            """,
            (me["user_id"], me["tenant_id"]),
        )

    def report_versions(self, token: str, report_id: str) -> list[dict[str, Any]]:
        _ = self.auth_me(token)
        return self.store.query_all(
            "SELECT version, created_at FROM report_version WHERE report_id = ? ORDER BY version DESC",
            (report_id,),
        )

    def report_export(self, token: str, report_id: str) -> dict[str, Any]:
        _ = self.auth_me(token)
        latest = self.store.query_one(
            "SELECT markdown, version FROM report_version WHERE report_id = ? ORDER BY version DESC LIMIT 1",
            (report_id,),
        )
        if not latest:
            return {"error": "not_found"}
        return {"report_id": report_id, "version": latest["version"], "markdown": latest["markdown"]}

    # ----------------- Docs center -----------------
    def doc_upsert(self, *, doc_id: str, filename: str, parse_confidence: float, needs_review: bool, user_id: int | None = None, tenant_id: int | None = None) -> None:
        self.store.execute(
            """
            INSERT OR REPLACE INTO doc_index (doc_id, user_id, tenant_id, filename, status, parse_confidence, needs_review)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (doc_id, user_id, tenant_id, filename, "indexed", parse_confidence, int(needs_review)),
        )

    def docs_list(self, token: str) -> list[dict[str, Any]]:
        me = self.auth_me(token)
        return self.store.query_all(
            """
            SELECT doc_id, filename, status, parse_confidence, needs_review, created_at
            FROM doc_index
            WHERE tenant_id = ? OR tenant_id IS NULL
            ORDER BY created_at DESC
            """,
            (me["tenant_id"],),
        )

    def docs_review_queue(self, token: str) -> list[dict[str, Any]]:
        me = self.auth_me(token)
        return self.store.query_all(
            """
            SELECT doc_id, filename, parse_confidence, created_at
            FROM doc_index
            WHERE (tenant_id = ? OR tenant_id IS NULL) AND needs_review = 1
            ORDER BY created_at DESC
            """,
            (me["tenant_id"],),
        )

    def docs_review_action(self, token: str, doc_id: str, action: str, comment: str = "") -> dict[str, Any]:
        me = self.require_role(token, {"admin", "ops"})
        if action not in {"approve", "reject"}:
            raise ValueError("invalid action")
        needs_review = 0 if action == "approve" else 1
        self.store.execute("UPDATE doc_index SET needs_review = ?, status = ? WHERE doc_id = ?", (needs_review, action, doc_id))
        self.store.execute(
            "INSERT INTO doc_review_log (doc_id, action, reviewer_user_id, comment) VALUES (?, ?, ?, ?)",
            (doc_id, action, me["user_id"], comment),
        )
        return {"status": "ok", "doc_id": doc_id, "action": action}

    def doc_pipeline_run_add(
        self,
        *,
        doc_id: str,
        stage: str,
        status: str,
        filename: str = "",
        parse_confidence: float = 0.0,
        chunk_count: int = 0,
        table_count: int = 0,
        parse_notes: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append one doc pipeline run event for observability and version tracking."""
        self.store.execute(
            """
            INSERT INTO doc_pipeline_run
            (doc_id, stage, status, filename, parse_confidence, chunk_count, table_count, parse_notes, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(doc_id or "").strip(),
                str(stage or "").strip()[:40],
                str(status or "").strip()[:24],
                str(filename or "").strip()[:260],
                max(0.0, min(1.0, float(parse_confidence or 0.0))),
                max(0, int(chunk_count or 0)),
                max(0, int(table_count or 0)),
                str(parse_notes or "").strip()[:500],
                json.dumps(metadata or {}, ensure_ascii=False),
            ),
        )

    def doc_pipeline_runs(self, token: str, doc_id: str, *, limit: int = 30) -> list[dict[str, Any]]:
        """List latest document pipeline runs for one doc."""
        _ = self.auth_me(token)
        safe_doc = str(doc_id or "").strip()
        if not safe_doc:
            raise ValueError("doc_id is required")
        safe_limit = max(1, min(200, int(limit)))
        rows = self.store.query_all(
            """
            SELECT id, doc_id, stage, status, filename, parse_confidence, chunk_count, table_count, parse_notes, metadata_json, created_at
            FROM doc_pipeline_run
            WHERE doc_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (safe_doc, safe_limit),
        )
        for row in rows:
            row["metadata"] = self._json_loads_or(row.get("metadata_json"), {})
            row.pop("metadata_json", None)
        return rows

    def doc_versions(self, token: str, doc_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Build version list from successful index runs (version = sequence of successful index)."""
        _ = self.auth_me(token)
        safe_doc = str(doc_id or "").strip()
        if not safe_doc:
            raise ValueError("doc_id is required")
        safe_limit = max(1, min(100, int(limit)))
        rows = self.store.query_all(
            """
            SELECT id, doc_id, filename, parse_confidence, chunk_count, table_count, parse_notes, metadata_json, created_at
            FROM doc_pipeline_run
            WHERE doc_id = ? AND stage = 'index' AND status = 'ok'
            ORDER BY id ASC
            LIMIT ?
            """,
            (safe_doc, safe_limit),
        )
        versions: list[dict[str, Any]] = []
        for idx, row in enumerate(rows, start=1):
            versions.append(
                {
                    "version": idx,
                    "doc_id": row.get("doc_id", ""),
                    "filename": row.get("filename", ""),
                    "parse_confidence": float(row.get("parse_confidence", 0.0) or 0.0),
                    "chunk_count": int(row.get("chunk_count", 0) or 0),
                    "table_count": int(row.get("table_count", 0) or 0),
                    "parse_notes": row.get("parse_notes", ""),
                    "metadata": self._json_loads_or(row.get("metadata_json"), {}),
                    "created_at": row.get("created_at", ""),
                }
            )
        versions.reverse()
        return versions

    # ----------------- RAG Assets -----------------
    def rag_source_policy_list(self, token: str) -> list[dict[str, Any]]:
        _ = self.require_role(token, {"admin", "ops"})
        rows = self.store.query_all(
            """
            SELECT source, auto_approve, trust_score, enabled, updated_at
            FROM rag_doc_source_policy
            ORDER BY source ASC
            """
        )
        for row in rows:
            row["auto_approve"] = bool(int(row.get("auto_approve", 0)))
            row["enabled"] = bool(int(row.get("enabled", 0)))
        return rows

    def rag_source_policy_upsert(
        self,
        token: str,
        *,
        source: str,
        auto_approve: bool,
        trust_score: float,
        enabled: bool,
    ) -> dict[str, Any]:
        _ = self.require_role(token, {"admin", "ops"})
        normalized = str(source or "").strip().lower()
        if not normalized:
            raise ValueError("source is required")
        self.store.execute(
            """
            INSERT OR REPLACE INTO rag_doc_source_policy (source, auto_approve, trust_score, enabled, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                normalized,
                int(bool(auto_approve)),
                float(max(0.0, min(1.0, trust_score))),
                int(bool(enabled)),
            ),
        )
        return self.rag_source_policy_get(normalized)

    def rag_source_policy_get(self, source: str) -> dict[str, Any]:
        row = self.store.query_one(
            """
            SELECT source, auto_approve, trust_score, enabled, updated_at
            FROM rag_doc_source_policy
            WHERE source = ?
            """,
            (str(source or "").strip().lower(),),
        )
        if not row:
            return {}
        row["auto_approve"] = bool(int(row.get("auto_approve", 0)))
        row["enabled"] = bool(int(row.get("enabled", 0)))
        return row

    def rag_doc_chunk_replace(
        self,
        *,
        doc_id: str,
        source: str,
        source_url: str,
        effective_status: str,
        quality_score: float,
        chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self.store.execute("DELETE FROM rag_doc_chunk WHERE doc_id = ?", (doc_id,))
        for chunk in chunks:
            self.store.execute(
                """
                INSERT INTO rag_doc_chunk
                (chunk_id, doc_id, chunk_no, chunk_text, chunk_text_redacted, source, source_url,
                 stock_codes_json, industry_tags_json, effective_status, quality_score, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    str(chunk.get("chunk_id", "")),
                    str(doc_id),
                    int(chunk.get("chunk_no", 0)),
                    str(chunk.get("chunk_text", "")),
                    str(chunk.get("chunk_text_redacted", "")),
                    str(source),
                    str(source_url or ""),
                    json.dumps(chunk.get("stock_codes", []), ensure_ascii=False),
                    json.dumps(chunk.get("industry_tags", []), ensure_ascii=False),
                    str(effective_status),
                    float(max(0.0, min(1.0, quality_score))),
                ),
            )
        return {"doc_id": doc_id, "chunk_count": len(chunks), "status": "ok"}

    def rag_doc_chunk_set_status(self, token: str, *, chunk_id: str, status: str) -> dict[str, Any]:
        _ = self.require_role(token, {"admin", "ops"})
        normalized = str(status).strip().lower()
        if normalized not in {"active", "review", "rejected", "archived"}:
            raise ValueError("invalid status")
        self.store.execute(
            "UPDATE rag_doc_chunk SET effective_status = ?, updated_at = CURRENT_TIMESTAMP WHERE chunk_id = ?",
            (normalized, chunk_id),
        )
        row = self.store.query_one(
            """
            SELECT chunk_id, doc_id, chunk_no, source, effective_status, quality_score, updated_at
            FROM rag_doc_chunk
            WHERE chunk_id = ?
            """,
            (chunk_id,),
        )
        return row or {"error": "not_found", "chunk_id": chunk_id}

    def rag_doc_chunk_set_status_by_doc(self, *, doc_id: str, status: str) -> None:
        normalized = str(status).strip().lower()
        if normalized not in {"active", "review", "rejected", "archived"}:
            raise ValueError("invalid status")
        self.store.execute(
            "UPDATE rag_doc_chunk SET effective_status = ?, updated_at = CURRENT_TIMESTAMP WHERE doc_id = ?",
            (normalized, doc_id),
        )

    def rag_doc_chunk_list(
        self,
        token: str,
        *,
        doc_id: str = "",
        status: str = "",
        source: str = "",
        stock_code: str = "",
        limit: int = 60,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        _ = self.require_role(token, {"admin", "ops"})
        safe_limit = max(1, min(500, int(limit)))
        safe_offset = max(0, int(offset))
        cond = ["1=1"]
        params: list[Any] = []
        if doc_id.strip():
            cond.append("doc_id = ?")
            params.append(doc_id.strip())
        if status.strip():
            cond.append("effective_status = ?")
            params.append(status.strip().lower())
        if source.strip():
            cond.append("source = ?")
            params.append(source.strip().lower())
        sql = f"""
            SELECT chunk_id, doc_id, chunk_no, source, source_url, effective_status, quality_score,
                   stock_codes_json, industry_tags_json, updated_at
            FROM rag_doc_chunk
            WHERE {' AND '.join(cond)}
            ORDER BY updated_at DESC, chunk_no ASC
            LIMIT ? OFFSET ?
            """
        params.extend([safe_limit, safe_offset])
        rows = self.store.query_all(sql, tuple(params))
        filtered: list[dict[str, Any]] = []
        for row in rows:
            row["stock_codes"] = self._json_loads_or(row.get("stock_codes_json"), [])
            row["industry_tags"] = self._json_loads_or(row.get("industry_tags_json"), [])
            row.pop("stock_codes_json", None)
            row.pop("industry_tags_json", None)
            if stock_code.strip():
                target = stock_code.strip().upper()
                if target not in {str(x).upper() for x in row.get("stock_codes", [])}:
                    continue
            filtered.append(row)
        return filtered

    def rag_doc_chunk_list_internal(
        self,
        *,
        status: str = "",
        stock_code: str = "",
        limit: int = 800,
    ) -> list[dict[str, Any]]:
        """内部查询：给服务层构建检索语料使用，不走权限检查。"""
        safe_limit = max(1, min(5000, int(limit)))
        cond = ["1=1"]
        params: list[Any] = []
        if status.strip():
            cond.append("effective_status = ?")
            params.append(status.strip().lower())
        sql = f"""
            SELECT chunk_id, doc_id, chunk_no, chunk_text, chunk_text_redacted, source, source_url,
                   stock_codes_json, industry_tags_json, quality_score, updated_at
            FROM rag_doc_chunk
            WHERE {' AND '.join(cond)}
            ORDER BY updated_at DESC, chunk_no ASC
            LIMIT ?
            """
        params.append(safe_limit)
        rows = self.store.query_all(sql, tuple(params))
        target = stock_code.strip().upper()
        result: list[dict[str, Any]] = []
        for row in rows:
            row["stock_codes"] = self._json_loads_or(row.get("stock_codes_json"), [])
            row["industry_tags"] = self._json_loads_or(row.get("industry_tags_json"), [])
            row.pop("stock_codes_json", None)
            row.pop("industry_tags_json", None)
            if target:
                if target not in {str(x).upper() for x in row.get("stock_codes", [])}:
                    continue
            result.append(row)
        return result

    def rag_qa_memory_add(
        self,
        *,
        memory_id: str,
        user_id: str,
        stock_code: str,
        query_text: str,
        answer_text: str,
        answer_redacted: str,
        summary_text: str,
        citations: list[dict[str, Any]],
        risk_flags: list[str],
        intent: str,
        quality_score: float,
        share_scope: str,
        retrieval_enabled: bool,
    ) -> None:
        self.store.execute(
            """
            INSERT OR REPLACE INTO rag_qa_memory
            (memory_id, user_id, stock_code, query_text, answer_text, answer_redacted, summary_text,
             citations_json, risk_flags_json, intent, quality_score, share_scope, retrieval_enabled, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                memory_id,
                user_id,
                stock_code,
                query_text,
                answer_text,
                answer_redacted,
                summary_text,
                json.dumps(citations, ensure_ascii=False),
                json.dumps(risk_flags, ensure_ascii=False),
                intent,
                float(max(0.0, min(1.0, quality_score))),
                share_scope,
                int(bool(retrieval_enabled)),
            ),
        )

    def rag_qa_memory_list(
        self,
        token: str,
        *,
        stock_code: str = "",
        retrieval_enabled: int = -1,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        _ = self.require_role(token, {"admin", "ops"})
        return self.rag_qa_memory_list_internal(
            stock_code=stock_code,
            retrieval_enabled=retrieval_enabled,
            limit=limit,
            offset=offset,
        )

    def rag_qa_memory_list_internal(
        self,
        *,
        stock_code: str = "",
        retrieval_enabled: int = -1,
        limit: int = 300,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(2000, int(limit)))
        safe_offset = max(0, int(offset))
        cond = ["1=1"]
        params: list[Any] = []
        if stock_code.strip():
            cond.append("stock_code = ?")
            params.append(stock_code.strip().upper())
        if retrieval_enabled in (0, 1):
            cond.append("retrieval_enabled = ?")
            params.append(int(retrieval_enabled))
        sql = f"""
            SELECT memory_id, user_id, stock_code, query_text, answer_text, answer_redacted, summary_text,
                   citations_json, risk_flags_json, intent, quality_score, share_scope, retrieval_enabled, created_at
            FROM rag_qa_memory
            WHERE {' AND '.join(cond)}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """
        params.extend([safe_limit, safe_offset])
        rows = self.store.query_all(sql, tuple(params))
        for row in rows:
            row["citations"] = self._json_loads_or(row.get("citations_json"), [])
            row["risk_flags"] = self._json_loads_or(row.get("risk_flags_json"), [])
            row["retrieval_enabled"] = bool(int(row.get("retrieval_enabled", 0)))
            row.pop("citations_json", None)
            row.pop("risk_flags_json", None)
        return rows

    def rag_qa_memory_toggle(self, token: str, *, memory_id: str, retrieval_enabled: bool) -> dict[str, Any]:
        _ = self.require_role(token, {"admin", "ops"})
        self.store.execute(
            "UPDATE rag_qa_memory SET retrieval_enabled = ?, updated_at = CURRENT_TIMESTAMP WHERE memory_id = ?",
            (int(bool(retrieval_enabled)), memory_id),
        )
        row = self.store.query_one(
            """
            SELECT memory_id, stock_code, retrieval_enabled, quality_score, updated_at
            FROM rag_qa_memory
            WHERE memory_id = ?
            """,
            (memory_id,),
        )
        if not row:
            return {"error": "not_found", "memory_id": memory_id}
        row["retrieval_enabled"] = bool(int(row.get("retrieval_enabled", 0)))
        return row

    def rag_qa_feedback_add(self, *, memory_id: str, signal: str) -> None:
        self.store.execute(
            "INSERT INTO rag_qa_feedback (memory_id, signal) VALUES (?, ?)",
            (memory_id, str(signal).strip().lower()),
        )

    def rag_retrieval_trace_add(
        self,
        *,
        trace_id: str,
        query_text: str,
        query_type: str,
        retrieved_ids: list[str],
        selected_ids: list[str],
        latency_ms: int,
    ) -> None:
        self.store.execute(
            """
            INSERT INTO rag_retrieval_trace
            (trace_id, query_text, query_type, retrieved_ids_json, selected_ids_json, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                trace_id,
                query_text,
                query_type,
                json.dumps(retrieved_ids, ensure_ascii=False),
                json.dumps(selected_ids, ensure_ascii=False),
                int(max(0, latency_ms)),
            ),
        )

    def rag_retrieval_trace_list(self, token: str, *, trace_id: str = "", limit: int = 120) -> list[dict[str, Any]]:
        _ = self.require_role(token, {"admin", "ops"})
        safe_limit = max(1, min(2000, int(limit)))
        cond = ["1=1"]
        params: list[Any] = []
        if trace_id.strip():
            cond.append("trace_id = ?")
            params.append(trace_id.strip())
        sql = f"""
            SELECT id, trace_id, query_text, query_type, retrieved_ids_json, selected_ids_json, latency_ms, created_at
            FROM rag_retrieval_trace
            WHERE {' AND '.join(cond)}
            ORDER BY id DESC
            LIMIT ?
            """
        params.append(safe_limit)
        rows = self.store.query_all(sql, tuple(params))
        for row in rows:
            row["retrieved_ids"] = self._json_loads_or(row.get("retrieved_ids_json"), [])
            row["selected_ids"] = self._json_loads_or(row.get("selected_ids_json"), [])
            row.pop("retrieved_ids_json", None)
            row.pop("selected_ids_json", None)
        return rows

    def analysis_intel_feedback_add(
        self,
        *,
        stock_code: str,
        trace_id: str,
        signal: str,
        confidence: float,
        position_hint: str,
        feedback: str,
        baseline_trade_date: str,
        baseline_price: float,
    ) -> dict[str, Any]:
        self.store.execute(
            """
            INSERT INTO analysis_intel_feedback
            (stock_code, trace_id, signal, confidence, position_hint, feedback, baseline_trade_date, baseline_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(stock_code).strip().upper(),
                str(trace_id).strip(),
                str(signal).strip().lower() or "hold",
                float(confidence or 0.0),
                str(position_hint or "").strip(),
                str(feedback).strip().lower() or "watch",
                str(baseline_trade_date).strip(),
                float(baseline_price or 0.0),
            ),
        )
        row = self.store.query_one(
            """
            SELECT id, stock_code, trace_id, signal, confidence, position_hint, feedback,
                   baseline_trade_date, baseline_price, created_at
            FROM analysis_intel_feedback
            ORDER BY id DESC
            LIMIT 1
            """
        )
        return row or {}

    def analysis_intel_feedback_list(self, *, stock_code: str = "", limit: int = 100) -> list[dict[str, Any]]:
        safe_limit = max(1, min(2000, int(limit)))
        cond = ["1=1"]
        params: list[Any] = []
        if str(stock_code).strip():
            cond.append("stock_code = ?")
            params.append(str(stock_code).strip().upper())
        sql = f"""
            SELECT id, stock_code, trace_id, signal, confidence, position_hint, feedback,
                   baseline_trade_date, baseline_price, created_at
            FROM analysis_intel_feedback
            WHERE {' AND '.join(cond)}
            ORDER BY id DESC
            LIMIT ?
            """
        params.append(safe_limit)
        return self.store.query_all(sql, tuple(params))

    def rag_upload_asset_get_by_hash(self, file_sha256: str) -> dict[str, Any] | None:
        row = self.store.query_one(
            """
            SELECT upload_id, doc_id, filename, source, source_url, file_sha256, file_size, content_type,
                   stock_codes_json, tags_json, parse_note, status, created_by, created_at, updated_at
            FROM rag_upload_asset
            WHERE file_sha256 = ?
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (str(file_sha256).strip().lower(),),
        )
        if not row:
            return None
        row["stock_codes"] = self._json_loads_or(row.get("stock_codes_json"), [])
        row["tags"] = self._json_loads_or(row.get("tags_json"), [])
        row.pop("stock_codes_json", None)
        row.pop("tags_json", None)
        return row

    def rag_upload_asset_upsert(
        self,
        token: str,
        *,
        upload_id: str,
        doc_id: str,
        filename: str,
        source: str,
        source_url: str,
        file_sha256: str,
        file_size: int,
        content_type: str,
        stock_codes: list[str],
        tags: list[str],
        parse_note: str,
        status: str,
        created_by: str,
    ) -> dict[str, Any]:
        _ = self.require_role(token, {"admin", "ops"})
        self.store.execute(
            """
            INSERT OR REPLACE INTO rag_upload_asset
            (upload_id, doc_id, filename, source, source_url, file_sha256, file_size, content_type,
             stock_codes_json, tags_json, parse_note, status, created_by, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM rag_upload_asset WHERE upload_id = ?), CURRENT_TIMESTAMP), CURRENT_TIMESTAMP)
            """,
            (
                str(upload_id),
                str(doc_id),
                str(filename),
                str(source),
                str(source_url or ""),
                str(file_sha256).strip().lower(),
                int(max(0, file_size)),
                str(content_type or ""),
                json.dumps(stock_codes, ensure_ascii=False),
                json.dumps(tags, ensure_ascii=False),
                str(parse_note or ""),
                str(status or "uploaded"),
                str(created_by or ""),
                str(upload_id),
            ),
        )
        row = self.store.query_one(
            """
            SELECT upload_id, doc_id, filename, source, source_url, file_sha256, file_size, content_type,
                   stock_codes_json, tags_json, parse_note, status, created_by, created_at, updated_at
            FROM rag_upload_asset
            WHERE upload_id = ?
            """,
            (str(upload_id),),
        )
        if not row:
            return {"error": "not_found", "upload_id": upload_id}
        row["stock_codes"] = self._json_loads_or(row.get("stock_codes_json"), [])
        row["tags"] = self._json_loads_or(row.get("tags_json"), [])
        row.pop("stock_codes_json", None)
        row.pop("tags_json", None)
        return row

    def rag_upload_asset_list(
        self,
        token: str,
        *,
        status: str = "",
        source: str = "",
        limit: int = 40,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        _ = self.require_role(token, {"admin", "ops"})
        safe_limit = max(1, min(300, int(limit)))
        safe_offset = max(0, int(offset))
        cond = ["1=1"]
        params: list[Any] = []
        if status.strip():
            cond.append("status = ?")
            params.append(status.strip().lower())
        if source.strip():
            cond.append("source = ?")
            params.append(source.strip().lower())
        sql = f"""
            SELECT upload_id, doc_id, filename, source, source_url, file_sha256, file_size, content_type,
                   stock_codes_json, tags_json, parse_note, status, created_by, created_at, updated_at
            FROM rag_upload_asset
            WHERE {' AND '.join(cond)}
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
            """
        params.extend([safe_limit, safe_offset])
        rows = self.store.query_all(sql, tuple(params))
        for row in rows:
            row["stock_codes"] = self._json_loads_or(row.get("stock_codes_json"), [])
            row["tags"] = self._json_loads_or(row.get("tags_json"), [])
            row.pop("stock_codes_json", None)
            row.pop("tags_json", None)
        return rows

    def rag_upload_asset_set_status(self, *, doc_id: str, status: str, parse_note: str = "") -> None:
        self.store.execute(
            """
            UPDATE rag_upload_asset
            SET status = ?, parse_note = ?, updated_at = CURRENT_TIMESTAMP
            WHERE doc_id = ?
            """,
            (str(status or "uploaded"), str(parse_note or ""), str(doc_id)),
        )

    def rag_ops_meta_set(self, *, key: str, value: str) -> None:
        self.store.execute(
            """
            INSERT OR REPLACE INTO rag_ops_meta (meta_key, meta_value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
            (str(key), str(value)),
        )

    def rag_ops_meta_get(self, *, key: str) -> dict[str, Any]:
        row = self.store.query_one(
            """
            SELECT meta_key, meta_value, updated_at
            FROM rag_ops_meta
            WHERE meta_key = ?
            """,
            (str(key),),
        )
        return row or {}

    def rag_dashboard_summary(self, token: str) -> dict[str, Any]:
        _ = self.require_role(token, {"admin", "ops"})
        doc_total = int(self.store.query_one("SELECT COUNT(1) AS cnt FROM doc_index", ())["cnt"])
        active_chunks = int(
            self.store.query_one(
                "SELECT COUNT(1) AS cnt FROM rag_doc_chunk WHERE effective_status = 'active'",
                (),
            )["cnt"]
        )
        review_pending = int(
            self.store.query_one(
                "SELECT COUNT(1) AS cnt FROM doc_index WHERE needs_review = 1",
                (),
            )["cnt"]
        )
        qa_memory_total = int(self.store.query_one("SELECT COUNT(1) AS cnt FROM rag_qa_memory", ())["cnt"])
        trace_total = int(
            self.store.query_one(
                "SELECT COUNT(1) AS cnt FROM rag_retrieval_trace WHERE created_at >= datetime('now', '-7 day')",
                (),
            )["cnt"]
        )
        trace_hit = int(
            self.store.query_one(
                """
                SELECT COUNT(1) AS cnt
                FROM rag_retrieval_trace
                WHERE created_at >= datetime('now', '-7 day')
                  AND selected_ids_json IS NOT NULL
                  AND selected_ids_json != '[]'
                """,
                (),
            )["cnt"]
        )
        hit_rate = round((trace_hit / trace_total), 4) if trace_total > 0 else 0.0
        latest_reindex = self.rag_ops_meta_get(key="last_reindex_at")
        return {
            "doc_total": doc_total,
            "active_chunks": active_chunks,
            "review_pending": review_pending,
            "qa_memory_total": qa_memory_total,
            "retrieval_hit_rate_7d": hit_rate,
            "retrieval_trace_count_7d": trace_total,
            "last_reindex_at": str(latest_reindex.get("meta_value", "")),
        }

    # ----------------- Ops / Health / Alerts -----------------
    def source_health_upsert(self, source_id: str, success_rate: float, circuit_open: bool, last_error: str = "") -> None:
        self.store.execute(
            """
            INSERT OR REPLACE INTO source_health (source_id, success_rate, circuit_open, last_error, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (source_id, success_rate, int(circuit_open), last_error),
        )

    def source_health_list(self, token: str) -> list[dict[str, Any]]:
        _ = self.require_role(token, {"admin", "ops"})
        return self.store.query_all(
            "SELECT source_id, success_rate, circuit_open, last_error, updated_at FROM source_health ORDER BY source_id",
            (),
        )

    def create_alert(self, alert_type: str, severity: str, message: str) -> int:
        cur = self.store.execute(
            "INSERT INTO alert_event (alert_type, severity, message, status) VALUES (?, ?, ?, 'open')",
            (alert_type, severity, message),
        )
        return int(cur.lastrowid)

    def alerts_list(self, token: str) -> list[dict[str, Any]]:
        _ = self.require_role(token, {"admin", "ops"})
        return self.store.query_all(
            "SELECT id, alert_type, severity, message, status, created_at FROM alert_event ORDER BY id DESC",
            (),
        )

    def alerts_ack(self, token: str, alert_id: int) -> dict[str, Any]:
        me = self.require_role(token, {"admin", "ops"})
        self.store.execute("UPDATE alert_event SET status='acked' WHERE id = ?", (alert_id,))
        self.store.execute("INSERT INTO alert_ack (alert_id, user_id) VALUES (?, ?)", (alert_id, me["user_id"]))
        return {"status": "ok", "alert_id": alert_id}

    def alert_rule_create(self, token: str, payload: dict[str, Any]) -> dict[str, Any]:
        me = self.auth_me(token)
        rule_name = str(payload.get("rule_name", "")).strip()
        rule_type = str(payload.get("rule_type", "")).strip().lower()
        stock_code = str(payload.get("stock_code", "")).strip().upper()
        operator = str(payload.get("operator", "")).strip()
        target_value = float(payload.get("target_value", 0.0) or 0.0)
        event_type = str(payload.get("event_type", "")).strip().lower()
        if not rule_name:
            raise ValueError("rule_name is required")
        if rule_type not in {"price", "event"}:
            raise ValueError("rule_type must be price or event")
        if not stock_code:
            raise ValueError("stock_code is required")
        if rule_type == "price":
            if operator not in {">", "<", ">=", "<="}:
                raise ValueError("operator must be one of >, <, >=, <=")
        cur = self.store.execute(
            """
            INSERT INTO alert_rule
            (user_id, tenant_id, rule_name, rule_type, stock_code, operator, target_value, event_type, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                me["user_id"],
                me["tenant_id"],
                rule_name,
                rule_type,
                stock_code,
                operator,
                target_value,
                event_type,
                int(bool(payload.get("is_active", True))),
            ),
        )
        return {"status": "ok", "rule_id": int(cur.lastrowid)}

    def alert_rule_list(self, token: str) -> list[dict[str, Any]]:
        me = self.auth_me(token)
        rows = self.store.query_all(
            """
            SELECT id AS rule_id, rule_name, rule_type, stock_code, operator, target_value, event_type, is_active, created_at, updated_at
            FROM alert_rule
            WHERE user_id = ? AND tenant_id = ?
            ORDER BY updated_at DESC, id DESC
            """,
            (me["user_id"], me["tenant_id"]),
        )
        for row in rows:
            row["is_active"] = bool(int(row.get("is_active", 0) or 0))
        return rows

    def alert_rule_delete(self, token: str, rule_id: int) -> dict[str, Any]:
        me = self.auth_me(token)
        self.store.execute(
            "DELETE FROM alert_rule WHERE id = ? AND user_id = ? AND tenant_id = ?",
            (int(rule_id), me["user_id"], me["tenant_id"]),
        )
        return {"status": "ok", "rule_id": int(rule_id)}

    def alert_trigger_log_add(self, *, rule_id: int, stock_code: str, trigger_message: str, trigger_data: dict[str, Any]) -> int:
        cur = self.store.execute(
            """
            INSERT INTO alert_trigger_log (rule_id, stock_code, trigger_message, trigger_data_json)
            VALUES (?, ?, ?, ?)
            """,
            (int(rule_id), str(stock_code).upper(), str(trigger_message), json.dumps(trigger_data, ensure_ascii=False)),
        )
        return int(cur.lastrowid)

    def alert_trigger_log_list(self, token: str, *, limit: int = 100) -> list[dict[str, Any]]:
        me = self.auth_me(token)
        safe_limit = max(1, min(1000, int(limit)))
        rows = self.store.query_all(
            """
            SELECT l.id AS alert_id, l.rule_id, l.stock_code, l.trigger_message, l.trigger_data_json, l.triggered_at
            FROM alert_trigger_log l
            JOIN alert_rule r ON r.id = l.rule_id
            WHERE r.user_id = ? AND r.tenant_id = ?
            ORDER BY l.id DESC
            LIMIT ?
            """,
            (me["user_id"], me["tenant_id"], safe_limit),
        )
        for row in rows:
            row["trigger_data"] = self._json_loads_or(row.get("trigger_data_json"), {})
            row.pop("trigger_data_json", None)
        return rows

    def rag_eval_add(self, *, query_text: str, positive_source_ids: list[str], predicted_source_ids: list[str]) -> None:
        self.store.execute(
            "INSERT INTO rag_eval_case (query_text, positive_source_ids, predicted_source_ids) VALUES (?, ?, ?)",
            (
                query_text,
                json.dumps(positive_source_ids, ensure_ascii=False),
                json.dumps(predicted_source_ids, ensure_ascii=False),
            ),
        )

    def rag_eval_recent(self, limit: int = 200) -> list[dict[str, Any]]:
        rows = self.store.query_all(
            """
            SELECT id, query_text, positive_source_ids, predicted_source_ids, created_at
            FROM rag_eval_case
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        for row in rows:
            try:
                row["positive_source_ids"] = json.loads(row.get("positive_source_ids", "[]"))
            except Exception:
                row["positive_source_ids"] = []
            try:
                row["predicted_source_ids"] = json.loads(row.get("predicted_source_ids", "[]"))
            except Exception:
                row["predicted_source_ids"] = []
        return rows

    # ----------------- Deep Think / A2A -----------------
    def deep_think_create_session(
        self,
        *,
        session_id: str,
        user_id: str,
        question: str,
        stock_codes: list[str],
        agent_profile: list[str],
        max_rounds: int,
        budget: dict[str, Any],
        mode: str,
        trace_id: str,
    ) -> dict[str, Any]:
        self.store.execute(
            """
            INSERT INTO deep_think_session
            (session_id, user_id, question, stock_codes, agent_profile, max_rounds, current_round, budget_json, mode, status, trace_id)
            VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, 'created', ?)
            """,
            (
                session_id,
                user_id,
                question,
                json.dumps(stock_codes, ensure_ascii=False),
                json.dumps(agent_profile, ensure_ascii=False),
                max_rounds,
                json.dumps(budget, ensure_ascii=False),
                mode,
                trace_id,
            ),
        )
        return self.deep_think_get_session(session_id)

    def deep_think_get_session(self, session_id: str) -> dict[str, Any]:
        session = self.store.query_one(
            """
            SELECT session_id, user_id, question, stock_codes, agent_profile, max_rounds,
                   current_round, budget_json, mode, status, trace_id, created_at, updated_at
            FROM deep_think_session
            WHERE session_id = ?
            """,
            (session_id,),
        )
        if not session:
            return {}
        session["stock_codes"] = self._json_loads_or(session.get("stock_codes"), [])
        session["agent_profile"] = self._json_loads_or(session.get("agent_profile"), [])
        session["budget"] = self._json_loads_or(session.get("budget_json"), {})
        session.pop("budget_json", None)

        rounds = self.store.query_all(
            """
            SELECT round_id, session_id, round_no, status, consensus_signal, disagreement_score,
                   conflict_sources, counter_view, task_graph, replan_triggered, stop_reason, budget_usage, created_at
            FROM deep_think_round
            WHERE session_id = ?
            ORDER BY round_no ASC
            """,
            (session_id,),
        )
        for rnd in rounds:
            rnd["conflict_sources"] = self._json_loads_or(rnd.get("conflict_sources"), [])
            rnd["task_graph"] = self._json_loads_or(rnd.get("task_graph"), [])
            rnd["replan_triggered"] = bool(int(rnd.get("replan_triggered", 0)))
            rnd["budget_usage"] = self._json_loads_or(rnd.get("budget_usage"), {})
            opinions = self.store.query_all(
                """
                SELECT agent_id, signal, confidence, reason, evidence_ids, risk_tags, created_at
                FROM deep_think_opinion
                WHERE round_id = ?
                ORDER BY id ASC
                """,
                (rnd["round_id"],),
            )
            for opinion in opinions:
                opinion["evidence_ids"] = self._json_loads_or(opinion.get("evidence_ids"), [])
                opinion["risk_tags"] = self._json_loads_or(opinion.get("risk_tags"), [])
            rnd["opinions"] = opinions
        session["rounds"] = rounds
        return session

    def deep_think_append_round(
        self,
        *,
        session_id: str,
        round_id: str,
        round_no: int,
        status: str,
        consensus_signal: str,
        disagreement_score: float,
        conflict_sources: list[str],
        counter_view: str,
        task_graph: list[dict[str, Any]],
        replan_triggered: bool,
        stop_reason: str,
        budget_usage: dict[str, Any],
        opinions: list[dict[str, Any]],
        session_status: str,
    ) -> dict[str, Any]:
        self.store.execute(
            """
            INSERT INTO deep_think_round
            (round_id, session_id, round_no, status, consensus_signal, disagreement_score, conflict_sources, counter_view, task_graph, replan_triggered, stop_reason, budget_usage)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                round_id,
                session_id,
                round_no,
                status,
                consensus_signal,
                disagreement_score,
                json.dumps(conflict_sources, ensure_ascii=False),
                counter_view,
                json.dumps(task_graph, ensure_ascii=False),
                int(replan_triggered),
                stop_reason,
                json.dumps(budget_usage, ensure_ascii=False),
            ),
        )
        for opinion in opinions:
            self.store.execute(
                """
                INSERT INTO deep_think_opinion
                (round_id, agent_id, signal, confidence, reason, evidence_ids, risk_tags)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    round_id,
                    str(opinion.get("agent", "")),
                    str(opinion.get("signal", "hold")),
                    float(opinion.get("confidence", 0.5)),
                    str(opinion.get("reason", "")),
                    json.dumps(opinion.get("evidence_ids", []), ensure_ascii=False),
                    json.dumps(opinion.get("risk_tags", []), ensure_ascii=False),
                ),
            )

        self.store.execute(
            """
            UPDATE deep_think_session
            SET current_round = ?, status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE session_id = ?
            """,
            (round_no, session_status, session_id),
        )
        return self.deep_think_get_session(session_id)

    def deep_think_replace_round_events(
        self,
        *,
        session_id: str,
        round_id: str,
        round_no: int,
        events: list[dict[str, Any]],
        max_events: int = 1200,
    ) -> None:
        self.store.execute(
            """
            DELETE FROM deep_think_event
            WHERE session_id = ? AND round_id = ?
            """,
            (session_id, round_id),
        )
        for idx, item in enumerate(events, start=1):
            event_name = str(item.get("event", "message"))
            data = item.get("data", {})
            self.store.execute(
                """
                INSERT INTO deep_think_event
                (session_id, round_id, round_no, event_seq, event_name, data_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    round_id,
                    round_no,
                    idx,
                    event_name,
                    json.dumps(data, ensure_ascii=False),
                ),
            )
        self.deep_think_trim_events(session_id=session_id, max_events=max_events)

    def deep_think_trim_events(self, *, session_id: str, max_events: int = 1200) -> None:
        safe_max = max(1, min(5000, int(max_events)))
        self.store.execute(
            """
            DELETE FROM deep_think_event
            WHERE session_id = ?
            AND id NOT IN (
                SELECT id
                FROM deep_think_event
                WHERE session_id = ?
                ORDER BY round_no DESC, event_seq DESC, id DESC
                LIMIT ?
            )
            """,
            (session_id, session_id, safe_max),
        )

    def deep_think_list_events_page(
        self,
        *,
        session_id: str,
        round_id: str | None = None,
        limit: int = 200,
        event_name: str | None = None,
        cursor: int | None = None,
        created_from: str | None = None,
        created_to: str | None = None,
    ) -> dict[str, Any]:
        safe_limit = max(1, min(2000, int(limit)))
        safe_cursor = None
        if cursor is not None:
            try:
                safe_cursor = max(0, int(cursor))
            except Exception:  # noqa: BLE001
                safe_cursor = None
        conditions = ["session_id = ?"]
        params: list[Any] = [session_id]
        if round_id:
            conditions.append("round_id = ?")
            params.append(round_id)
        if event_name:
            conditions.append("event_name = ?")
            params.append(event_name)
        if created_from:
            conditions.append("created_at >= ?")
            params.append(created_from)
        if created_to:
            conditions.append("created_at <= ?")
            params.append(created_to)
        if safe_cursor is not None and safe_cursor > 0:
            conditions.append("id > ?")
            params.append(safe_cursor)
        params.append(safe_limit + 1)
        sql = f"""
            SELECT id, session_id, round_id, round_no, event_seq, event_name, data_json, created_at
            FROM deep_think_event
            WHERE {' AND '.join(conditions)}
            ORDER BY id ASC
            LIMIT ?
            """
        rows = self.store.query_all(sql, tuple(params))
        has_more = len(rows) > safe_limit
        page_rows = rows[:safe_limit]
        for row in page_rows:
            row["event_id"] = int(row.pop("id"))
            row["data"] = self._json_loads_or(row.get("data_json"), {})
            row.pop("data_json", None)
            row["event"] = row.pop("event_name", "message")
        next_cursor = int(page_rows[-1]["event_id"]) if has_more and page_rows else None
        return {"events": page_rows, "has_more": has_more, "next_cursor": next_cursor}

    def deep_think_list_events(
        self,
        *,
        session_id: str,
        round_id: str | None = None,
        limit: int = 200,
        event_name: str | None = None,
        cursor: int | None = None,
        created_from: str | None = None,
        created_to: str | None = None,
    ) -> list[dict[str, Any]]:
        page = self.deep_think_list_events_page(
            session_id=session_id,
            round_id=round_id,
            limit=limit,
            event_name=event_name,
            cursor=cursor,
            created_from=created_from,
            created_to=created_to,
        )
        return list(page.get("events", []))

    def deep_think_export_task_create(
        self,
        *,
        task_id: str,
        session_id: str,
        status: str,
        format: str,
        filters: dict[str, Any],
        max_attempts: int = 2,
    ) -> None:
        self.store.execute(
            """
            INSERT INTO deep_think_export_task
            (task_id, session_id, status, format, filters_json, attempt_count, max_attempts, updated_at)
            VALUES (?, ?, ?, ?, ?, 0, ?, CURRENT_TIMESTAMP)
            """,
            (
                task_id,
                session_id,
                status,
                format,
                json.dumps(filters, ensure_ascii=False),
                max(1, int(max_attempts)),
            ),
        )

    def deep_think_export_task_update(
        self,
        *,
        task_id: str,
        status: str,
        filename: str = "",
        media_type: str = "",
        content_text: str = "",
        row_count: int = 0,
        error: str = "",
    ) -> None:
        if status == "completed":
            self.store.execute(
                """
                UPDATE deep_think_export_task
                SET status = ?, filename = ?, media_type = ?, content_text = ?, row_count = ?, error = ?,
                    updated_at = CURRENT_TIMESTAMP, completed_at = CURRENT_TIMESTAMP
                WHERE task_id = ?
                """,
                (status, filename, media_type, content_text, row_count, error, task_id),
            )
            return
        self.store.execute(
            """
            UPDATE deep_think_export_task
            SET status = ?, filename = ?, media_type = ?, content_text = ?, row_count = ?, error = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE task_id = ?
            """,
            (status, filename, media_type, content_text, row_count, error, task_id),
        )

    def deep_think_export_task_try_claim(self, task_id: str, *, session_id: str | None = None) -> dict[str, Any]:
        if session_id:
            cur = self.store.execute(
                """
                UPDATE deep_think_export_task
                SET status = 'running', attempt_count = attempt_count + 1, updated_at = CURRENT_TIMESTAMP
                WHERE task_id = ? AND session_id = ? AND status = 'queued'
                """,
                (task_id, session_id),
            )
        else:
            cur = self.store.execute(
                """
                UPDATE deep_think_export_task
                SET status = 'running', attempt_count = attempt_count + 1, updated_at = CURRENT_TIMESTAMP
                WHERE task_id = ? AND status = 'queued'
                """,
                (task_id,),
            )
        if int(cur.rowcount or 0) <= 0:
            return {}
        return self.deep_think_export_task_get(task_id, session_id=session_id, include_content=False)

    def deep_think_export_task_requeue(self, *, task_id: str, error: str) -> None:
        self.store.execute(
            """
            UPDATE deep_think_export_task
            SET status = 'queued', error = ?, updated_at = CURRENT_TIMESTAMP
            WHERE task_id = ?
            """,
            (error, task_id),
        )

    def deep_think_export_task_get(
        self,
        task_id: str,
        *,
        session_id: str | None = None,
        include_content: bool = False,
    ) -> dict[str, Any]:
        row = self.store.query_one(
            """
            SELECT task_id, session_id, status, format, filters_json, filename, media_type, content_text,
                   row_count, error, attempt_count, max_attempts, created_at, updated_at, completed_at
            FROM deep_think_export_task
            WHERE task_id = ?
            """,
            (task_id,),
        )
        if not row:
            return {}
        if session_id and str(row.get("session_id", "")) != str(session_id):
            return {}
        row["filters"] = self._json_loads_or(row.get("filters_json"), {})
        row.pop("filters_json", None)
        if not include_content:
            row.pop("content_text", None)
        return row

    def deep_think_archive_audit_log(
        self,
        *,
        session_id: str,
        action: str,
        status: str,
        duration_ms: int,
        result_count: int = 0,
        export_bytes: int = 0,
        detail: dict[str, Any] | None = None,
    ) -> None:
        self.store.execute(
            """
            INSERT INTO deep_think_archive_audit
            (session_id, action, status, duration_ms, result_count, export_bytes, detail_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                action,
                status,
                max(0, int(duration_ms)),
                max(0, int(result_count)),
                max(0, int(export_bytes)),
                json.dumps(detail or {}, ensure_ascii=False),
            ),
        )

    def deep_think_archive_audit_metrics(self, *, window_hours: int = 24) -> dict[str, Any]:
        safe_hours = max(1, min(24 * 30, int(window_hours)))
        window_expr = f"-{safe_hours} hours"
        latency_rows = self.store.query_all(
            """
            SELECT duration_ms
            FROM deep_think_archive_audit
            WHERE created_at >= datetime('now', ?)
            ORDER BY duration_ms ASC
            """,
            (window_expr,),
        )
        latencies = [int(x.get("duration_ms", 0) or 0) for x in latency_rows]

        def _percentile(sorted_values: list[int], ratio: float) -> int:
            if not sorted_values:
                return 0
            clamped = min(1.0, max(0.0, float(ratio)))
            idx = int(round((len(sorted_values) - 1) * clamped))
            idx = max(0, min(len(sorted_values) - 1, idx))
            return int(sorted_values[idx])

        summary = self.store.query_one(
            """
            SELECT
                COUNT(1) AS total_calls,
                AVG(duration_ms) AS avg_latency_ms,
                MAX(duration_ms) AS max_latency_ms,
                SUM(CASE WHEN duration_ms >= 1000 THEN 1 ELSE 0 END) AS slow_calls_over_1000ms,
                SUM(result_count) AS total_result_count,
                SUM(export_bytes) AS total_export_bytes
            FROM deep_think_archive_audit
            WHERE created_at >= datetime('now', ?)
            """,
            (window_expr,),
        ) or {}
        by_action = self.store.query_all(
            """
            SELECT
                action,
                COUNT(1) AS call_count,
                AVG(duration_ms) AS avg_latency_ms,
                MAX(duration_ms) AS max_latency_ms,
                SUM(export_bytes) AS export_bytes
            FROM deep_think_archive_audit
            WHERE created_at >= datetime('now', ?)
            GROUP BY action
            ORDER BY action ASC
            """,
            (window_expr,),
        )
        by_action_status = self.store.query_all(
            """
            SELECT
                action,
                status,
                COUNT(1) AS call_count
            FROM deep_think_archive_audit
            WHERE created_at >= datetime('now', ?)
            GROUP BY action, status
            ORDER BY action ASC, status ASC
            """,
            (window_expr,),
        )
        top_sessions = self.store.query_all(
            """
            SELECT
                session_id,
                COUNT(1) AS call_count,
                AVG(duration_ms) AS avg_latency_ms,
                SUM(export_bytes) AS export_bytes
            FROM deep_think_archive_audit
            WHERE created_at >= datetime('now', ?)
            GROUP BY session_id
            ORDER BY call_count DESC, session_id ASC
            LIMIT 10
            """,
            (window_expr,),
        )
        by_status = self.store.query_all(
            """
            SELECT
                status,
                COUNT(1) AS call_count
            FROM deep_think_archive_audit
            WHERE created_at >= datetime('now', ?)
            GROUP BY status
            ORDER BY status ASC
            """,
            (window_expr,),
        )
        return {
            "window_hours": safe_hours,
            "total_calls": int(summary.get("total_calls", 0) or 0),
            "avg_latency_ms": round(float(summary.get("avg_latency_ms", 0) or 0.0), 2),
            "max_latency_ms": int(summary.get("max_latency_ms", 0) or 0),
            "p50_latency_ms": _percentile(latencies, 0.50),
            "p95_latency_ms": _percentile(latencies, 0.95),
            "p99_latency_ms": _percentile(latencies, 0.99),
            "slow_calls_over_1000ms": int(summary.get("slow_calls_over_1000ms", 0) or 0),
            "total_result_count": int(summary.get("total_result_count", 0) or 0),
            "total_export_bytes": int(summary.get("total_export_bytes", 0) or 0),
            "by_action": [
                {
                    "action": str(x.get("action", "")),
                    "call_count": int(x.get("call_count", 0) or 0),
                    "avg_latency_ms": round(float(x.get("avg_latency_ms", 0) or 0.0), 2),
                    "max_latency_ms": int(x.get("max_latency_ms", 0) or 0),
                    "export_bytes": int(x.get("export_bytes", 0) or 0),
                }
                for x in by_action
            ],
            "by_status": [
                {"status": str(x.get("status", "")), "call_count": int(x.get("call_count", 0) or 0)}
                for x in by_status
            ],
            "by_action_status": [
                {
                    "action": str(x.get("action", "")),
                    "status": str(x.get("status", "")),
                    "call_count": int(x.get("call_count", 0) or 0),
                }
                for x in by_action_status
            ],
            "top_sessions": [
                {
                    "session_id": str(x.get("session_id", "")),
                    "call_count": int(x.get("call_count", 0) or 0),
                    "avg_latency_ms": round(float(x.get("avg_latency_ms", 0) or 0.0), 2),
                    "export_bytes": int(x.get("export_bytes", 0) or 0),
                }
                for x in top_sessions
            ],
        }

    def register_agent_card(
        self,
        *,
        agent_id: str,
        display_name: str,
        description: str,
        capabilities: list[str],
        version: str = "1.0.0",
        status: str = "active",
    ) -> None:
        self.store.execute(
            """
            INSERT OR REPLACE INTO agent_card_registry
            (agent_id, display_name, description, capabilities, version, status, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                agent_id,
                display_name,
                description,
                json.dumps(capabilities, ensure_ascii=False),
                version,
                status,
            ),
        )

    def list_agent_cards(self) -> list[dict[str, Any]]:
        rows = self.store.query_all(
            """
            SELECT agent_id, display_name, description, capabilities, version, status, updated_at
            FROM agent_card_registry
            ORDER BY agent_id ASC
            """,
            (),
        )
        for row in rows:
            row["capabilities"] = self._json_loads_or(row.get("capabilities"), [])
        return rows

    def a2a_create_task(
        self,
        *,
        task_id: str,
        session_id: str | None,
        agent_id: str,
        status: str,
        payload: dict[str, Any],
        result: dict[str, Any] | None,
        trace_ref: str,
    ) -> None:
        self.store.execute(
            """
            INSERT INTO a2a_task
            (task_id, session_id, agent_id, status, payload_json, result_json, trace_ref)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                session_id,
                agent_id,
                status,
                json.dumps(payload, ensure_ascii=False),
                json.dumps(result or {}, ensure_ascii=False),
                trace_ref,
            ),
        )

    def a2a_update_task(self, *, task_id: str, status: str, result: dict[str, Any] | None) -> None:
        self.store.execute(
            """
            UPDATE a2a_task
            SET status = ?, result_json = ?, updated_at = CURRENT_TIMESTAMP
            WHERE task_id = ?
            """,
            (
                status,
                json.dumps(result or {}, ensure_ascii=False),
                task_id,
            ),
        )

    def a2a_get_task(self, task_id: str) -> dict[str, Any]:
        row = self.store.query_one(
            """
            SELECT task_id, session_id, agent_id, status, payload_json, result_json, trace_ref, created_at, updated_at
            FROM a2a_task
            WHERE task_id = ?
            """,
            (task_id,),
        )
        if not row:
            return {}
        row["payload"] = self._json_loads_or(row.get("payload_json"), {})
        row["result"] = self._json_loads_or(row.get("result_json"), {})
        row.pop("payload_json", None)
        row.pop("result_json", None)
        return row

    def add_group_knowledge_card(
        self,
        *,
        card_id: str,
        topic: str,
        normalized_question: str,
        fact_summary: str,
        citation_ids: list[str],
        quality_score: float,
    ) -> None:
        self.store.execute(
            """
            INSERT OR REPLACE INTO group_knowledge_card
            (card_id, topic, normalized_question, fact_summary, citation_ids, quality_score)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                card_id,
                topic,
                normalized_question,
                fact_summary,
                json.dumps(citation_ids, ensure_ascii=False),
                quality_score,
            ),
        )

    def _json_loads_or(self, payload: Any, default: Any) -> Any:
        if payload is None:
            return default
        if isinstance(payload, (list, dict)):
            return payload
        try:
            return json.loads(str(payload))
        except Exception:
            return default

    def close(self) -> None:
        self.store.close()

    # ----------------- Stock Universe -----------------
    def stock_universe_sync(self, token: str) -> dict[str, Any]:
        self.require_role(token, {"admin", "ops"})
        result = self.universe.sync_from_akshare()
        return {
            "status": "ok",
            "source": result.source,
            "total_stocks": result.total_stocks,
            "total_industry_links": result.total_industry_links,
        }

    def stock_universe_search(
        self,
        token: str,
        *,
        keyword: str = "",
        exchange: str = "",
        market_tier: str = "",
        listing_board: str = "",
        industry_l1: str = "",
        industry_l2: str = "",
        industry_l3: str = "",
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        _ = self.auth_me(token)
        return self.universe.search(
            keyword=keyword,
            exchange=exchange,
            market_tier=market_tier,
            listing_board=listing_board,
            industry_l1=industry_l1,
            industry_l2=industry_l2,
            industry_l3=industry_l3,
            limit=limit,
        )

    def stock_universe_filters(self, token: str) -> dict[str, list[str]]:
        _ = self.auth_me(token)
        return self.universe.filters()
