from __future__ import annotations

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

    # ----------------- Reports -----------------
    def save_report_index(self, *, report_id: str, user_id: int | None, tenant_id: int | None, stock_code: str, report_type: str, markdown: str) -> None:
        self.store.execute(
            """
            INSERT OR REPLACE INTO report_index (report_id, user_id, tenant_id, stock_code, report_type)
            VALUES (?, ?, ?, ?, ?)
            """,
            (report_id, user_id, tenant_id, stock_code, report_type),
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
            SELECT report_id, stock_code, report_type, created_at
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
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        _ = self.auth_me(token)
        return self.universe.search(
            keyword=keyword,
            exchange=exchange,
            market_tier=market_tier,
            listing_board=listing_board,
            industry_l1=industry_l1,
            limit=limit,
        )

    def stock_universe_filters(self, token: str) -> dict[str, list[str]]:
        _ = self.auth_me(token)
        return self.universe.filters()
