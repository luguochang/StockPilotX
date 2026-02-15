from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class WebStore:
    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS user (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS tenant (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS user_tenant_role (
                user_id INTEGER NOT NULL,
                tenant_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                PRIMARY KEY (user_id, tenant_id)
            );

            CREATE TABLE IF NOT EXISTS auth_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                action TEXT NOT NULL,
                success INTEGER NOT NULL,
                detail TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                tenant_id INTEGER NOT NULL,
                stock_code TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, tenant_id, stock_code)
            );

            CREATE TABLE IF NOT EXISTS dashboard_pref (
                user_id INTEGER NOT NULL,
                tenant_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY(user_id, tenant_id, key)
            );

            CREATE TABLE IF NOT EXISTS report_index (
                report_id TEXT PRIMARY KEY,
                user_id INTEGER,
                tenant_id INTEGER,
                stock_code TEXT NOT NULL,
                report_type TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS report_version (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                markdown TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS doc_index (
                doc_id TEXT PRIMARY KEY,
                user_id INTEGER,
                tenant_id INTEGER,
                filename TEXT NOT NULL,
                status TEXT NOT NULL,
                parse_confidence REAL DEFAULT 0,
                needs_review INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS doc_review_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                action TEXT NOT NULL,
                reviewer_user_id INTEGER,
                comment TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS source_health (
                source_id TEXT PRIMARY KEY,
                success_rate REAL NOT NULL,
                circuit_open INTEGER NOT NULL,
                last_error TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS alert_event (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS alert_ack (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS stock_universe (
                stock_code TEXT PRIMARY KEY,
                stock_name TEXT NOT NULL,
                exchange TEXT NOT NULL,
                market_tier TEXT NOT NULL DEFAULT '',
                listing_board TEXT NOT NULL,
                industry_l1 TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT '',
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS stock_universe_industry_map (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_code TEXT NOT NULL,
                industry_l1 TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT '',
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_stock_universe_name ON stock_universe(stock_name);
            CREATE INDEX IF NOT EXISTS idx_stock_universe_exchange ON stock_universe(exchange);
            CREATE INDEX IF NOT EXISTS idx_stock_universe_tier ON stock_universe(market_tier);
            CREATE INDEX IF NOT EXISTS idx_stock_universe_board ON stock_universe(listing_board);
            CREATE INDEX IF NOT EXISTS idx_stock_universe_industry ON stock_universe(industry_l1);
            CREATE INDEX IF NOT EXISTS idx_stock_industry_map_code ON stock_universe_industry_map(stock_code);
            CREATE INDEX IF NOT EXISTS idx_stock_industry_map_l1 ON stock_universe_industry_map(industry_l1);
            """
        )
        # 兼容历史库：补齐新增列
        self._ensure_column("stock_universe", "market_tier", "TEXT NOT NULL DEFAULT ''")
        self.conn.commit()

    def _ensure_column(self, table: str, column: str, sql_type: str) -> None:
        cols = self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        if any(str(c[1]) == column for c in cols):
            return
        self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        cur = self.conn.execute(sql, params)
        self.conn.commit()
        return cur

    def query_all(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        rows = self.conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def query_one(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        row = self.conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def close(self) -> None:
        self.conn.close()
