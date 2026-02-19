from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any


class WebStore:
    def __init__(self, db_path: str) -> None:
        # 单连接跨线程访问时需串行化，避免 sqlite API misuse。
        self._lock = threading.RLock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            # 兼容历史库：老版本 stock_universe 缺少分层字段时，先补列再执行索引创建。
            self._pre_migrate_legacy_schema()
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

            CREATE TABLE IF NOT EXISTS watchlist_pool (
                pool_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                tenant_id INTEGER NOT NULL,
                pool_name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                is_default INTEGER NOT NULL DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS watchlist_pool_stock (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pool_id TEXT NOT NULL,
                stock_code TEXT NOT NULL,
                source_filters_json TEXT NOT NULL DEFAULT '{}',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(pool_id, stock_code)
            );

            CREATE TABLE IF NOT EXISTS dashboard_pref (
                user_id INTEGER NOT NULL,
                tenant_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY(user_id, tenant_id, key)
            );

            CREATE TABLE IF NOT EXISTS query_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                tenant_id INTEGER NOT NULL,
                question TEXT NOT NULL,
                stock_codes_json TEXT NOT NULL DEFAULT '[]',
                trace_id TEXT NOT NULL DEFAULT '',
                intent TEXT NOT NULL DEFAULT '',
                cache_hit INTEGER NOT NULL DEFAULT 0,
                latency_ms INTEGER NOT NULL DEFAULT 0,
                summary TEXT NOT NULL DEFAULT '',
                error TEXT NOT NULL DEFAULT '',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS report_index (
                report_id TEXT PRIMARY KEY,
                user_id INTEGER,
                tenant_id INTEGER,
                stock_code TEXT NOT NULL,
                report_type TEXT NOT NULL,
                run_id TEXT NOT NULL DEFAULT '',
                pool_snapshot_id TEXT NOT NULL DEFAULT '',
                template_id TEXT NOT NULL DEFAULT '',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS report_version (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                markdown TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                tenant_id INTEGER NOT NULL,
                portfolio_name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                initial_capital REAL NOT NULL,
                current_value REAL NOT NULL DEFAULT 0,
                total_profit_loss REAL NOT NULL DEFAULT 0,
                total_profit_loss_pct REAL NOT NULL DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS portfolio_position (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER NOT NULL,
                stock_code TEXT NOT NULL,
                quantity REAL NOT NULL DEFAULT 0,
                avg_cost REAL NOT NULL DEFAULT 0,
                current_price REAL NOT NULL DEFAULT 0,
                market_value REAL NOT NULL DEFAULT 0,
                profit_loss REAL NOT NULL DEFAULT 0,
                profit_loss_pct REAL NOT NULL DEFAULT 0,
                weight REAL NOT NULL DEFAULT 0,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(portfolio_id, stock_code)
            );

            CREATE TABLE IF NOT EXISTS portfolio_transaction (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER NOT NULL,
                stock_code TEXT NOT NULL,
                transaction_type TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                fee REAL NOT NULL DEFAULT 0,
                amount REAL NOT NULL,
                notes TEXT NOT NULL DEFAULT '',
                transaction_date DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS investment_journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                tenant_id INTEGER NOT NULL,
                journal_type TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                stock_code TEXT NOT NULL DEFAULT '',
                decision_type TEXT NOT NULL DEFAULT '',
                related_research_id TEXT NOT NULL DEFAULT '',
                related_portfolio_id INTEGER,
                tags_json TEXT NOT NULL DEFAULT '[]',
                sentiment TEXT NOT NULL DEFAULT '',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS journal_reflection (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                journal_id INTEGER NOT NULL,
                reflection_content TEXT NOT NULL,
                ai_insights TEXT NOT NULL DEFAULT '',
                lessons_learned TEXT NOT NULL DEFAULT '',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS journal_ai_reflection (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                journal_id INTEGER NOT NULL UNIQUE,
                status TEXT NOT NULL DEFAULT 'ready',
                summary TEXT NOT NULL DEFAULT '',
                insights_json TEXT NOT NULL DEFAULT '[]',
                lessons_json TEXT NOT NULL DEFAULT '[]',
                confidence REAL NOT NULL DEFAULT 0,
                provider TEXT NOT NULL DEFAULT '',
                model TEXT NOT NULL DEFAULT '',
                trace_id TEXT NOT NULL DEFAULT '',
                error_code TEXT NOT NULL DEFAULT '',
                error_message TEXT NOT NULL DEFAULT '',
                generated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
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

            CREATE TABLE IF NOT EXISTS doc_pipeline_run (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                status TEXT NOT NULL,
                filename TEXT NOT NULL DEFAULT '',
                parse_confidence REAL NOT NULL DEFAULT 0,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                table_count INTEGER NOT NULL DEFAULT 0,
                parse_notes TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
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

            CREATE TABLE IF NOT EXISTS alert_rule (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                tenant_id INTEGER NOT NULL,
                rule_name TEXT NOT NULL,
                rule_type TEXT NOT NULL,
                stock_code TEXT NOT NULL DEFAULT '',
                operator TEXT NOT NULL DEFAULT '',
                target_value REAL NOT NULL DEFAULT 0,
                event_type TEXT NOT NULL DEFAULT '',
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS alert_trigger_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_id INTEGER NOT NULL,
                stock_code TEXT NOT NULL,
                trigger_message TEXT NOT NULL,
                trigger_data_json TEXT NOT NULL DEFAULT '{}',
                triggered_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS stock_universe (
                stock_code TEXT PRIMARY KEY,
                stock_name TEXT NOT NULL,
                exchange TEXT NOT NULL,
                exchange_name TEXT NOT NULL DEFAULT '',
                market_tier TEXT NOT NULL DEFAULT '',
                listing_board TEXT NOT NULL,
                board_code TEXT NOT NULL DEFAULT '',
                industry_l1 TEXT NOT NULL DEFAULT '',
                industry_l2 TEXT NOT NULL DEFAULT '',
                industry_l3 TEXT NOT NULL DEFAULT '',
                is_active INTEGER NOT NULL DEFAULT 1,
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

            CREATE TABLE IF NOT EXISTS rag_eval_case (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT NOT NULL,
                positive_source_ids TEXT NOT NULL,
                predicted_source_ids TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS rag_doc_source_policy (
                source TEXT PRIMARY KEY,
                auto_approve INTEGER NOT NULL DEFAULT 0,
                trust_score REAL NOT NULL DEFAULT 0.6,
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS rag_doc_chunk (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                chunk_no INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                chunk_text_redacted TEXT NOT NULL,
                source TEXT NOT NULL,
                source_url TEXT NOT NULL DEFAULT '',
                stock_codes_json TEXT NOT NULL DEFAULT '[]',
                industry_tags_json TEXT NOT NULL DEFAULT '[]',
                effective_status TEXT NOT NULL DEFAULT 'review',
                quality_score REAL NOT NULL DEFAULT 0.0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS rag_qa_memory (
                memory_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                stock_code TEXT NOT NULL,
                query_text TEXT NOT NULL,
                answer_text TEXT NOT NULL,
                answer_redacted TEXT NOT NULL,
                summary_text TEXT NOT NULL,
                citations_json TEXT NOT NULL DEFAULT '[]',
                risk_flags_json TEXT NOT NULL DEFAULT '[]',
                intent TEXT NOT NULL DEFAULT 'fact',
                quality_score REAL NOT NULL DEFAULT 0.0,
                share_scope TEXT NOT NULL DEFAULT 'global',
                retrieval_enabled INTEGER NOT NULL DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS rag_qa_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL,
                signal TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS rag_retrieval_trace (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                query_text TEXT NOT NULL,
                query_type TEXT NOT NULL DEFAULT 'query',
                retrieved_ids_json TEXT NOT NULL DEFAULT '[]',
                selected_ids_json TEXT NOT NULL DEFAULT '[]',
                latency_ms INTEGER NOT NULL DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS rag_upload_asset (
                upload_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                source TEXT NOT NULL,
                source_url TEXT NOT NULL DEFAULT '',
                file_sha256 TEXT NOT NULL,
                file_size INTEGER NOT NULL DEFAULT 0,
                content_type TEXT NOT NULL DEFAULT '',
                stock_codes_json TEXT NOT NULL DEFAULT '[]',
                tags_json TEXT NOT NULL DEFAULT '[]',
                parse_note TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'uploaded',
                created_by TEXT NOT NULL DEFAULT '',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS rag_ops_meta (
                meta_key TEXT PRIMARY KEY,
                meta_value TEXT NOT NULL DEFAULT '',
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS deep_think_session (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                question TEXT NOT NULL,
                stock_codes TEXT NOT NULL,
                agent_profile TEXT NOT NULL,
                max_rounds INTEGER NOT NULL,
                current_round INTEGER NOT NULL DEFAULT 0,
                budget_json TEXT NOT NULL,
                mode TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'created',
                trace_id TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS deep_think_round (
                round_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                round_no INTEGER NOT NULL,
                status TEXT NOT NULL,
                consensus_signal TEXT NOT NULL,
                disagreement_score REAL NOT NULL,
                conflict_sources TEXT NOT NULL,
                counter_view TEXT NOT NULL DEFAULT '',
                task_graph TEXT NOT NULL DEFAULT '[]',
                replan_triggered INTEGER NOT NULL DEFAULT 0,
                stop_reason TEXT NOT NULL DEFAULT '',
                budget_usage TEXT NOT NULL DEFAULT '{}',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(session_id, round_no)
            );

            CREATE TABLE IF NOT EXISTS deep_think_opinion (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                round_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                signal TEXT NOT NULL,
                confidence REAL NOT NULL,
                reason TEXT NOT NULL,
                evidence_ids TEXT NOT NULL,
                risk_tags TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS deep_think_event (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                round_id TEXT NOT NULL,
                round_no INTEGER NOT NULL,
                event_seq INTEGER NOT NULL,
                event_name TEXT NOT NULL,
                data_json TEXT NOT NULL DEFAULT '{}',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(session_id, round_id, event_seq)
            );

            CREATE TABLE IF NOT EXISTS deep_think_export_task (
                task_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                status TEXT NOT NULL,
                format TEXT NOT NULL,
                filters_json TEXT NOT NULL DEFAULT '{}',
                filename TEXT NOT NULL DEFAULT '',
                media_type TEXT NOT NULL DEFAULT '',
                content_text TEXT NOT NULL DEFAULT '',
                row_count INTEGER NOT NULL DEFAULT 0,
                error TEXT NOT NULL DEFAULT '',
                attempt_count INTEGER NOT NULL DEFAULT 0,
                max_attempts INTEGER NOT NULL DEFAULT 2,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                completed_at DATETIME
            );

            CREATE TABLE IF NOT EXISTS deep_think_archive_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                action TEXT NOT NULL,
                status TEXT NOT NULL,
                duration_ms INTEGER NOT NULL DEFAULT 0,
                result_count INTEGER NOT NULL DEFAULT 0,
                export_bytes INTEGER NOT NULL DEFAULT 0,
                detail_json TEXT NOT NULL DEFAULT '{}',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS agent_card_registry (
                agent_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                description TEXT NOT NULL,
                capabilities TEXT NOT NULL,
                version TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS a2a_task (
                task_id TEXT PRIMARY KEY,
                session_id TEXT,
                agent_id TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                result_json TEXT NOT NULL DEFAULT '{}',
                trace_ref TEXT NOT NULL DEFAULT '',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS group_knowledge_card (
                card_id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                normalized_question TEXT NOT NULL,
                fact_summary TEXT NOT NULL,
                citation_ids TEXT NOT NULL,
                quality_score REAL NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_stock_universe_name ON stock_universe(stock_name);
            CREATE INDEX IF NOT EXISTS idx_stock_universe_exchange ON stock_universe(exchange);
            CREATE INDEX IF NOT EXISTS idx_stock_universe_tier ON stock_universe(market_tier);
            CREATE INDEX IF NOT EXISTS idx_stock_universe_board ON stock_universe(listing_board);
            CREATE INDEX IF NOT EXISTS idx_stock_universe_industry ON stock_universe(industry_l1);
            CREATE INDEX IF NOT EXISTS idx_stock_universe_industry_l2 ON stock_universe(industry_l2);
            CREATE INDEX IF NOT EXISTS idx_stock_universe_industry_l3 ON stock_universe(industry_l3);
            CREATE INDEX IF NOT EXISTS idx_watchlist_pool_user ON watchlist_pool(user_id, tenant_id, updated_at);
            CREATE INDEX IF NOT EXISTS idx_watchlist_pool_stock_pool ON watchlist_pool_stock(pool_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_watchlist_pool_stock_code ON watchlist_pool_stock(stock_code);
            CREATE INDEX IF NOT EXISTS idx_query_history_user_created ON query_history(user_id, tenant_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_portfolio_user_created ON portfolio(user_id, tenant_id, updated_at);
            CREATE INDEX IF NOT EXISTS idx_portfolio_position_pid ON portfolio_position(portfolio_id, market_value);
            CREATE INDEX IF NOT EXISTS idx_portfolio_tx_pid_date ON portfolio_transaction(portfolio_id, transaction_date);
            CREATE INDEX IF NOT EXISTS idx_investment_journal_user_created ON investment_journal(user_id, tenant_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_investment_journal_stock ON investment_journal(stock_code, created_at);
            CREATE INDEX IF NOT EXISTS idx_journal_reflection_journal ON journal_reflection(journal_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_journal_ai_reflection_generated ON journal_ai_reflection(generated_at);
            CREATE INDEX IF NOT EXISTS idx_alert_rule_user_active ON alert_rule(user_id, tenant_id, is_active, updated_at);
            CREATE INDEX IF NOT EXISTS idx_alert_trigger_rule_time ON alert_trigger_log(rule_id, triggered_at);
            CREATE INDEX IF NOT EXISTS idx_doc_pipeline_run_doc_created ON doc_pipeline_run(doc_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_doc_pipeline_run_stage_status ON doc_pipeline_run(stage, status, created_at);
            CREATE INDEX IF NOT EXISTS idx_stock_industry_map_code ON stock_universe_industry_map(stock_code);
            CREATE INDEX IF NOT EXISTS idx_stock_industry_map_l1 ON stock_universe_industry_map(industry_l1);
            CREATE INDEX IF NOT EXISTS idx_rag_doc_chunk_doc_status ON rag_doc_chunk(doc_id, effective_status, chunk_no);
            CREATE INDEX IF NOT EXISTS idx_rag_doc_chunk_source_status ON rag_doc_chunk(source, effective_status, updated_at);
            CREATE INDEX IF NOT EXISTS idx_rag_qa_memory_stock_retrieval ON rag_qa_memory(stock_code, retrieval_enabled, created_at);
            CREATE INDEX IF NOT EXISTS idx_rag_qa_feedback_memory ON rag_qa_feedback(memory_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_rag_retrieval_trace_trace_id ON rag_retrieval_trace(trace_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_rag_upload_asset_created_at ON rag_upload_asset(created_at);
            CREATE INDEX IF NOT EXISTS idx_rag_upload_asset_doc_status ON rag_upload_asset(doc_id, status, updated_at);
            CREATE INDEX IF NOT EXISTS idx_rag_upload_asset_sha ON rag_upload_asset(file_sha256, updated_at);
            CREATE INDEX IF NOT EXISTS idx_deep_think_round_session ON deep_think_round(session_id, round_no);
            CREATE INDEX IF NOT EXISTS idx_deep_think_opinion_round ON deep_think_opinion(round_id);
            CREATE INDEX IF NOT EXISTS idx_deep_think_event_session ON deep_think_event(session_id, round_no, event_seq);
            CREATE INDEX IF NOT EXISTS idx_deep_think_event_name ON deep_think_event(session_id, event_name, round_no, event_seq);
            CREATE INDEX IF NOT EXISTS idx_deep_think_export_task_session ON deep_think_export_task(session_id, status, created_at);
            CREATE INDEX IF NOT EXISTS idx_deep_think_export_task_queue ON deep_think_export_task(status, updated_at, created_at);
            CREATE INDEX IF NOT EXISTS idx_deep_think_archive_audit_action ON deep_think_archive_audit(action, created_at);
            CREATE INDEX IF NOT EXISTS idx_deep_think_archive_audit_session ON deep_think_archive_audit(session_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_a2a_task_agent_status ON a2a_task(agent_id, status);
            CREATE INDEX IF NOT EXISTS idx_group_knowledge_topic ON group_knowledge_card(topic, quality_score);
                """
            )
            # 兼容历史库：补齐新增列
            self._ensure_column("stock_universe", "market_tier", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column("stock_universe", "exchange_name", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column("stock_universe", "board_code", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column("stock_universe", "industry_l2", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column("stock_universe", "industry_l3", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column("stock_universe", "is_active", "INTEGER NOT NULL DEFAULT 1")
            self._ensure_column("report_index", "run_id", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column("report_index", "pool_snapshot_id", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column("report_index", "template_id", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column("deep_think_round", "task_graph", "TEXT NOT NULL DEFAULT '[]'")
            self._ensure_column("deep_think_round", "replan_triggered", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column("deep_think_round", "stop_reason", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column("deep_think_round", "budget_usage", "TEXT NOT NULL DEFAULT '{}'")
            self._ensure_column("deep_think_export_task", "attempt_count", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column("deep_think_export_task", "max_attempts", "INTEGER NOT NULL DEFAULT 2")
            self.conn.commit()

    def _table_exists(self, table: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
            (table,),
        ).fetchone()
        return row is not None

    def _pre_migrate_legacy_schema(self) -> None:
        if not self._table_exists("stock_universe"):
            return
        self._ensure_column("stock_universe", "market_tier", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("stock_universe", "exchange_name", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("stock_universe", "board_code", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("stock_universe", "industry_l2", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("stock_universe", "industry_l3", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("stock_universe", "is_active", "INTEGER NOT NULL DEFAULT 1")

    def _ensure_column(self, table: str, column: str, sql_type: str) -> None:
        cols = self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        if any(str(c[1]) == column for c in cols):
            return
        self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

    def query_all(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def query_one(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute(sql, params).fetchone()
            return dict(row) if row else None

    def close(self) -> None:
        with self._lock:
            self.conn.close()
