from __future__ import annotations

import json
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

    def deep_think_list_events(
        self,
        *,
        session_id: str,
        round_id: str | None = None,
        limit: int = 200,
        event_name: str | None = None,
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(2000, int(limit)))
        conditions = ["session_id = ?"]
        params: list[Any] = [session_id]
        if round_id:
            conditions.append("round_id = ?")
            params.append(round_id)
        if event_name:
            conditions.append("event_name = ?")
            params.append(event_name)
        params.append(safe_limit)
        sql = f"""
            SELECT session_id, round_id, round_no, event_seq, event_name, data_json, created_at
            FROM deep_think_event
            WHERE {' AND '.join(conditions)}
            ORDER BY round_no ASC, event_seq ASC, id ASC
            LIMIT ?
            """
        rows = self.store.query_all(sql, tuple(params))
        for row in rows:
            row["data"] = self._json_loads_or(row.get("data_json"), {})
            row.pop("data_json", None)
            row["event"] = row.pop("event_name", "message")
        return rows

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
