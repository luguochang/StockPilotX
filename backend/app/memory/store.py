from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
from typing import Any


class MemoryStore:
    """长期记忆存储（SQLite）。"""

    def __init__(self, db_path: str) -> None:
        """初始化数据库连接并创建表结构。"""
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        """创建长期记忆表。"""
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS long_term_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                content_json TEXT NOT NULL,
                ttl_seconds INTEGER,
                expires_at DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # Lightweight migration for existing local DB files.
        existing_cols = {str(row[1]) for row in self.conn.execute("PRAGMA table_info(long_term_memory)").fetchall()}
        if "ttl_seconds" not in existing_cols:
            self.conn.execute("ALTER TABLE long_term_memory ADD COLUMN ttl_seconds INTEGER")
        if "expires_at" not in existing_cols:
            self.conn.execute("ALTER TABLE long_term_memory ADD COLUMN expires_at DATETIME")
        self.conn.commit()
        self._logger = logging.getLogger("memory.store")
        self._stats = {
            "similarity_queries": 0,
            "similarity_hit_queries": 0,
            "cleanup_runs": 0,
            "cleanup_deleted_rows": 0,
        }

    def add_memory(
        self,
        user_id: str,
        memory_type: str,
        content: dict[str, Any],
        ttl_seconds: int | None = None,
    ) -> int:
        """写入一条记忆并返回自增 ID。"""
        ttl_value = int(ttl_seconds) if ttl_seconds is not None else None
        expires_at = None
        if ttl_value is not None and ttl_value > 0:
            expires_at = (datetime.now(timezone.utc) + timedelta(seconds=ttl_value)).isoformat()
        cur = self.conn.execute(
            """
            INSERT INTO long_term_memory (user_id, memory_type, content_json, ttl_seconds, expires_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, memory_type, json.dumps(content, ensure_ascii=False), ttl_value, expires_at),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def list_memory(self, user_id: str, memory_type: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        """按用户查询记忆，可选按类型过滤。"""
        now_iso = datetime.now(timezone.utc).isoformat()
        if memory_type:
            rows = self.conn.execute(
                """
                SELECT id, user_id, memory_type, content_json, ttl_seconds, expires_at, created_at
                FROM long_term_memory
                WHERE user_id = ? AND memory_type = ?
                  AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, memory_type, now_iso, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT id, user_id, memory_type, content_json, ttl_seconds, expires_at, created_at
                FROM long_term_memory
                WHERE user_id = ?
                  AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, now_iso, limit),
            ).fetchall()

        result: list[dict[str, Any]] = []
        for rid, uid, rtype, content_json, ttl_seconds, expires_at, created_at in rows:
            result.append(
                {
                    "id": rid,
                    "user_id": uid,
                    "memory_type": rtype,
                    "content": json.loads(content_json),
                    "ttl_seconds": ttl_seconds,
                    "expires_at": expires_at,
                    "created_at": created_at,
                }
            )
        return result

    @staticmethod
    def _char_ngrams(text: str, n: int = 2) -> set[str]:
        cleaned = re.sub(r"\s+", "", text.lower())
        if len(cleaned) < n:
            return {cleaned} if cleaned else set()
        return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}

    @staticmethod
    def _content_to_text(content: dict[str, Any]) -> str:
        fields = []
        for key in ("question", "summary", "content", "answer", "text"):
            val = content.get(key)
            if isinstance(val, str) and val.strip():
                fields.append(val.strip())
        if not fields:
            fields.append(json.dumps(content, ensure_ascii=False))
        return " ".join(fields)

    def similarity_search(
        self,
        user_id: str,
        query: str,
        memory_type: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search memory by n-gram Jaccard similarity for lightweight semantic reuse."""
        query_ngrams = self._char_ngrams(str(query or ""))
        if not query_ngrams:
            return []
        rows = self.list_memory(user_id=user_id, memory_type=memory_type, limit=200)
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            content = row.get("content", {})
            if not isinstance(content, dict):
                continue
            memory_text = self._content_to_text(content)
            memory_ngrams = self._char_ngrams(memory_text)
            if not memory_ngrams:
                continue
            union = len(query_ngrams | memory_ngrams) or 1
            score = len(query_ngrams & memory_ngrams) / union
            if score <= 0:
                continue
            row_copy = dict(row)
            row_copy["similarity_score"] = round(score, 6)
            scored.append((score, row_copy))
        scored.sort(key=lambda x: x[0], reverse=True)
        result = [x[1] for x in scored[: max(1, int(top_k))]]
        self._stats["similarity_queries"] += 1
        if result:
            self._stats["similarity_hit_queries"] += 1
        self._logger.info(
            "memory_similarity_search user=%s memory_type=%s top_k=%s hit_count=%s",
            user_id,
            memory_type or "*",
            top_k,
            len(result),
        )
        return result

    def cleanup_expired(self) -> int:
        """Delete expired memory records and return deleted row count."""
        now_iso = datetime.now(timezone.utc).isoformat()
        cur = self.conn.execute("DELETE FROM long_term_memory WHERE expires_at IS NOT NULL AND expires_at <= ?", (now_iso,))
        deleted = int(cur.rowcount or 0)
        self.conn.commit()
        self._stats["cleanup_runs"] += 1
        self._stats["cleanup_deleted_rows"] += deleted
        self._logger.info("memory_cleanup_expired deleted=%s", deleted)
        return deleted

    def stats(self) -> dict[str, Any]:
        """Return hit-rate and cleanup counters for observability/reporting."""
        similarity_queries = int(self._stats.get("similarity_queries", 0))
        similarity_hit = int(self._stats.get("similarity_hit_queries", 0))
        hit_rate = float(similarity_hit / similarity_queries) if similarity_queries else 0.0
        return {
            "similarity_queries": similarity_queries,
            "similarity_hit_queries": similarity_hit,
            "similarity_hit_rate": round(hit_rate, 4),
            "cleanup_runs": int(self._stats.get("cleanup_runs", 0)),
            "cleanup_deleted_rows": int(self._stats.get("cleanup_deleted_rows", 0)),
        }

    def close(self) -> None:
        self.conn.close()
