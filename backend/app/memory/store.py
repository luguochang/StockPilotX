from __future__ import annotations

import json
import sqlite3
from pathlib import Path
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
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.commit()

    def add_memory(self, user_id: str, memory_type: str, content: dict[str, Any]) -> int:
        """写入一条记忆并返回自增 ID。"""
        cur = self.conn.execute(
            "INSERT INTO long_term_memory (user_id, memory_type, content_json) VALUES (?, ?, ?)",
            (user_id, memory_type, json.dumps(content, ensure_ascii=False)),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def list_memory(self, user_id: str, memory_type: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        """按用户查询记忆，可选按类型过滤。"""
        if memory_type:
            rows = self.conn.execute(
                """
                SELECT id, user_id, memory_type, content_json, created_at
                FROM long_term_memory
                WHERE user_id = ? AND memory_type = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, memory_type, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT id, user_id, memory_type, content_json, created_at
                FROM long_term_memory
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()

        result: list[dict[str, Any]] = []
        for rid, uid, rtype, content_json, created_at in rows:
            result.append(
                {
                    "id": rid,
                    "user_id": uid,
                    "memory_type": rtype,
                    "content": json.loads(content_json),
                    "created_at": created_at,
                }
            )
        return result

    def close(self) -> None:
        self.conn.close()
