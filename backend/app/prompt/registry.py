from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class PromptRegistry:
    """Prompt 资产管理。"""

    def __init__(self, db_path: str) -> None:
        """初始化数据库并准备默认 Prompt。"""
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
        self._seed_default_prompts()

    def _init_schema(self) -> None:
        """创建 Prompt 注册表和发布表。"""
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prompt_registry (
                prompt_id TEXT NOT NULL,
                version TEXT NOT NULL,
                scenario TEXT NOT NULL,
                template_system TEXT NOT NULL,
                template_policy TEXT NOT NULL,
                template_task TEXT NOT NULL,
                variables_schema TEXT NOT NULL,
                status TEXT NOT NULL,
                PRIMARY KEY (prompt_id, version)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prompt_release (
                release_id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id TEXT NOT NULL,
                version TEXT NOT NULL,
                target_env TEXT NOT NULL,
                gate_result TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prompt_eval_result (
                eval_run_id TEXT PRIMARY KEY,
                prompt_id TEXT NOT NULL,
                version TEXT NOT NULL,
                suite_id TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                pass_gate INTEGER NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.commit()

    def _seed_default_prompts(self) -> None:
        """若数据库为空，插入默认稳定 Prompt。"""
        exists = self.conn.execute(
            "SELECT COUNT(1) FROM prompt_registry WHERE prompt_id = 'fact_qa'"
        ).fetchone()[0]
        if exists:
            return

        self.upsert_prompt(
            {
                "prompt_id": "fact_qa",
                "version": "1.0.0",
                "scenario": "fact",
                "template_system": "你是A股研究助手。输出必须可追溯，并避免确定性投资建议。",
                "template_policy": "关键结论必须附引用；无法验证要明确不确定性。",
                "template_task": "问题：{question}\n股票：{stock_codes}\n证据：{evidence}",
                "variables_schema": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "stock_codes": {"type": "array"},
                        "evidence": {"type": "string"},
                    },
                    "required": ["question", "stock_codes", "evidence"],
                },
                "status": "stable",
            }
        )

    def upsert_prompt(self, payload: dict[str, Any]) -> None:
        """插入或更新一条 Prompt 版本。"""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO prompt_registry
            (prompt_id, version, scenario, template_system, template_policy, template_task, variables_schema, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["prompt_id"],
                payload["version"],
                payload["scenario"],
                payload["template_system"],
                payload["template_policy"],
                payload["template_task"],
                json.dumps(payload["variables_schema"], ensure_ascii=False),
                payload["status"],
            ),
        )
        self.conn.commit()

    def get_stable_prompt(self, prompt_id: str) -> dict[str, Any]:
        """读取指定 Prompt 的稳定版本。"""
        row = self.conn.execute(
            """
            SELECT prompt_id, version, scenario, template_system, template_policy, template_task, variables_schema, status
            FROM prompt_registry
            WHERE prompt_id = ? AND status = 'stable'
            ORDER BY version DESC
            LIMIT 1
            """,
            (prompt_id,),
        ).fetchone()
        if not row:
            raise KeyError(f"stable prompt not found: {prompt_id}")
        return {
            "prompt_id": row[0],
            "version": row[1],
            "scenario": row[2],
            "template_system": row[3],
            "template_policy": row[4],
            "template_task": row[5],
            "variables_schema": json.loads(row[6]),
            "status": row[7],
        }

    def get_prompt(self, prompt_id: str, version: str) -> dict[str, Any]:
        """读取指定 Prompt 的特定版本。"""
        row = self.conn.execute(
            """
            SELECT prompt_id, version, scenario, template_system, template_policy, template_task, variables_schema, status
            FROM prompt_registry
            WHERE prompt_id = ? AND version = ?
            LIMIT 1
            """,
            (prompt_id, version),
        ).fetchone()
        if not row:
            raise KeyError(f"prompt not found: {prompt_id}@{version}")
        return {
            "prompt_id": row[0],
            "version": row[1],
            "scenario": row[2],
            "template_system": row[3],
            "template_policy": row[4],
            "template_task": row[5],
            "variables_schema": json.loads(row[6]),
            "status": row[7],
        }

    def list_prompt_versions(self, prompt_id: str) -> list[dict[str, Any]]:
        """列出某个 Prompt 的所有版本。"""
        rows = self.conn.execute(
            """
            SELECT prompt_id, version, scenario, status
            FROM prompt_registry
            WHERE prompt_id = ?
            ORDER BY version DESC
            """,
            (prompt_id,),
        ).fetchall()
        return [
            {
                "prompt_id": r[0],
                "version": r[1],
                "scenario": r[2],
                "status": r[3],
            }
            for r in rows
        ]

    def save_eval_result(
        self,
        *,
        eval_run_id: str,
        prompt_id: str,
        version: str,
        suite_id: str,
        metrics: dict[str, Any],
        pass_gate: bool,
    ) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO prompt_eval_result
            (eval_run_id, prompt_id, version, suite_id, metrics_json, pass_gate)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (eval_run_id, prompt_id, version, suite_id, json.dumps(metrics, ensure_ascii=False), int(pass_gate)),
        )
        self.conn.commit()

    def create_release(self, *, prompt_id: str, version: str, target_env: str, gate_result: str) -> int:
        if target_env == "stable" and gate_result != "pass":
            raise ValueError("release gate failed: stable promotion is blocked")
        cur = self.conn.execute(
            """
            INSERT INTO prompt_release (prompt_id, version, target_env, gate_result)
            VALUES (?, ?, ?, ?)
            """,
            (prompt_id, version, target_env, gate_result),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def close(self) -> None:
        self.conn.close()
