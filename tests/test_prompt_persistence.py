from __future__ import annotations

import sqlite3
import tempfile
import unittest

from backend.app.config import Settings
from backend.app.service import AShareAgentService


class PromptPersistenceTestCase(unittest.TestCase):
    """PROMPT-002：评测结果回写与发布记录测试。"""

    def test_evals_run_writes_prompt_eval_and_release(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = Settings(
                memory_db_path=f"{td}/memory.db",
                prompt_db_path=f"{td}/prompt.db",
            )
            svc = AShareAgentService(settings=settings)
            result = svc.evals_run(
                [
                    {"fact_correct": True, "has_citation": True, "hallucination": False, "violation": False},
                    {"fact_correct": True, "has_citation": True, "hallucination": False, "violation": False},
                ]
            )
            self.assertIn("eval_run_id", result)

            conn = sqlite3.connect(settings.prompt_db_path)
            eval_count = conn.execute("SELECT COUNT(1) FROM prompt_eval_result").fetchone()[0]
            release_count = conn.execute("SELECT COUNT(1) FROM prompt_release").fetchone()[0]
            self.assertGreaterEqual(eval_count, 1)
            self.assertGreaterEqual(release_count, 1)
            conn.close()
            svc.close()


if __name__ == "__main__":
    unittest.main()
