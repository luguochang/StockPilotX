from __future__ import annotations

import os
from pathlib import Path
import unittest

from backend.app.config import Settings
from backend.app.service import AShareAgentService


class LlmFallbackTestCase(unittest.TestCase):
    """验证外部LLM不可用时，系统会自动回退本地模型。"""

    def setUp(self) -> None:
        self._old = {
            "LLM_EXTERNAL_ENABLED": os.getenv("LLM_EXTERNAL_ENABLED"),
            "LLM_CONFIG_PATH": os.getenv("LLM_CONFIG_PATH"),
            "LLM_FALLBACK_TO_LOCAL": os.getenv("LLM_FALLBACK_TO_LOCAL"),
        }

    def tearDown(self) -> None:
        for k, v in self._old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_fallback_to_local_when_external_unavailable(self) -> None:
        os.environ["LLM_EXTERNAL_ENABLED"] = "true"
        os.environ["LLM_FALLBACK_TO_LOCAL"] = "true"
        os.environ["LLM_CONFIG_PATH"] = str(Path("backend/config/not-exists.json"))

        svc = AShareAgentService(settings=Settings.from_env())
        result = svc.query(
            {
                "user_id": "u-fallback",
                "question": "请分析SH600000近期风险和机会",
                "stock_codes": ["SH600000"],
            }
        )

        self.assertIn("answer", result)
        self.assertIn("external_model_failed", result.get("risk_flags", []))


if __name__ == "__main__":
    unittest.main()

