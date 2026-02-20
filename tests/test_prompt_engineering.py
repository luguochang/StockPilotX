from __future__ import annotations

import tempfile
import unittest

from backend.app.evals.service import EvalService
from backend.app.prompt.registry import PromptRegistry
from backend.app.prompt.runtime import PromptRuntime


class PromptEngineeringTestCase(unittest.TestCase):
    """PROMPT-001/002/003：模板装配、评测回写、门禁测试。"""

    def test_runtime_builds_three_layer_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            registry = PromptRegistry(f"{td}/prompt.db")
            runtime = PromptRuntime(registry)
            prompt, meta = runtime.build(
                "fact_qa",
                {"question": "测试问题", "stock_codes": ["SH600000"], "evidence": "source:cninfo"},
            )
            self.assertIn("[SYSTEM]", prompt)
            self.assertIn("[POLICY]", prompt)
            self.assertIn("[TASK]", prompt)
            self.assertEqual(meta["prompt_id"], "fact_qa")
            self.assertIn(meta["prompt_engine"], ("langchain_chat_prompt", "python_format"))
            prompt_v, meta_v = runtime.build_version(
                "fact_qa",
                "1.0.0",
                {"question": "版本回放", "stock_codes": ["SH600000"], "evidence": "source:cninfo"},
            )
            self.assertIn("[TASK]", prompt_v)
            self.assertEqual(meta_v["prompt_version"], "1.0.0")
            versions = registry.list_prompt_versions("fact_qa")
            self.assertTrue(any(v["version"] == "1.0.0" for v in versions))
            self.assertTrue(any(v["version"] == "1.1.0" for v in versions))
            registry.close()

    def test_eval_service_contains_prompt_suite_metrics(self) -> None:
        metrics = EvalService().run_eval(
            [
                {"fact_correct": True, "has_citation": True, "hallucination": False, "violation": False},
                {"fact_correct": True, "has_citation": True, "hallucination": False, "violation": False},
            ]
        )["metrics"]
        self.assertIn("prompt_total_pass_rate", metrics)
        self.assertIn("prompt_redteam_pass_rate", metrics)
        self.assertIn("prompt_freshness_timestamp_rate", metrics)
        self.assertIn("prompt_failed_case_count", metrics)
        self.assertIn("prompt_failed_case_ids", metrics)
        self.assertIn("prompt_group_stats", metrics)

    def test_stable_release_blocked_when_gate_failed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            registry = PromptRegistry(f"{td}/prompt.db")
            with self.assertRaises(ValueError):
                registry.create_release(
                    prompt_id="fact_qa",
                    version="1.0.0",
                    target_env="stable",
                    gate_result="fail",
                )
            registry.close()


if __name__ == "__main__":
    unittest.main()
