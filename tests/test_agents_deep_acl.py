from __future__ import annotations

import unittest

from backend.app.agents.tools import (
    LANGCHAIN_TOOLING_AVAILABLE,
    LangChainToolRunner,
    QuoteToolInput,
    ToolAccessController,
)
from backend.app.service import AShareAgentService


class AgentsDeepAclTestCase(unittest.TestCase):
    """AGT-001/AGT-002：Deep Agents 与工具 ACL 测试。"""

    def test_deep_agents_trigger(self) -> None:
        svc = AShareAgentService()
        result = svc.query(
            {
                "user_id": "deep-u1",
                "question": "请做多维并行的长期风险对比分析",
                "stock_codes": ["SH600000"],
            }
        )
        self.assertEqual(result["mode"], "agentic_rag")
        self.assertGreaterEqual(len(result["citations"]), 1)

    def test_tool_acl_denies_unauthorized_tool(self) -> None:
        acl = ToolAccessController()
        acl.register("quote_tool", lambda payload: payload)
        with self.assertRaises(RuntimeError):
            acl.call("router", "quote_tool", {"x": 1})

    def test_langchain_tool_runner_fallback(self) -> None:
        acl = ToolAccessController()
        acl.register("quote_tool", lambda payload: {"status": "ok", "queried": payload.get("stock_codes", [])})
        runner = LangChainToolRunner(acl)
        result = runner.call("data", "quote_tool", {"stock_codes": ["SH600000"]})
        self.assertEqual(result["status"], "ok")

    @unittest.skipUnless(LANGCHAIN_TOOLING_AVAILABLE, "langchain tooling not installed")
    def test_langchain_tool_runner_bound_call(self) -> None:
        acl = ToolAccessController()
        runner = LangChainToolRunner(acl)
        acl.register("quote_tool", lambda payload: {"status": "ok", "queried": payload.get("stock_codes", [])})
        runner.register("quote_tool", lambda payload: {"status": "ok", "queried": payload.get("stock_codes", [])}, "quote tool")
        result = runner.call("data", "quote_tool", {"stock_codes": ["SZ000001"]})
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["queried"], ["SZ000001"])

    @unittest.skipUnless(LANGCHAIN_TOOLING_AVAILABLE, "langchain tooling not installed")
    def test_langchain_tool_runner_bound_with_schema(self) -> None:
        acl = ToolAccessController()
        runner = LangChainToolRunner(acl)
        acl.register("quote_tool", lambda payload: {"status": "ok", "queried": payload.get("stock_codes", [])})
        runner.register(
            "quote_tool",
            lambda payload: {"status": "ok", "queried": payload.get("stock_codes", [])},
            "quote tool",
            args_schema=QuoteToolInput,
        )
        result = runner.call("data", "quote_tool", {"stock_codes": ["SH600000"]})
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["queried"], ["SH600000"])


if __name__ == "__main__":
    unittest.main()
