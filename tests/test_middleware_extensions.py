from __future__ import annotations

import unittest

from backend.app.config import Settings
from backend.app.middleware.hooks import (
    BudgetMiddleware,
    CacheMiddleware,
    MiddlewareStack,
    PIIMiddleware,
    RateLimitMiddleware,
)
from backend.app.state import AgentState


class MiddlewareExtensionsTestCase(unittest.TestCase):
    def test_rate_limit_blocks_excess_requests(self) -> None:
        stack = MiddlewareStack(
            [RateLimitMiddleware(max_requests=1, window_seconds=60)],
            settings=Settings(),
        )
        stack.run_before_agent(AgentState(user_id="u1", question="q1"))
        with self.assertRaises(RuntimeError):
            stack.run_before_agent(AgentState(user_id="u1", question="q2"))

    def test_cache_middleware_reuses_model_output(self) -> None:
        stack = MiddlewareStack(
            [CacheMiddleware(ttl_seconds=60), BudgetMiddleware()],
            settings=Settings(),
        )
        state = AgentState(user_id="u-cache", question="q")
        counter = {"n": 0}

        def fake_model(_state: AgentState, _prompt: str) -> str:
            counter["n"] += 1
            return f"resp-{counter['n']}"

        one = stack.call_model(state, "prompt-1", fake_model)
        two = stack.call_model(state, "prompt-1", fake_model)
        self.assertEqual(one, two)
        self.assertEqual(counter["n"], 1)

    def test_pii_middleware_redacts_prompt_and_output(self) -> None:
        stack = MiddlewareStack(
            [PIIMiddleware()],
            settings=Settings(),
        )
        state = AgentState(user_id="u-pii", question="q")
        prompt = "mail test@example.com phone 13800138000"
        redacted_prompt = stack.run_before_model(state, prompt)
        self.assertIn("[REDACTED_EMAIL]", redacted_prompt)
        self.assertIn("[REDACTED_PHONE]", redacted_prompt)
        output = stack.run_after_model(state, "身份证号 110101199001011234")
        self.assertIn("[REDACTED_ID]", output)


if __name__ == "__main__":
    unittest.main()
