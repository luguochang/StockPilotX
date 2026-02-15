from __future__ import annotations

import unittest

from backend.app.agents.langgraph_runtime import (
    LANGGRAPH_AVAILABLE,
    DirectWorkflowRuntime,
    LangGraphWorkflowRuntime,
    build_workflow_runtime,
)
from backend.app.state import AgentState


class _DummyWorkflow:
    def prepare_prompt(self, state: AgentState, memory_hint=None) -> str:  # noqa: ANN001
        state.analysis["prepared"] = True
        return "dummy_prompt"

    def apply_before_model(self, state: AgentState, prompt: str) -> str:
        return f"{prompt}|before"

    def invoke_model(self, state: AgentState, prompt: str) -> str:
        return f"{prompt}|model"

    def apply_after_model(self, state: AgentState, output: str) -> str:
        return f"{output}|after"

    def finalize_with_output(self, state: AgentState, output: str) -> AgentState:
        state.report = output
        state.citations = [{"source_id": "t", "source_url": "u", "excerpt": "x", "reliability_score": 0.9}]
        return state

    def stream_model_collect(self, state: AgentState, prompt: str, chunk_size: int = 80) -> tuple[str, list[dict]]:  # noqa: ARG002
        output = f"{prompt}|stream"
        return output, [{"event": "answer_delta", "data": {"delta": output}}]


class LangGraphRuntimeTests(unittest.TestCase):
    def test_direct_runtime(self) -> None:
        wf = _DummyWorkflow()
        runtime = DirectWorkflowRuntime(wf)  # type: ignore[arg-type]
        state = AgentState(user_id="u1", question="q1", trace_id="t1")
        result = runtime.run(state, memory_hint=[{"content": "m"}])
        self.assertEqual(result.runtime, "direct")
        self.assertIn("|after", result.state.report)
        self.assertTrue(result.state.analysis.get("prepared"))

    def test_build_runtime_fallback(self) -> None:
        wf = _DummyWorkflow()
        runtime = build_workflow_runtime(wf, prefer_langgraph=False)  # type: ignore[arg-type]
        self.assertEqual(runtime.runtime_name, "direct")

    def test_direct_runtime_stream(self) -> None:
        wf = _DummyWorkflow()
        runtime = DirectWorkflowRuntime(wf)  # type: ignore[arg-type]
        state = AgentState(user_id="u1", question="q1", trace_id="t3")
        events = list(runtime.run_stream(state))
        names = [e.get("event") for e in events]
        self.assertIn("meta", names)
        self.assertIn("answer_delta", names)
        self.assertIn("done", names)

    @unittest.skipUnless(LANGGRAPH_AVAILABLE, "langgraph not installed")
    def test_langgraph_runtime(self) -> None:
        wf = _DummyWorkflow()
        runtime = LangGraphWorkflowRuntime(wf)  # type: ignore[arg-type]
        state = AgentState(user_id="u1", question="q1", trace_id="t2")
        result = runtime.run(state)
        self.assertEqual(result.runtime, "langgraph")
        self.assertIn("|before|model|after", result.state.report)
        self.assertEqual(len(result.state.citations), 1)

    @unittest.skipUnless(LANGGRAPH_AVAILABLE, "langgraph not installed")
    def test_langgraph_runtime_stream(self) -> None:
        wf = _DummyWorkflow()
        runtime = LangGraphWorkflowRuntime(wf)  # type: ignore[arg-type]
        state = AgentState(user_id="u1", question="q1", trace_id="t4")
        events = list(runtime.run_stream(state))
        names = [e.get("event") for e in events]
        self.assertIn("meta", names)
        self.assertIn("answer_delta", names)
        self.assertIn("done", names)


if __name__ == "__main__":
    unittest.main()
