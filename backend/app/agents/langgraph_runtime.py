from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.app.agents.workflow import AgentWorkflow
from backend.app.state import AgentState

try:
    from langgraph.graph import END, StateGraph

    LANGGRAPH_AVAILABLE = True
except Exception:  # pragma: no cover
    END = None
    StateGraph = None
    LANGGRAPH_AVAILABLE = False


@dataclass(slots=True)
class RuntimeResult:
    state: AgentState
    runtime: str


class WorkflowRuntime:
    """工作流运行时抽象：便于 direct/langgraph 双实现切换。"""

    runtime_name = "direct"

    def run(self, state: AgentState, memory_hint: list[dict[str, Any]] | None = None) -> RuntimeResult:
        raise NotImplementedError

    def run_stream(self, state: AgentState, memory_hint: list[dict[str, Any]] | None = None):  # noqa: ANN201
        raise NotImplementedError


class DirectWorkflowRuntime(WorkflowRuntime):
    runtime_name = "direct"

    def __init__(self, workflow: AgentWorkflow) -> None:
        self.workflow = workflow

    def run(self, state: AgentState, memory_hint: list[dict[str, Any]] | None = None) -> RuntimeResult:
        prompt = self.workflow.prepare_prompt(state, memory_hint=memory_hint)
        prompt = self.workflow.apply_before_model(state, prompt)
        output = self.workflow.invoke_model(state, prompt)
        output = self.workflow.apply_after_model(state, output)
        result = self.workflow.finalize_with_output(state, output)
        return RuntimeResult(state=result, runtime=self.runtime_name)

    def run_stream(self, state: AgentState, memory_hint: list[dict[str, Any]] | None = None):  # noqa: ANN201
        prompt = self.workflow.prepare_prompt(state, memory_hint=memory_hint)
        prompt = self.workflow.apply_before_model(state, prompt)
        yield {"event": "meta", "data": {"trace_id": state.trace_id, "intent": state.intent, "mode": state.mode}}
        output, stream_events = self.workflow.stream_model_collect(state, prompt)
        for event in stream_events:
            yield event
        output = self.workflow.apply_after_model(state, output)
        self.workflow.finalize_with_output(state, output)
        yield {"event": "citations", "data": {"citations": state.citations}}
        yield {"event": "done", "data": {"ok": True, "trace_id": state.trace_id}}


class LangGraphWorkflowRuntime(WorkflowRuntime):
    """LangGraph 运行时：使用显式节点图编排 AgentWorkflow 阶段。"""

    runtime_name = "langgraph"

    def __init__(self, workflow: AgentWorkflow) -> None:
        if not LANGGRAPH_AVAILABLE:
            raise RuntimeError("langgraph is not installed")
        self.workflow = workflow
        self._graph = self._build_graph()
        self._stream_pre_graph = self._build_stream_pre_graph()
        self._stream_post_graph = self._build_stream_post_graph()

    def _build_graph(self):
        # 中文注释：graph state 只保存流程所需最小字段，避免大对象复制。
        graph = StateGraph(dict)

        def n_prepare(gstate: dict[str, Any]) -> dict[str, Any]:
            state: AgentState = gstate["state"]
            memory_hint = gstate.get("memory_hint")
            prompt = self.workflow.prepare_prompt(state, memory_hint=memory_hint)
            return {"state": state, "memory_hint": memory_hint, "prompt": prompt}

        def n_before_model(gstate: dict[str, Any]) -> dict[str, Any]:
            state: AgentState = gstate["state"]
            prompt = self.workflow.apply_before_model(state, str(gstate.get("prompt", "")))
            gstate["prompt"] = prompt
            return gstate

        def n_model(gstate: dict[str, Any]) -> dict[str, Any]:
            state: AgentState = gstate["state"]
            output = self.workflow.invoke_model(state, str(gstate.get("prompt", "")))
            gstate["output"] = output
            return gstate

        def n_after_model(gstate: dict[str, Any]) -> dict[str, Any]:
            state: AgentState = gstate["state"]
            output = self.workflow.apply_after_model(state, str(gstate.get("output", "")))
            gstate["output"] = output
            return gstate

        def n_finalize(gstate: dict[str, Any]) -> dict[str, Any]:
            state: AgentState = gstate["state"]
            final_state = self.workflow.finalize_with_output(state, str(gstate.get("output", "")))
            gstate["state"] = final_state
            return gstate

        graph.add_node("prepare", n_prepare)
        graph.add_node("before_model", n_before_model)
        graph.add_node("model", n_model)
        graph.add_node("after_model", n_after_model)
        graph.add_node("finalize", n_finalize)
        graph.set_entry_point("prepare")
        graph.add_edge("prepare", "before_model")
        graph.add_edge("before_model", "model")
        graph.add_edge("model", "after_model")
        graph.add_edge("after_model", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    def run(self, state: AgentState, memory_hint: list[dict[str, Any]] | None = None) -> RuntimeResult:
        result = self._graph.invoke({"state": state, "memory_hint": memory_hint})
        return RuntimeResult(state=result["state"], runtime=self.runtime_name)

    def _build_stream_pre_graph(self):
        graph = StateGraph(dict)

        def n_prepare(gstate: dict[str, Any]) -> dict[str, Any]:
            state: AgentState = gstate["state"]
            memory_hint = gstate.get("memory_hint")
            prompt = self.workflow.prepare_prompt(state, memory_hint=memory_hint)
            return {"state": state, "memory_hint": memory_hint, "prompt": prompt}

        def n_before_model(gstate: dict[str, Any]) -> dict[str, Any]:
            state: AgentState = gstate["state"]
            prompt = self.workflow.apply_before_model(state, str(gstate.get("prompt", "")))
            gstate["prompt"] = prompt
            return gstate

        graph.add_node("prepare", n_prepare)
        graph.add_node("before_model", n_before_model)
        graph.set_entry_point("prepare")
        graph.add_edge("prepare", "before_model")
        graph.add_edge("before_model", END)
        return graph.compile()

    def _build_stream_post_graph(self):
        graph = StateGraph(dict)

        def n_after_model(gstate: dict[str, Any]) -> dict[str, Any]:
            state: AgentState = gstate["state"]
            output = self.workflow.apply_after_model(state, str(gstate.get("output", "")))
            gstate["output"] = output
            return gstate

        def n_finalize(gstate: dict[str, Any]) -> dict[str, Any]:
            state: AgentState = gstate["state"]
            final_state = self.workflow.finalize_with_output(state, str(gstate.get("output", "")))
            gstate["state"] = final_state
            return gstate

        graph.add_node("after_model", n_after_model)
        graph.add_node("finalize", n_finalize)
        graph.set_entry_point("after_model")
        graph.add_edge("after_model", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    def run_stream(self, state: AgentState, memory_hint: list[dict[str, Any]] | None = None):  # noqa: ANN201
        # 中文注释：流式链路采用“前置图节点 + 模型流式 + 后置图节点”。
        pre = self._stream_pre_graph.invoke({"state": state, "memory_hint": memory_hint})
        state = pre["state"]
        prompt = str(pre.get("prompt", ""))

        yield {"event": "meta", "data": {"trace_id": state.trace_id, "intent": state.intent, "mode": state.mode}}
        output, stream_events = self.workflow.stream_model_collect(state, prompt)
        for event in stream_events:
            yield event

        post = self._stream_post_graph.invoke({"state": state, "output": output})
        state = post["state"]
        yield {"event": "citations", "data": {"citations": state.citations}}
        yield {"event": "done", "data": {"ok": True, "trace_id": state.trace_id}}


def build_workflow_runtime(workflow: AgentWorkflow, prefer_langgraph: bool = True) -> WorkflowRuntime:
    if prefer_langgraph and LANGGRAPH_AVAILABLE:
        return LangGraphWorkflowRuntime(workflow)
    return DirectWorkflowRuntime(workflow)
