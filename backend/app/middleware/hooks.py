from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from backend.app.config import Settings
from backend.app.state import AgentState


ModelCall = Callable[[AgentState, str], str]
ToolCall = Callable[[str, dict[str, Any]], dict[str, Any]]


@dataclass(slots=True)
class MiddlewareContext:
    """中间件共享上下文。"""

    settings: Settings
    logs: list[str] = field(default_factory=list)
    model_call_count: int = 0
    tool_call_count: int = 0


class Middleware:
    """中间件基类。

    可类比 Java Filter/Interceptor + Around Advice。
    """

    name = "base"

    def before_agent(self, state: AgentState, ctx: MiddlewareContext) -> None:
        """Agent 主流程前置钩子。"""
        pass

    def before_model(self, state: AgentState, prompt: str, ctx: MiddlewareContext) -> str:
        """模型调用前，可改写 prompt。"""
        return prompt

    def after_model(self, state: AgentState, output: str, ctx: MiddlewareContext) -> str:
        """模型调用后，可改写输出。"""
        return output

    def after_agent(self, state: AgentState, ctx: MiddlewareContext) -> None:
        """Agent 主流程后置钩子。"""
        pass

    def wrap_model_call(self, state: AgentState, prompt: str, call_next: ModelCall, ctx: MiddlewareContext) -> str:
        """包裹模型调用（洋葱模型）。"""
        return call_next(state, prompt)

    def wrap_tool_call(
        self,
        tool_name: str,
        payload: dict[str, Any],
        call_next: ToolCall,
        ctx: MiddlewareContext,
    ) -> dict[str, Any]:
        """包裹工具调用（洋葱模型）。"""
        return call_next(tool_name, payload)


class GuardrailMiddleware(Middleware):
    """风控中间件：约束高风险输出。"""

    name = "guardrail"

    def before_agent(self, state: AgentState, ctx: MiddlewareContext) -> None:
        """在流程入口识别高风险投资请求。"""
        if "保证收益" in state.question or "确定买点" in state.question:
            state.risk_flags.append("high_risk_investment_request")
        ctx.logs.append("before_agent:guardrail")

    def before_model(self, state: AgentState, prompt: str, ctx: MiddlewareContext) -> str:
        """在 prompt 中追加安全规则。"""
        ctx.logs.append("before_model:guardrail")
        return prompt + "\n[RULE] 不得输出确定性投资建议。"

    def after_model(self, state: AgentState, output: str, ctx: MiddlewareContext) -> str:
        """在输出后做安全兜底。"""
        ctx.logs.append("after_model:guardrail")
        if "买入" in output and "仅供研究参考" not in output:
            output += "\n\n仅供研究参考，不构成投资建议。"
        return output

    def after_agent(self, state: AgentState, ctx: MiddlewareContext) -> None:
        """记录流程结束日志。"""
        ctx.logs.append("after_agent:guardrail")


class BudgetMiddleware(Middleware):
    """预算中间件：限制调用次数和上下文长度。"""

    name = "budget"

    def before_model(self, state: AgentState, prompt: str, ctx: MiddlewareContext) -> str:
        """截断超长 prompt，控制成本和延迟。"""
        ctx.logs.append("before_model:budget")
        max_chars = ctx.settings.max_context_chars
        if len(prompt) <= max_chars:
            return prompt
        return prompt[:max_chars]

    def wrap_model_call(self, state: AgentState, prompt: str, call_next: ModelCall, ctx: MiddlewareContext) -> str:
        """限制模型调用次数。"""
        if ctx.model_call_count >= ctx.settings.max_model_calls:
            raise RuntimeError("model call limit exceeded")
        ctx.model_call_count += 1
        return call_next(state, prompt)

    def wrap_tool_call(
        self, tool_name: str, payload: dict[str, Any], call_next: ToolCall, ctx: MiddlewareContext
    ) -> dict[str, Any]:
        """限制工具调用次数。"""
        if ctx.tool_call_count >= ctx.settings.max_tool_calls:
            raise RuntimeError("tool call limit exceeded")
        ctx.tool_call_count += 1
        return call_next(tool_name, payload)


class MiddlewareStack:
    """中间件执行栈。"""

    def __init__(self, middlewares: list[Middleware], settings: Settings) -> None:
        """初始化中间件列表和共享上下文。"""
        self.middlewares = middlewares
        self.ctx = MiddlewareContext(settings=settings)

    def run_before_agent(self, state: AgentState) -> None:
        """按注册顺序执行 `before_agent`。"""
        # 每次新请求都重置计数和日志，避免跨请求污染导致误触发预算上限。
        self.ctx.logs = []
        self.ctx.model_call_count = 0
        self.ctx.tool_call_count = 0
        for m in self.middlewares:
            m.before_agent(state, self.ctx)

    def run_after_agent(self, state: AgentState) -> None:
        """按逆序执行 `after_agent`。"""
        for m in reversed(self.middlewares):
            m.after_agent(state, self.ctx)

    def run_before_model(self, state: AgentState, prompt: str) -> str:
        """按顺序执行 `before_model` 并传递改写后的 prompt。"""
        value = prompt
        for m in self.middlewares:
            value = m.before_model(state, value, self.ctx)
        return value

    def run_after_model(self, state: AgentState, output: str) -> str:
        """按逆序执行 `after_model` 并传递改写后的输出。"""
        value = output
        for m in reversed(self.middlewares):
            value = m.after_model(state, value, self.ctx)
        return value

    def call_model(self, state: AgentState, prompt: str, model_call: ModelCall) -> str:
        """使用洋葱模型包裹并执行模型调用。"""
        call = model_call
        for m in reversed(self.middlewares):
            next_call = call

            def wrapped(s: AgentState, p: str, mm: Middleware = m, nc: ModelCall = next_call) -> str:
                return mm.wrap_model_call(s, p, nc, self.ctx)

            call = wrapped
        return call(state, prompt)

    def call_tool(self, tool_name: str, payload: dict[str, Any], tool_call: ToolCall) -> dict[str, Any]:
        """使用洋葱模型包裹并执行工具调用。"""
        call = tool_call
        for m in reversed(self.middlewares):
            next_call = call

            def wrapped(
                t: str, d: dict[str, Any], mm: Middleware = m, nc: ToolCall = next_call
            ) -> dict[str, Any]:
                return mm.wrap_tool_call(t, d, nc, self.ctx)

            call = wrapped
        return call(tool_name, payload)
