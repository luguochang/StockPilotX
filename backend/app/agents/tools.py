from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, Field

try:
    from langchain_core.tools import StructuredTool

    LANGCHAIN_TOOLING_AVAILABLE = True
except Exception:  # pragma: no cover
    StructuredTool = None
    LANGCHAIN_TOOLING_AVAILABLE = False


ToolFn = Callable[[dict[str, Any]], dict[str, Any]]


class ToolPayloadSchema(BaseModel):
    """LangChain 工具统一 payload schema。"""

    payload: dict[str, Any] = Field(default_factory=dict)


class QuoteToolInput(BaseModel):
    stock_codes: list[str] = Field(default_factory=list)


class AnnouncementToolInput(BaseModel):
    stock_codes: list[str] = Field(default_factory=list)


class RetrieveToolInput(BaseModel):
    query: str = ""


class GraphToolInput(BaseModel):
    symbol: str = ""


class ToolAccessController:
    """工具权限控制（AGT-002）。"""

    def __init__(self) -> None:
        self.allowed: dict[str, set[str]] = {
            "router": {"intent_hint_tool"},
            "data": {"quote_tool", "announcement_tool"},
            "rag": {"retrieve_tool", "graph_tool"},
            "analysis": {"analysis_tool"},
            "report": {"format_tool"},
            "critic": {"citation_check_tool"},
        }
        self.tools: dict[str, ToolFn] = {}

    def register(self, name: str, fn: ToolFn) -> None:
        self.tools[name] = fn

    def call(self, role: str, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        if tool_name not in self.allowed.get(role, set()):
            raise RuntimeError(f"tool denied for role={role}: {tool_name}")
        fn = self.tools.get(tool_name)
        if not fn:
            raise RuntimeError(f"tool not found: {tool_name}")
        return fn(payload)


class LangChainToolRunner:
    """LangChain 结构化工具绑定层（可选启用）。"""

    def __init__(self, acl: ToolAccessController) -> None:
        self.acl = acl
        self.enabled = LANGCHAIN_TOOLING_AVAILABLE
        self.bound_tools: dict[str, Any] = {}

    def register(self, name: str, fn: ToolFn, description: str = "", args_schema: type[BaseModel] | None = None) -> None:
        if not self.enabled:
            return
        tool_desc = description or f"structured tool: {name}"
        if args_schema is None:
            tool = StructuredTool.from_function(  # type: ignore[union-attr]
                func=fn,
                name=name,
                description=tool_desc,
                args_schema=ToolPayloadSchema,
            )
        else:
            # 中文注释：将结构化参数统一转回 payload dict，复用现有工具函数实现。
            def wrapped(**kwargs):  # noqa: ANN003, ANN202
                return fn(dict(kwargs))

            tool = StructuredTool.from_function(  # type: ignore[union-attr]
                func=wrapped,
                name=name,
                description=tool_desc,
                args_schema=args_schema,
            )
        self.bound_tools[name] = tool

    def call(self, role: str, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        # 中文注释：始终先走 ACL 鉴权，防止结构化工具绕过权限边界。
        if tool_name not in self.acl.allowed.get(role, set()):
            raise RuntimeError(f"tool denied for role={role}: {tool_name}")
        if self.enabled and tool_name in self.bound_tools:
            tool = self.bound_tools[tool_name]
            args_schema = getattr(tool, "args_schema", None)
            if args_schema is ToolPayloadSchema:
                result = tool.invoke({"payload": payload})
            else:
                result = tool.invoke(payload)
            return result if isinstance(result, dict) else {"status": "ok", "result": result}
        # 回退原始 ACL 调用链，保证兼容性。
        return self.acl.call(role, tool_name, payload)
