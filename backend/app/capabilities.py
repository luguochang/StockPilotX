from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.app.agents.langgraph_runtime import LANGGRAPH_AVAILABLE
from backend.app.agents.tools import LANGCHAIN_TOOLING_AVAILABLE
from backend.app.config import Settings
from backend.app.prompt.runtime import LANGCHAIN_PROMPT_AVAILABLE


@dataclass(slots=True)
class CapabilityItem:
    key: str
    status: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"key": self.key, "status": self.status, "detail": self.detail}


def build_capability_snapshot(settings: Settings, *, workflow_runtime: str, llm_external_enabled: bool) -> dict[str, Any]:
    """构建技术点落地快照，供接口与文档核查复用。"""
    items: list[CapabilityItem] = [
        CapabilityItem("state_management", "implemented", "AgentState + scheduler state 已接入"),
        CapabilityItem("langsmith", "implemented_optional", "配置 LANGSMITH_API_KEY 后启用上报"),
        CapabilityItem("multi_agent", "implemented", "Router/Data/RAG/Analysis/Report/Critic"),
        CapabilityItem("long_term_memory", "implemented", "SQLite long_term_memory 已接入"),
        CapabilityItem(
            "langgraph",
            "implemented" if LANGGRAPH_AVAILABLE else "fallback_direct",
            f"runtime={workflow_runtime}",
        ),
        CapabilityItem("middleware", "implemented", "before/after + wrap_model/tool 洋葱模型"),
        CapabilityItem("deep_agents", "implemented", "多子任务并行检索合并"),
        CapabilityItem("doc_pipeline", "implemented", "上传/索引/复核队列已打通"),
        CapabilityItem("rag_agentic", "implemented", "HybridRetriever + AgenticRAG"),
        CapabilityItem("rag_graph", "implemented", "GraphRAG + Neo4j/InMemory 回退"),
        CapabilityItem(
            "langchain",
            "implemented_partial" if (LANGCHAIN_PROMPT_AVAILABLE or LANGCHAIN_TOOLING_AVAILABLE) else "not_available",
            f"prompt={LANGCHAIN_PROMPT_AVAILABLE}, tool_binding={LANGCHAIN_TOOLING_AVAILABLE}",
        ),
        CapabilityItem("prompt_engineering", "implemented", "registry/release/eval/gate 已接入"),
        CapabilityItem(
            "llm_gateway",
            "implemented_external" if llm_external_enabled else "implemented_local_fallback",
            "多 provider 网关 + 回退策略",
        ),
    ]
    return {
        "runtime": {
            "workflow_runtime": workflow_runtime,
            "langgraph_available": LANGGRAPH_AVAILABLE,
            "langchain_prompt_available": LANGCHAIN_PROMPT_AVAILABLE,
            "langchain_tooling_available": LANGCHAIN_TOOLING_AVAILABLE,
        },
        "config": {
            "use_langgraph_runtime": settings.use_langgraph_runtime,
            "llm_external_enabled": settings.llm_external_enabled,
            "llm_fallback_to_local": settings.llm_fallback_to_local,
        },
        "capabilities": [x.as_dict() for x in items],
    }
