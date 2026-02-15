from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AgentState:
    """Agent 主流程上下文。

    可类比 Java 中贯穿多层调用的 Context 对象。
    """

    # 基础输入
    user_id: str
    question: str

    # 路由和检索计划
    intent: str = "fact"
    stock_codes: list[str] = field(default_factory=list)
    retrieval_plan: dict[str, Any] = field(default_factory=dict)
    # A-share market regime context shared by query/deep-think for strategy-aware prompting and post-processing.
    market_regime_context: dict[str, Any] = field(default_factory=dict)

    # 证据与风险
    evidence_pack: list[dict[str, Any]] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)

    # 追踪与模式
    trace_id: str = ""
    mode: str = "agentic_rag"

    # 推理中间产物与最终输出
    analysis: dict[str, Any] = field(default_factory=dict)
    report: str = ""
