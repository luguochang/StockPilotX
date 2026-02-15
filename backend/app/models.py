from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """问答请求 DTO（对应 `/v1/query`）。"""

    user_id: str = Field(min_length=1)
    question: str = Field(min_length=2)
    stock_codes: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """证据引用结构，确保结论可追溯。"""

    source_id: str
    source_url: str
    event_time: datetime | None = None
    reliability_score: float = 0.5
    excerpt: str


class QueryResponse(BaseModel):
    """问答响应 DTO。"""

    trace_id: str
    intent: Literal["fact", "deep", "doc_qa", "compare"]
    answer: str
    citations: list[Citation]
    risk_flags: list[str] = Field(default_factory=list)
    mode: Literal["agentic_rag", "graph_rag"] = "agentic_rag"
    workflow_runtime: str = "unknown"


class ReportRequest(BaseModel):
    """报告生成请求 DTO（对应 `/v1/report/generate`）。"""

    user_id: str
    stock_code: str
    period: str = "1y"
    report_type: Literal["fact", "research"] = "fact"


class ReportResponse(BaseModel):
    """报告生成结果 DTO。"""

    report_id: str
    trace_id: str
    markdown: str
    citations: list[Citation]


class IngestResponse(BaseModel):
    """数据摄取任务结果 DTO。"""

    task_name: str
    success_count: int
    failed_count: int
    details: list[dict[str, Any]] = Field(default_factory=list)


class EvalResponse(BaseModel):
    """评测任务结果 DTO。"""

    eval_run_id: str
    metrics: dict[str, float]
    pass_gate: bool


class PredictRunRequest(BaseModel):
    """预测任务请求 DTO（对应 `/v1/predict/run`）。"""

    stock_codes: list[str] = Field(default_factory=list)
    horizons: list[str] = Field(default_factory=lambda: ["5d", "20d"])
    as_of_date: str | None = None


class PredictHorizonResult(BaseModel):
    """单周期预测结果。"""

    horizon: str
    score: float
    expected_excess_return: float
    up_probability: float
    risk_tier: Literal["low", "medium", "high"]
    signal: Literal["strong_buy", "buy", "hold", "reduce", "strong_reduce"]


class PredictItem(BaseModel):
    """单股票预测结果。"""

    stock_code: str
    as_of_date: str
    horizons: list[PredictHorizonResult]
    factors: dict[str, float]
    source: dict[str, Any]


class PredictRunResponse(BaseModel):
    """预测任务响应 DTO。"""

    run_id: str
    trace_id: str
    as_of_date: str
    horizons: list[str]
    results: list[PredictItem]
