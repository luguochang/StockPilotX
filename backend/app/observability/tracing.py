from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TraceEvent:
    ts_ms: int
    name: str
    payload: dict[str, Any] = field(default_factory=dict)


class LangSmithAdapter:
    """LangSmith 适配器（可选启用）。"""

    def __init__(self) -> None:
        self.enabled = bool(os.getenv("LANGSMITH_API_KEY"))
        self._client = None
        if self.enabled:
            try:
                from langsmith import Client  # type: ignore

                self._client = Client()
            except Exception:
                self.enabled = False

    def send_event(self, trace_id: str, event: TraceEvent) -> None:
        if not self.enabled or not self._client:
            return
        # 轻量上报：以 feedback 形式记录事件快照
        try:
            self._client.create_feedback(
                run_id=trace_id,
                key=event.name,
                score=1.0,
                comment=str(event.payload)[:500],
            )
        except Exception:
            # 上报失败不影响主流程
            return


class TraceStore:
    """本地追踪 + LangSmith 上报双写。"""

    def __init__(self) -> None:
        self._events: dict[str, list[TraceEvent]] = {}
        self.langsmith = LangSmithAdapter()

    def new_trace(self) -> str:
        trace_id = str(uuid.uuid4())
        self._events[trace_id] = []
        return trace_id

    def emit(self, trace_id: str, name: str, payload: dict[str, Any] | None = None) -> None:
        if trace_id not in self._events:
            self._events[trace_id] = []
        event = TraceEvent(ts_ms=int(time.time() * 1000), name=name, payload=payload or {})
        self._events[trace_id].append(event)
        self.langsmith.send_event(trace_id, event)

    def list_events(self, trace_id: str) -> list[TraceEvent]:
        return list(self._events.get(trace_id, []))

