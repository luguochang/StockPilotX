from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable


JobFn = Callable[[], dict[str, Any]]


@dataclass(slots=True)
class JobConfig:
    """调度任务配置。"""

    name: str
    cadence: str  # daily / intraday / weekly
    fn: JobFn
    max_retries: int = 3
    cooldown_seconds: int = 600


@dataclass(slots=True)
class JobState:
    """调度任务状态。"""

    failure_count: int = 0
    last_status: str = "never"
    last_error: str = ""
    last_run_at: str = ""
    circuit_open_until: str = ""
    paused: bool = False
    history: list[dict[str, Any]] | None = None


class LocalJobScheduler:
    """本地调度器（MVP）。

    说明：
    - 统一调度入口，后续可替换为 Temporal/Airflow 适配层。
    - 当前实现重试 + 熔断 + 状态查询，便于先满足工程化要求。
    """

    def __init__(self) -> None:
        self.jobs: dict[str, JobConfig] = {}
        self.states: dict[str, JobState] = {}

    def register(self, cfg: JobConfig) -> None:
        self.jobs[cfg.name] = cfg
        self.states.setdefault(cfg.name, JobState())

    def run_once(self, name: str) -> dict[str, Any]:
        if name not in self.jobs:
            return {"status": "not_found", "job": name}
        cfg = self.jobs[name]
        state = self.states[name]
        if state.history is None:
            state.history = []
        if state.paused:
            result = {"status": "paused", "job": name}
            state.history.append(result)
            return result

        now = datetime.now(timezone.utc)
        if state.circuit_open_until:
            until = datetime.fromisoformat(state.circuit_open_until)
            if now < until:
                return {"status": "circuit_open", "job": name, "retry_after": state.circuit_open_until}

        last_error = ""
        for attempt in range(1, cfg.max_retries + 1):
            try:
                payload = cfg.fn()
                state.failure_count = 0
                state.last_status = "ok"
                state.last_error = ""
                state.last_run_at = now.isoformat()
                state.circuit_open_until = ""
                result = {"status": "ok", "job": name, "attempt": attempt, "payload": payload}
                state.history.append(result)
                return result
            except Exception as ex:  # noqa: BLE001
                last_error = str(ex)

        # 全部重试失败，打开熔断
        state.failure_count += 1
        state.last_status = "failed"
        state.last_error = last_error
        state.last_run_at = now.isoformat()
        state.circuit_open_until = (now + timedelta(seconds=cfg.cooldown_seconds)).isoformat()
        result = {
            "status": "failed",
            "job": name,
            "error": last_error,
            "circuit_open_until": state.circuit_open_until,
        }
        state.history.append(result)
        return result

    def pause(self, name: str) -> dict[str, Any]:
        if name not in self.states:
            return {"status": "not_found", "job": name}
        self.states[name].paused = True
        return {"status": "ok", "job": name, "paused": True}

    def resume(self, name: str) -> dict[str, Any]:
        if name not in self.states:
            return {"status": "not_found", "job": name}
        self.states[name].paused = False
        return {"status": "ok", "job": name, "paused": False}

    def list_status(self) -> dict[str, dict[str, Any]]:
        status: dict[str, dict[str, Any]] = {}
        for name, cfg in self.jobs.items():
            state = self.states[name]
            status[name] = {
                "cadence": cfg.cadence,
                "failure_count": state.failure_count,
                "last_status": state.last_status,
                "last_error": state.last_error,
                "last_run_at": state.last_run_at,
                "circuit_open_until": state.circuit_open_until,
                "paused": state.paused,
                "history": (state.history or [])[-10:],
            }
        return status
