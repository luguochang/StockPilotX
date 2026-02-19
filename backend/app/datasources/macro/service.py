from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol

from backend.app.datasources.base.adapter import DataSourceConfig
from backend.app.datasources.macro.eastmoney_macro import EastmoneyMacroAdapter


class MacroAdapter(Protocol):
    source_id: str

    def fetch_macro_indicators(self, limit: int = 20) -> list[dict]:
        ...


class MockMacroAdapter:
    def __init__(self, source_id: str = "macro_mock", reliability_score: float = 0.58) -> None:
        self.source_id = source_id
        self.reliability_score = reliability_score

    def fetch_macro_indicators(self, limit: int = 20) -> list[dict]:
        now_iso = datetime.now(timezone.utc).isoformat()
        return [
            {
                "metric_name": "CPI",
                "metric_value": "2.1",
                "report_date": now_iso[:10],
                "event_time": now_iso,
                "source_id": self.source_id,
                "source_url": "https://macro.mock.example",
                "reliability_score": self.reliability_score,
            },
            {
                "metric_name": "PMI",
                "metric_value": "50.3",
                "report_date": now_iso[:10],
                "event_time": now_iso,
                "source_id": self.source_id,
                "source_url": "https://macro.mock.example",
                "reliability_score": self.reliability_score,
            },
        ][: max(1, limit)]


class MacroService:
    def __init__(self, adapters: list[MacroAdapter]) -> None:
        self.adapters = adapters

    def fetch_macro_indicators(self, limit: int = 20) -> list[dict]:
        errors: list[str] = []
        for adapter in self.adapters:
            try:
                rows = adapter.fetch_macro_indicators(limit=limit)
                if rows:
                    return rows
            except Exception as ex:  # noqa: BLE001
                errors.append(f"{getattr(adapter, 'source_id', 'unknown')}: {ex}")
        raise RuntimeError("all macro sources failed: " + "; ".join(errors))

    @classmethod
    def build_default(
        cls,
        *,
        timeout_seconds: float = 2.0,
        retry_count: int = 2,
        retry_backoff_seconds: float = 0.3,
        proxy_url: str = "",
    ) -> "MacroService":
        cfg = DataSourceConfig(
            source_id="eastmoney_macro",
            reliability_score=0.80,
            timeout_seconds=timeout_seconds,
            retry_count=retry_count,
            retry_backoff_seconds=retry_backoff_seconds,
            proxy_url=proxy_url,
        )
        adapters: list[MacroAdapter] = [EastmoneyMacroAdapter(cfg), MockMacroAdapter()]
        return cls(adapters)

