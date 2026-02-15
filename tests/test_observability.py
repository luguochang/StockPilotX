from __future__ import annotations

import unittest

from backend.app.observability.tracing import TraceStore


class ObservabilityTestCase(unittest.TestCase):
    """OBS-001：本地 trace 与 LangSmith 适配降级测试。"""

    def test_trace_store_collects_events(self) -> None:
        ts = TraceStore()
        trace_id = ts.new_trace()
        ts.emit(trace_id, "step1", {"a": 1})
        ts.emit(trace_id, "step2", {"b": 2})
        events = ts.list_events(trace_id)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].name, "step1")


if __name__ == "__main__":
    unittest.main()

